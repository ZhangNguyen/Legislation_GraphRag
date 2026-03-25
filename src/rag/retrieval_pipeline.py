from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.rag.auto_filter import infer_filters, infer_query_profile
from src.rag.hybrid import bm25_score, tokenize
from src.rag.openai_clients import get_embedings
from src.rag.rerank_cross import cross_rerank

logger = logging.getLogger(__name__)

_DOCUMENT_TOP_K = 3
_PASSAGES_PER_DOC = 8
_FINAL_TOP_K = 8

_DOC_EMBED_CACHE: Dict[str, List[float]] = {}
_PASSAGE_EMBED_CACHE: Dict[str, List[float]] = {}
_QUESTION_EMBED_CACHE: Dict[str, List[float]] = {}


# =========================
# Basic helpers
# =========================

def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(_norm_space(text).encode("utf-8")).hexdigest()


def _graph_runtime_cache(graph: Dict[str, Any]) -> Dict[str, Any]:
    cache = graph.get("__retrieval_runtime_cache__")
    if isinstance(cache, dict):
        return cache
    cache = {}
    graph["__retrieval_runtime_cache__"] = cache
    return cache


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _node_index(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx = graph.get("node_index", {}) or {}
    if idx:
        return idx
    runtime_cache = _graph_runtime_cache(graph)
    cached = runtime_cache.get("node_index") or {}
    if cached:
        return cached
    out: Dict[str, Dict[str, Any]] = {}
    for node in graph.get("nodes", []) or []:
        node_id = str(node.get("node_id") or "").strip()
        if node_id:
            out[node_id] = node
    runtime_cache["node_index"] = out
    return out


def _artifact_type(node: Dict[str, Any]) -> str:
    return str((node.get("metadata", {}) or {}).get("artifact_type") or "evidence").strip().lower()


def _node_type(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or (node.get("metadata", {}) or {}).get("node_type") or "").strip().lower()


def _candidate_doc_key(metadata: Dict[str, Any]) -> str:
    for key in ["doc_id", "law_name", "official_title", "document_title", "file_stem"]:
        value = str(metadata.get(key) or "").strip().lower()
        if value:
            return value
    return "unknown"


def _path_label(md: Dict[str, Any]) -> str:
    for key in ["path_title", "title", "heading_title", "article", "clause", "point"]:
        value = md.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "-"


def _first_sentences(text: str, *, max_sentences: int = 2, max_chars: int = 420) -> str:
    clean = _norm_space(text)
    if not clean:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part and part.strip()]
    if not parts:
        return clean[:max_chars]
    merged = " ".join(parts[: max(1, int(max_sentences))]).strip()
    if len(merged) <= max_chars:
        return merged
    return merged[: max(0, max_chars - 3)].rstrip() + "..."


def _short(text: str, *, limit: int = 260) -> str:
    clean = _norm_space(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _batch_embed_texts(texts: List[str], *, cache: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    misses: List[Tuple[str, str]] = []
    for text in texts:
        key = _text_cache_key(text)
        if key in cache:
            out[key] = cache[key]
        else:
            misses.append((key, text))

    if misses:
        embeddings = get_embedings()
        vectors = embeddings.embed_documents([text for _, text in misses])
        for (key, _), vec in zip(misses, vectors):
            cache[key] = list(vec)
            out[key] = cache[key]
    return out


def _question_embedding(question: str) -> List[float]:
    q = _norm_space(question)
    key = _text_cache_key(q)
    cached = _QUESTION_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    embeddings = get_embedings()
    vec = list(embeddings.embed_query(q))
    _QUESTION_EMBED_CACHE[key] = vec
    return vec


def _bm25_value(question: str, text: str) -> float:
    return bm25_score(tokenize(question), tokenize(text))


def _sorted_ids_by_score(items: List[Dict[str, Any]], score_key: str) -> List[str]:
    ranked = sorted(items, key=lambda x: float(x.get(score_key, 0.0) or 0.0), reverse=True)
    return [str(item.get("id") or "") for item in ranked if str(item.get("id") or "")]


def _rrf_fuse(rank_lists: List[List[str]], *, k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, item_id in enumerate(ranked, start=1):
            if not item_id:
                continue
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


# =========================
# Graph catalog
# =========================

def _sorted_node_ids(node_ids: Iterable[str], node_idx: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(
        {str(nid) for nid in node_ids if str(nid)},
        key=lambda nid: int((node_idx.get(nid, {}).get("metadata", {}) or {}).get("order_index") or 0),
    )


def _build_doc_catalog(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    runtime_cache = _graph_runtime_cache(graph)
    cached = runtime_cache.get("doc_catalog")
    if isinstance(cached, list) and cached:
        return cached

    node_idx = _node_index(graph)
    buckets: Dict[str, Dict[str, Any]] = {}

    for node_id, node in node_idx.items():
        md = dict(node.get("metadata") or {})
        doc_key = _candidate_doc_key(md)
        if doc_key == "unknown":
            continue
        bucket = buckets.setdefault(
            doc_key,
            {
                "id": doc_key,
                "doc_key": doc_key,
                "doc_id": md.get("doc_id") or doc_key,
                "law_name": md.get("official_title") or md.get("law_name") or doc_key,
                "law_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "source": md.get("source") or md.get("issuing_agency") or "LocalFile",
                "year": int(md.get("year") or 0),
                "doc_number": md.get("doc_number") or "",
                "doc_sketch": "",
                "doc_sketch_node_id": None,
                "doc_sketch_source_node_ids": [],
                "article_bundle_ids": [],
                "evidence_ids": [],
                "summary_ids": [],
                "article_to_bundle_id": {},
                "article_to_evidence_ids": defaultdict(list),
                "member_node_ids": set(),
            },
        )
        bucket["member_node_ids"].add(node_id)

        artifact = _artifact_type(node)
        if artifact == "doc_sketch":
            bucket["doc_sketch"] = _norm_space(str(node.get("text") or ""))
            bucket["doc_sketch_node_id"] = node_id
            bucket["doc_sketch_source_node_ids"] = list(md.get("source_node_ids") or [])
        elif artifact == "article_bundle":
            bucket["article_bundle_ids"].append(node_id)
            article = str(md.get("article") or "").strip()
            if article and article not in bucket["article_to_bundle_id"]:
                bucket["article_to_bundle_id"][article] = node_id
        elif artifact == "evidence":
            bucket["evidence_ids"].append(node_id)
            article = str(md.get("article") or "").strip()
            if article:
                bucket["article_to_evidence_ids"][article].append(node_id)
        elif artifact == "summary":
            bucket["summary_ids"].append(node_id)

    out: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        bucket["article_bundle_ids"] = _sorted_node_ids(bucket["article_bundle_ids"], node_idx)
        bucket["evidence_ids"] = _sorted_node_ids(bucket["evidence_ids"], node_idx)
        bucket["summary_ids"] = _sorted_node_ids(bucket["summary_ids"], node_idx)
        bucket["member_node_ids"] = sorted(bucket["member_node_ids"])
        bucket["article_to_evidence_ids"] = {
            art: _sorted_node_ids(ids, node_idx) for art, ids in bucket["article_to_evidence_ids"].items()
        }
        if not bucket["doc_sketch"]:
            article_titles = []
            for bundle_id in bucket["article_bundle_ids"][:15]:
                md = dict((node_idx.get(bundle_id) or {}).get("metadata") or {})
                title = _path_label(md)
                if title:
                    article_titles.append(title)
            parts = []
            if bucket["law_type"]:
                parts.append(f"Loại văn bản: {bucket['law_type']}")
            if bucket["law_name"]:
                parts.append(f"Tiêu đề: {bucket['law_name']}")
            if bucket["doc_number"]:
                parts.append(f"Số văn bản: {bucket['doc_number']}")
            if article_titles:
                parts.append("Các điều chính:\n" + "\n".join(f"- {x}" for x in article_titles))
            bucket["doc_sketch"] = "\n".join(parts).strip()
        out.append(bucket)

    out.sort(key=lambda x: str(x.get("law_name") or x.get("doc_key") or ""))
    runtime_cache["doc_catalog"] = out
    return out


def _matches_filters(doc_or_md: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    for key in ["law_type", "doc_number"]:
        want = _norm_space(str(filters.get(key) or ""))
        if not want:
            continue
        have = _norm_space(str(doc_or_md.get(key) or doc_or_md.get(key.lower()) or ""))
        if want.lower() != have.lower():
            return False
    want_year = int(filters.get("year") or 0)
    if want_year and int(doc_or_md.get("year") or 0) != want_year:
        return False
    return True


# =========================
# Ranking helpers
# =========================

def _doc_boost(doc: Dict[str, Any], query_profile: Dict[str, Any]) -> float:
    boost = 0.0
    filters = query_profile.get("filters") or {}
    doc_number = _norm_space(str(filters.get("doc_number") or ""))
    if doc_number and doc_number.lower() == _norm_space(str(doc.get("doc_number") or "")).lower():
        boost += 0.35
    law_type = _norm_space(str(filters.get("law_type") or ""))
    if law_type and law_type.lower() == _norm_space(str(doc.get("law_type") or "")).lower():
        boost += 0.08
    return boost


def rank_documents_by_sketch_rrf(
    question: str,
    graph: Dict[str, Any],
    *,
    top_k: int = _DOCUMENT_TOP_K,
    question_vec: Optional[List[float]] = None,
    filters: Optional[Dict[str, Any]] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    docs = [d for d in _build_doc_catalog(graph) if _matches_filters(d, filters or {})]
    if not docs:
        return []

    qv = question_vec or _question_embedding(question)
    sketch_texts = [str(doc.get("doc_sketch") or "") for doc in docs]
    sketch_embeds = _batch_embed_texts(sketch_texts, cache=_DOC_EMBED_CACHE)

    dense_items: List[Dict[str, Any]] = []
    bm25_items: List[Dict[str, Any]] = []
    id_to_doc: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        doc_id = str(doc.get("id") or "")
        sketch_text = str(doc.get("doc_sketch") or "")
        embed = sketch_embeds.get(_text_cache_key(sketch_text), [])
        dense_score = _cosine(qv, embed)
        bm25_val = _bm25_value(question, sketch_text)
        boost = _doc_boost(doc, query_profile or {})
        dense_items.append({"id": doc_id, "dense_score": dense_score + boost})
        bm25_items.append({"id": doc_id, "bm25_score": bm25_val + boost})
        id_to_doc[doc_id] = doc

    fused = _rrf_fuse([
        _sorted_ids_by_score(dense_items, "dense_score"),
        _sorted_ids_by_score(bm25_items, "bm25_score"),
    ])

    ranked: List[Dict[str, Any]] = []
    for doc_id, doc in id_to_doc.items():
        sketch_text = str(doc.get("doc_sketch") or "")
        embed = sketch_embeds.get(_text_cache_key(sketch_text), [])
        dense_score = _cosine(qv, embed)
        bm25_val = _bm25_value(question, sketch_text)
        boost = _doc_boost(doc, query_profile or {})
        item = dict(doc)
        item["text"] = sketch_text
        item["dense_score"] = float(dense_score)
        item["bm25_score"] = float(bm25_val)
        item["rrf_score"] = float(fused.get(doc_id, 0.0) + boost)
        item["hybrid_score"] = item["rrf_score"]
        ranked.append(item)

    ranked.sort(key=lambda x: (float(x.get("rrf_score", 0.0)), float(x.get("dense_score", 0.0))), reverse=True)
    return ranked[: max(1, int(top_k or _DOCUMENT_TOP_K))]


def _field_dense_bm25(question: str, texts: List[str], question_vec: List[float]) -> Tuple[List[float], List[float]]:
    embeds = _batch_embed_texts(texts, cache=_PASSAGE_EMBED_CACHE)
    dense: List[float] = []
    bm25_vals: List[float] = []
    for text in texts:
        embed = embeds.get(_text_cache_key(text), [])
        dense.append(_cosine(question_vec, embed))
        bm25_vals.append(_bm25_value(question, text))
    return _minmax(dense), _minmax(bm25_vals)


def _passage_route_bonus(passage: Dict[str, Any], query_profile: Dict[str, Any]) -> float:
    profile = query_profile or {}
    route = str(profile.get("route") or "")
    md = dict(passage.get("metadata") or {})
    artifact = str(md.get("artifact_type") or "")
    heading = _norm_space(str(passage.get("heading_text") or "")).lower()
    self_text = _norm_space(str(passage.get("self_retrieval_text") or "")).lower()
    boost = 0.0

    if route == "heading_list":
        primary = str(profile.get("primary_heading_term") or "").lower()
        if artifact == "article_bundle":
            boost += 0.18
        if primary and primary in heading:
            boost += 0.20
        conflicting = [str(x).lower() for x in (profile.get("conflicting_heading_terms") or [])]
        if conflicting and any(term in heading for term in conflicting) and primary not in heading:
            boost -= 0.18

    if route == "version_change":
        terms = [str(x).lower() for x in (profile.get("change_terms") or [])]
        if any(term in heading or term in self_text for term in terms):
            boost += 0.18

    if route == "direct_reference":
        filters = profile.get("filters") or {}
        if filters.get("article") and _norm_space(str(filters.get("article"))).lower() == _norm_space(str(md.get("article") or "")).lower():
            boost += 0.12
        if filters.get("clause") and _norm_space(str(filters.get("clause"))).lower() == _norm_space(str(md.get("clause") or "")).lower():
            boost += 0.18
        if filters.get("point") and _norm_space(str(filters.get("point"))).lower() == _norm_space(str(md.get("point") or "")).lower():
            boost += 0.20

    if route == "condition_circumstance":
        if any(term in self_text for term in ["trường hợp", "khi", "nếu"]):
            boost += 0.06

    return boost


def _rank_passages_multifield(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    question_vec: Optional[List[float]] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not passages:
        return []

    qv = question_vec or _question_embedding(question)
    self_texts = [str(p.get("self_retrieval_text") or p.get("text") or "") for p in passages]
    heading_texts = [str(p.get("heading_text") or "") for p in passages]
    context_texts = [str(p.get("context_text") or "") for p in passages]

    self_dense, self_bm25 = _field_dense_bm25(question, self_texts, qv)
    head_dense, head_bm25 = _field_dense_bm25(question, heading_texts, qv)
    ctx_dense, ctx_bm25 = _field_dense_bm25(question, context_texts, qv)

    ranked: List[Dict[str, Any]] = []
    for idx, item in enumerate(passages):
        self_score = 0.60 * self_dense[idx] + 0.40 * self_bm25[idx]
        heading_score = 0.45 * head_dense[idx] + 0.55 * head_bm25[idx]
        context_score = 0.65 * ctx_dense[idx] + 0.35 * ctx_bm25[idx]
        boost = _passage_route_bonus(item, query_profile or {})
        enriched = dict(item)
        enriched["self_score"] = float(self_score)
        enriched["heading_score"] = float(heading_score)
        enriched["context_score"] = float(context_score)
        enriched["hybrid_score"] = float(0.55 * self_score + 0.30 * heading_score + 0.15 * context_score + boost)
        ranked.append(enriched)

    ranked.sort(
        key=lambda x: (
            float(x.get("hybrid_score", 0.0)),
            float(x.get("self_score", 0.0)),
            float(x.get("heading_score", 0.0)),
            float(x.get("context_score", 0.0)),
        ),
        reverse=True,
    )
    return ranked


# =========================
# Passage builders
# =========================

def _article_bundle_title(doc_info: Dict[str, Any], article: str, node_idx: Dict[str, Dict[str, Any]]) -> str:
    bundle_id = str((doc_info.get("article_to_bundle_id") or {}).get(article) or "")
    if not bundle_id:
        return article
    md = dict((node_idx.get(bundle_id) or {}).get("metadata") or {})
    return _path_label(md) or article


def _context_text_for_node(node: Dict[str, Any], doc_info: Dict[str, Any], graph: Dict[str, Any]) -> str:
    md = dict(node.get("metadata") or {})
    node_idx = _node_index(graph)
    parts: List[str] = []

    article = str(md.get("article") or "").strip()
    if article:
        article_title = _article_bundle_title(doc_info, article, node_idx)
        if article_title:
            parts.append(f"[Neo cha]\n{article_title}")

    child_ids = [str(x) for x in (md.get("children_ids") or []) if str(x)]
    child_titles: List[str] = []
    for cid in child_ids[:2]:
        child_md = dict((node_idx.get(cid) or {}).get("metadata") or {})
        title = _path_label(child_md)
        if title:
            child_titles.append(title)
    if child_titles:
        parts.append("[Neo con]\n" + "\n".join(child_titles))
    return "\n\n".join(parts).strip()


def _build_passage(node: Dict[str, Any], doc_info: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(node.get("metadata") or {})
    artifact = _artifact_type(node)
    doc_title = _norm_space(str(doc_info.get("law_name") or md.get("official_title") or md.get("law_name") or doc_info.get("doc_key") or ""))
    self_text = _norm_space(str(node.get("text") or ""))
    heading_text = _norm_space("\n".join(x for x in [str(md.get("path_title") or ""), str(md.get("heading_title") or ""), str(md.get("article") or ""), str(md.get("clause") or ""), str(md.get("point") or "")] if _norm_space(x)))
    context_text = _context_text_for_node(node, doc_info, graph)
    local_blocks = []
    if self_text:
        label = "Chính node" if artifact != "article_bundle" else "Nội dung điều"
        local_blocks.append(f"[{label}]\n{self_text}")
    if context_text:
        local_blocks.append(context_text)
    local_text = "\n\n".join(local_blocks)
    shared_text = f"[Văn bản] {doc_title}" if doc_title else ""

    if artifact == "article_bundle":
        rerank_label = "Nội dung điều"
    elif artifact == "doc_sketch":
        rerank_label = "Tóm tắt cấu trúc"
    else:
        rerank_label = "Chính node"

    rerank_text_short = "\n\n".join(
        part for part in [
            f"[Văn bản] {doc_title}" if doc_title else "",
            f"[Vị trí] {_path_label(md)} | artifact={artifact or _node_type(node) or '-'}",
            f"[{rerank_label}]\n{_short(self_text, limit=1500)}" if self_text else "",
            context_text and _short(context_text, limit=280),
        ] if _norm_space(part)
    )

    retrieval_text = "\n\n".join(
        part for part in [
            f"[Văn bản] {doc_title}" if doc_title else "",
            f"[Heading]\n{heading_text}" if heading_text else "",
            f"[Nội dung]\n{self_text}" if self_text else "",
            context_text,
        ] if _norm_space(part)
    )

    return {
        "node_id": str(node.get("node_id") or ""),
        "text": self_text,
        "snippet": _short(self_text, limit=700),
        "bundle_text": retrieval_text,
        "retrieval_text": retrieval_text,
        "rerank_text": rerank_text_short,
        "rerank_text_short": rerank_text_short,
        "self_retrieval_text": self_text,
        "heading_text": heading_text,
        "context_text": context_text,
        "shared_text": shared_text,
        "local_text": local_text,
        "metadata": {
            **md,
            "doc_key": doc_info.get("doc_key"),
            "law_name": doc_info.get("law_name") or md.get("law_name"),
            "law_type": doc_info.get("law_type") or md.get("law_type"),
            "source": doc_info.get("source") or md.get("source"),
        },
        "doc_key": str(doc_info.get("doc_key") or ""),
        "doc_score": float(doc_info.get("rrf_score", 0.0) or 0.0),
    }


def _build_doc_candidates(doc_info: Dict[str, Any], graph: Dict[str, Any], *, include_doc_sketch: bool = False) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    ids: List[str] = []
    if include_doc_sketch and doc_info.get("doc_sketch_node_id"):
        ids.append(str(doc_info.get("doc_sketch_node_id")))
    ids.extend(doc_info.get("article_bundle_ids") or [])
    ids.extend(doc_info.get("evidence_ids") or [])

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for nid in ids:
        if nid in seen or nid not in node_idx:
            continue
        seen.add(nid)
        out.append(_build_passage(node_idx[nid], doc_info, graph))
    return out


def _article_bundle_passage(doc_info: Dict[str, Any], article: str, graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    node_idx = _node_index(graph)
    bundle_id = str((doc_info.get("article_to_bundle_id") or {}).get(article) or "")
    if not bundle_id or bundle_id not in node_idx:
        return None
    return _build_passage(node_idx[bundle_id], doc_info, graph)


def _same_article_passages(doc_info: Dict[str, Any], article: str, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    ids = list((doc_info.get("article_to_evidence_ids") or {}).get(article) or [])
    out: List[Dict[str, Any]] = []
    for nid in ids:
        node = node_idx.get(str(nid))
        if not node:
            continue
        out.append(_build_passage(node, doc_info, graph))
    return out


def _find_doc_info(doc_groups: List[Dict[str, Any]], doc_key: str) -> Optional[Dict[str, Any]]:
    for doc in doc_groups:
        if str(doc.get("doc_key") or "") == str(doc_key or ""):
            return doc
    return None


# =========================
# Route-specific retrieval
# =========================

def _heading_list_route(question: str, graph: Dict[str, Any], doc_candidates: List[Dict[str, Any]], query_profile: Dict[str, Any], question_vec: List[float], final_top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    article_candidates: List[Dict[str, Any]] = []
    for doc in doc_candidates:
        for node_id in doc.get("article_bundle_ids") or []:
            node = _node_index(graph).get(str(node_id))
            if node:
                article_candidates.append(_build_passage(node, doc, graph))

    ranked_articles = _rank_passages_multifield(question, article_candidates, question_vec=question_vec, query_profile=query_profile)
    if not ranked_articles:
        return [], []

    top_article = ranked_articles[0]
    md = dict(top_article.get("metadata") or {})
    doc_info = _find_doc_info(doc_candidates, str(top_article.get("doc_key") or ""))
    article = str(md.get("article") or "").strip()

    final_passages: List[Dict[str, Any]] = [top_article]
    if doc_info and article:
        for p in _same_article_passages(doc_info, article, graph):
            if str(p.get("node_id") or "") == str(top_article.get("node_id") or ""):
                continue
            final_passages.append(p)

    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in final_passages:
        nid = str(item.get("node_id") or "")
        if nid in seen:
            continue
        seen.add(nid)
        deduped.append(item)
    return ranked_articles, deduped[:final_top_k]


def _version_change_route(question: str, graph: Dict[str, Any], doc_candidates: List[Dict[str, Any]], query_profile: Dict[str, Any], question_vec: List[float], cross_top_k: int, final_top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    change_terms = [str(x).lower() for x in (query_profile.get("change_terms") or [])]
    candidates: List[Dict[str, Any]] = []
    for doc in doc_candidates:
        for p in _build_doc_candidates(doc, graph):
            text = (_norm_space(str(p.get("self_retrieval_text") or "")).lower() + " " + _norm_space(str(p.get("heading_text") or "")).lower())
            if any(term in text for term in change_terms):
                candidates.append(p)
    if not candidates:
        for doc in doc_candidates:
            candidates.extend(_build_doc_candidates(doc, graph))
    ranked = _rank_passages_multifield(question, candidates, question_vec=question_vec, query_profile=query_profile)
    reranked = cross_rerank(question, ranked[:max(cross_top_k, 1)], top_n=min(max(cross_top_k, 1), len(ranked)), query_profile=query_profile)
    return ranked, reranked[:final_top_k]


def _generic_route(question: str, graph: Dict[str, Any], doc_candidates: List[Dict[str, Any]], query_profile: Dict[str, Any], question_vec: List[float], cross_top_k: int, final_top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filters = query_profile.get("filters") or {}
    want_article = _norm_space(str(filters.get("article") or ""))
    want_clause = _norm_space(str(filters.get("clause") or ""))
    want_point = _norm_space(str(filters.get("point") or ""))

    candidates: List[Dict[str, Any]] = []
    for doc in doc_candidates:
        for p in _build_doc_candidates(doc, graph):
            md = dict(p.get("metadata") or {})
            artifact = str(md.get("artifact_type") or "")
            if want_article:
                article = _norm_space(str(md.get("article") or ""))
                if artifact == "doc_sketch":
                    continue
                if want_clause or want_point:
                    if article.lower() != want_article.lower():
                        continue
                else:
                    if article.lower() != want_article.lower() and artifact != "article_bundle":
                        continue
            if want_clause and _norm_space(str(md.get("clause") or "")).lower() != want_clause.lower():
                continue
            if want_point and _norm_space(str(md.get("point") or "")).lower() != want_point.lower():
                continue
            candidates.append(p)

    if not candidates:
        for doc in doc_candidates:
            candidates.extend(_build_doc_candidates(doc, graph))

    ranked = _rank_passages_multifield(question, candidates, question_vec=question_vec, query_profile=query_profile)
    reranked = cross_rerank(question, ranked[:max(cross_top_k, 1)], top_n=min(max(cross_top_k, 1), len(ranked)), query_profile=query_profile)

    route = str(query_profile.get("route") or "")
    if reranked and route in {"direct_reference", "condition_circumstance", "factoid", "document_focus"}:
        top = reranked[0]
        md = dict(top.get("metadata") or {})
        article = str(md.get("article") or "").strip()
        artifact = str(md.get("artifact_type") or "")
        doc_info = _find_doc_info(doc_candidates, str(top.get("doc_key") or ""))
        extras: List[Dict[str, Any]] = []
        if doc_info and article and artifact != "article_bundle":
            bundle = _article_bundle_passage(doc_info, article, graph)
            if bundle:
                extras.append(bundle)
            if route == "condition_circumstance" and query_profile.get("wants_list_answer"):
                extras.extend(_same_article_passages(doc_info, article, graph))
        merged = []
        seen: Set[str] = set()
        for item in reranked + extras:
            nid = str(item.get("node_id") or "")
            if nid in seen:
                continue
            seen.add(nid)
            merged.append(item)
        reranked = merged

    return ranked, reranked[:final_top_k]


# =========================
# Main public API
# =========================

def retrieve_with_graph(
    question: str,
    graph: Dict[str, Any],
    *,
    filters: Optional[Dict[str, Any]] = None,
    qdrant_top_k: Optional[int] = None,
    graph_hops: int = 2,
    max_graph_nodes: int = 80,
    final_top_k: Optional[int] = None,
    cross_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    _ = qdrant_top_k
    _ = graph_hops
    _ = max_graph_nodes

    question = _norm_space(question)
    query_profile = infer_query_profile(question)
    merged_filters = dict(query_profile.get("filters") or {})
    for k, v in (filters or {}).items():
        if v not in (None, ""):
            merged_filters[k] = v
    if merged_filters:
        query_profile["filters"] = merged_filters

    question_vec = _question_embedding(question)
    doc_candidates = rank_documents_by_sketch_rrf(
        question,
        graph,
        top_k=_DOCUMENT_TOP_K,
        question_vec=question_vec,
        filters=merged_filters,
        query_profile=query_profile,
    )

    final_limit = max(1, int(final_top_k or _FINAL_TOP_K))
    cross_limit = max(1, int(cross_top_k or (_PASSAGES_PER_DOC * max(1, len(doc_candidates)))))

    route = str(query_profile.get("route") or "factoid")
    if route == "heading_list":
        candidate_pool, final_passages = _heading_list_route(question, graph, doc_candidates, query_profile, question_vec, final_limit)
        pipeline = "doc_sketch_rrf -> article_bundle_rank -> expand_same_article -> topk"
    elif route == "version_change":
        candidate_pool, final_passages = _version_change_route(question, graph, doc_candidates, query_profile, question_vec, cross_limit, final_limit)
        pipeline = "doc_sketch_rrf -> change_candidates -> multifield_hybrid -> cross_rerank -> topk"
    else:
        candidate_pool, final_passages = _generic_route(question, graph, doc_candidates, query_profile, question_vec, cross_limit, final_limit)
        pipeline = "doc_sketch_rrf -> top_docs -> provision/article candidates -> multifield_hybrid -> cross_rerank -> completion -> topk"

    for idx, passage in enumerate(final_passages, start=1):
        passage["rank"] = idx
        passage["final_score"] = float(
            passage.get("cross_score", passage.get("hybrid_score", 0.0)) or 0.0
        )

    doc_groups: List[Dict[str, Any]] = []
    for doc in doc_candidates:
        items = [p for p in final_passages if str(p.get("doc_key") or "") == str(doc.get("doc_key") or "")]
        doc_groups.append(
            {
                "doc_key": doc.get("doc_key"),
                "doc_score": float(doc.get("rrf_score", 0.0) or 0.0),
                "law_name": doc.get("law_name"),
                "source": doc.get("source"),
                "items": items,
                "doc_sketch": doc.get("doc_sketch"),
            }
        )

    return {
        "question": question,
        "mode": f"intent_route::{route}",
        "filters": merged_filters,
        "qdrant_top_k": None,
        "final_top_k": final_limit,
        "cross_top_k": cross_limit,
        "seed_candidates": doc_candidates,
        "doc_groups": doc_groups,
        "candidate_pool": candidate_pool,
        "passages": final_passages,
        "query_profile": {
            "pipeline": pipeline,
            "route": route,
            "document_top_k": _DOCUMENT_TOP_K,
            "passages_per_doc": _PASSAGES_PER_DOC,
            "uses_qdrant": False,
            "uses_graph_summary_nodes": False,
            "intent_terms": query_profile.get("intent_terms") or [],
            "filters": merged_filters,
        },
    }
