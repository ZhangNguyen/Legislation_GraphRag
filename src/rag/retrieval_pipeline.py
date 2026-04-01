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
                "official_title": md.get("official_title") or md.get("law_name") or doc_key,
                "law_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "doc_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "source": md.get("source") or md.get("issuing_agency") or "LocalFile",
                "year": int(md.get("year") or 0),
                "doc_number": md.get("doc_number") or "",
                "title_block": md.get("title_block") or "",
                "lead_block": md.get("lead_block") or "",
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
        if not bucket.get("doc_number") and md.get("doc_number"):
            bucket["doc_number"] = md.get("doc_number")
        if not bucket.get("law_name") and (md.get("official_title") or md.get("law_name")):
            bucket["law_name"] = md.get("official_title") or md.get("law_name")
        if not bucket.get("official_title") and (md.get("official_title") or md.get("law_name")):
            bucket["official_title"] = md.get("official_title") or md.get("law_name")
        if bucket.get("law_type") in (None, "", "Unknown") and (md.get("doc_type") or md.get("law_type")):
            bucket["law_type"] = md.get("doc_type") or md.get("law_type")
        if bucket.get("doc_type") in (None, "", "Unknown") and (md.get("doc_type") or md.get("law_type")):
            bucket["doc_type"] = md.get("doc_type") or md.get("law_type")
        if not bucket.get("title_block") and md.get("title_block"):
            bucket["title_block"] = md.get("title_block")
        if not bucket.get("lead_block") and md.get("lead_block"):
            bucket["lead_block"] = md.get("lead_block")
        if not bucket.get("source") and (md.get("source") or md.get("issuing_agency")):
            bucket["source"] = md.get("source") or md.get("issuing_agency")
        if not bucket.get("year") and md.get("year"):
            bucket["year"] = int(md.get("year") or 0)

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


DOC_NUMBER_ANY_RE = re.compile(r"\b0*(\d{1,4})\s*/\s*(\d{4})\s*/\s*([A-ZĐ\-][A-ZĐ\-\s]*)\b", re.IGNORECASE)


def _extract_doc_number(text: str) -> str:
    s = _norm_space(str(text or ""))
    if not s:
        return ""
    m = DOC_NUMBER_ANY_RE.search(s.upper())
    if not m:
        return ""
    first = str(int(m.group(1)))
    year = m.group(2)
    tail = re.sub(r"\s+", "", m.group(3).upper())
    return f"{first}/{year}/{tail}"


def _normalize_doc_number(text: str) -> str:
    s = _norm_space(str(text or "")).upper()
    if not s:
        return ""
    extracted = _extract_doc_number(s)
    if extracted:
        return extracted
    return re.sub(r"\s+", "", s)


def _normalize_law_type(text: str) -> str:
    raw = _norm_space(str(text or "")).lower()
    if not raw:
        return ""
    aliases = {
        "thông tư": "thông tư",
        "thong tu": "thông tư",
        "tt": "thông tư",
        "nghị định": "nghị định",
        "nghi dinh": "nghị định",
        "nd": "nghị định",
        "quyết định": "quyết định",
        "quyet dinh": "quyết định",
        "qd": "quyết định",
        "luật": "luật",
        "bo luat": "bộ luật",
        "bộ luật": "bộ luật",
        "nghị quyết": "nghị quyết",
        "nghi quyet": "nghị quyết",
    }
    return aliases.get(raw, raw)


def _infer_year_from_doc_number(text: str) -> int:
    norm = _normalize_doc_number(text) or _extract_doc_number(text)
    m = re.search(r"/(\d{4})/", norm)
    return int(m.group(1)) if m else 0


def _doc_filter_text(doc_or_md: Dict[str, Any]) -> str:
    parts = [
        str(doc_or_md.get("doc_number") or ""),
        str(doc_or_md.get("law_name") or ""),
        str(doc_or_md.get("official_title") or ""),
        str(doc_or_md.get("doc_sketch") or ""),
        str(doc_or_md.get("title_block") or ""),
        str(doc_or_md.get("lead_block") or ""),
        str(doc_or_md.get("law_type") or doc_or_md.get("doc_type") or ""),
    ]
    return _norm_space("\n".join(x for x in parts if _norm_space(x))).lower()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _matches_filters(doc_or_md: Dict[str, Any], filters: Dict[str, Any], *, relaxed: bool = False) -> bool:
    if not filters:
        return True

    haystack = _doc_filter_text(doc_or_md)
    haystack_compact = _compact(haystack)

    want_doc_number = _normalize_doc_number(filters.get("doc_number") or "")
    if want_doc_number:
        have_doc_number = _normalize_doc_number(doc_or_md.get("doc_number") or doc_or_md.get("doc_number".lower()) or "")
        if not have_doc_number:
            have_doc_number = _extract_doc_number(haystack)
        if have_doc_number:
            if relaxed:
                if want_doc_number != have_doc_number and want_doc_number not in have_doc_number and have_doc_number not in want_doc_number:
                    return False
            elif want_doc_number != have_doc_number:
                return False
        elif want_doc_number not in haystack_compact:
            return False

    want_law_type = _normalize_law_type(filters.get("law_type") or "")
    if want_law_type:
        have_law_type = _normalize_law_type(doc_or_md.get("law_type") or doc_or_md.get("law_type".lower()) or doc_or_md.get("doc_type") or "")
        if have_law_type:
            if not relaxed and want_law_type != have_law_type:
                return False
            if relaxed and want_law_type != have_law_type and want_law_type not in haystack:
                return False
        elif want_law_type not in haystack:
            return False

    want_year = int(filters.get("year") or 0)
    if want_year:
        have_year = int(doc_or_md.get("year") or 0)
        if not have_year:
            have_year = _infer_year_from_doc_number(doc_or_md.get("doc_number") or haystack)
        if have_year and have_year != want_year:
            return False

    return True


def _doc_boost(doc: Dict[str, Any], query_profile: Dict[str, Any]) -> float:
    boost = 0.0
    filters = query_profile.get("filters") or {}
    doc_number = _normalize_doc_number(filters.get("doc_number") or "")
    have_doc_number = _normalize_doc_number(doc.get("doc_number") or "") or _extract_doc_number(_doc_filter_text(doc))
    if doc_number and have_doc_number and doc_number == have_doc_number:
        boost += 0.38
    law_type = _normalize_law_type(filters.get("law_type") or "")
    have_law_type = _normalize_law_type(doc.get("law_type") or doc.get("doc_type") or "")
    if law_type and have_law_type and law_type == have_law_type:
        boost += 0.08
    want_year = int(filters.get("year") or 0)
    have_year = int(doc.get("year") or 0) or _infer_year_from_doc_number(doc.get("doc_number") or "")
    if want_year and have_year and want_year == have_year:
        boost += 0.04
    return boost


def _filter_docs_relaxed(docs: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not filters:
        return docs
    return [doc for doc in docs if _matches_filters(doc, filters, relaxed=True)]


def _doc_rank_fields(doc: Dict[str, Any]) -> Dict[str, str]:
    title = _norm_space(str(doc.get("official_title") or doc.get("law_name") or ""))
    sketch = _norm_space(str(doc.get("doc_sketch") or ""))
    doc_number = _norm_space(str(doc.get("doc_number") or ""))
    title_block = _norm_space(str(doc.get("title_block") or ""))
    lead_block = _first_sentences(str(doc.get("lead_block") or ""), max_sentences=3, max_chars=420)
    header = _norm_space("\n".join(x for x in [title_block, lead_block] if x))
    combined = _norm_space("\n".join(x for x in [doc_number, title, sketch, header] if x))
    return {
        "doc_number": doc_number,
        "title": title,
        "sketch": sketch,
        "header": header,
        "combined": combined,
    }


def _prefilter_docs_for_rescue(question: str, docs: List[Dict[str, Any]], filters: Dict[str, Any], *, limit: int = 64) -> List[Dict[str, Any]]:
    if not docs:
        return []
    want_doc_number = _normalize_doc_number(filters.get("doc_number") or "")
    want_law_type = _normalize_law_type(filters.get("law_type") or "")
    want_year = int(filters.get("year") or 0)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for doc in docs:
        fields = _doc_rank_fields(doc)
        score = 0.0
        combined = fields["combined"]
        title = fields["title"]
        sketch = fields["sketch"]
        score += 1.25 * _bm25_value(question, title)
        score += 1.00 * _bm25_value(question, sketch)
        score += 0.75 * _bm25_value(question, combined)
        have_doc_number = _normalize_doc_number(fields["doc_number"]) or _extract_doc_number(combined)
        if want_doc_number and have_doc_number == want_doc_number:
            score += 10.0
        elif want_doc_number and want_doc_number and want_doc_number in _compact(combined):
            score += 6.0
        have_law_type = _normalize_law_type(doc.get("law_type") or doc.get("doc_type") or "")
        if want_law_type and have_law_type == want_law_type:
            score += 1.5
        have_year = int(doc.get("year") or 0) or _infer_year_from_doc_number(fields["doc_number"])
        if want_year and have_year == want_year:
            score += 1.0
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: (x[0], len(str(x[1].get("doc_sketch") or ""))), reverse=True)
    return [doc for _, doc in scored[: max(1, int(limit or 1))]]


def _rank_documents_multifield_rrf(
    question: str,
    docs: List[Dict[str, Any]],
    *,
    top_k: int,
    question_vec: Optional[List[float]] = None,
    query_profile: Optional[Dict[str, Any]] = None,
    seed_stage: str = "strict",
) -> List[Dict[str, Any]]:
    if not docs:
        return []

    qv = question_vec or _question_embedding(question)
    fields_by_id: Dict[str, Dict[str, str]] = {}
    id_to_doc: Dict[str, Dict[str, Any]] = {}
    title_texts: List[str] = []
    sketch_texts: List[str] = []
    combined_texts: List[str] = []
    for doc in docs:
        doc_id = str(doc.get("id") or "")
        fields = _doc_rank_fields(doc)
        fields_by_id[doc_id] = fields
        id_to_doc[doc_id] = doc
        title_texts.append(fields["title"] or fields["combined"])
        sketch_texts.append(fields["sketch"] or fields["combined"])
        combined_texts.append(fields["combined"] or fields["title"] or fields["sketch"])

    title_embeds = _batch_embed_texts(title_texts, cache=_DOC_EMBED_CACHE)
    sketch_embeds = _batch_embed_texts(sketch_texts, cache=_DOC_EMBED_CACHE)
    combined_embeds = _batch_embed_texts(combined_texts, cache=_DOC_EMBED_CACHE)

    rank_lists: List[List[str]] = []
    scored_lists: List[Tuple[str, List[Dict[str, Any]]]] = []
    bm25_docno_items: List[Dict[str, Any]] = []
    bm25_title_items: List[Dict[str, Any]] = []
    bm25_sketch_items: List[Dict[str, Any]] = []
    bm25_header_items: List[Dict[str, Any]] = []
    dense_title_items: List[Dict[str, Any]] = []
    dense_sketch_items: List[Dict[str, Any]] = []
    dense_combined_items: List[Dict[str, Any]] = []

    for doc_id, doc in id_to_doc.items():
        fields = fields_by_id[doc_id]
        boost = _doc_boost(doc, query_profile or {})
        docno_text = fields["doc_number"] or fields["combined"]
        title_text = fields["title"] or fields["combined"]
        sketch_text = fields["sketch"] or fields["combined"]
        header_text = fields["header"] or fields["combined"]
        combined_text = fields["combined"] or title_text or sketch_text or header_text

        bm25_docno_items.append({"id": doc_id, "score": _bm25_value(question, docno_text) + 1.2 * boost})
        bm25_title_items.append({"id": doc_id, "score": _bm25_value(question, title_text) + 0.8 * boost})
        bm25_sketch_items.append({"id": doc_id, "score": _bm25_value(question, sketch_text) + 0.7 * boost})
        bm25_header_items.append({"id": doc_id, "score": _bm25_value(question, header_text) + 0.5 * boost})

        dense_title = _cosine(qv, title_embeds.get(_text_cache_key(title_text), []))
        dense_sketch = _cosine(qv, sketch_embeds.get(_text_cache_key(sketch_text), []))
        dense_combined = _cosine(qv, combined_embeds.get(_text_cache_key(combined_text), []))
        dense_title_items.append({"id": doc_id, "score": dense_title + 0.9 * boost})
        dense_sketch_items.append({"id": doc_id, "score": dense_sketch + 0.7 * boost})
        dense_combined_items.append({"id": doc_id, "score": dense_combined + 1.0 * boost})

    for items in [bm25_docno_items, bm25_title_items, bm25_sketch_items, bm25_header_items, dense_title_items, dense_sketch_items, dense_combined_items]:
        rank_lists.append(_sorted_ids_by_score(items, "score"))

    fused = _rrf_fuse(rank_lists)

    ranked: List[Dict[str, Any]] = []
    for doc_id, doc in id_to_doc.items():
        fields = fields_by_id[doc_id]
        title_text = fields["title"] or fields["combined"]
        sketch_text = fields["sketch"] or fields["combined"]
        combined_text = fields["combined"] or title_text or sketch_text
        dense_title = _cosine(qv, title_embeds.get(_text_cache_key(title_text), []))
        dense_sketch = _cosine(qv, sketch_embeds.get(_text_cache_key(sketch_text), []))
        dense_combined = _cosine(qv, combined_embeds.get(_text_cache_key(combined_text), []))
        bm25_docno = _bm25_value(question, fields["doc_number"] or combined_text)
        bm25_title = _bm25_value(question, title_text)
        bm25_sketch = _bm25_value(question, sketch_text)
        bm25_header = _bm25_value(question, fields["header"] or combined_text)
        boost = _doc_boost(doc, query_profile or {})
        item = dict(doc)
        item["text"] = combined_text
        item["dense_score"] = float(max(dense_title, dense_sketch, dense_combined))
        item["bm25_score"] = float(max(bm25_docno, bm25_title, bm25_sketch, bm25_header))
        item["rrf_score"] = float(fused.get(doc_id, 0.0) + boost)
        item["hybrid_score"] = item["rrf_score"]
        item["doc_seed_stage"] = seed_stage
        item["doc_rank_title"] = fields["title"]
        item["doc_rank_header"] = fields["header"]
        ranked.append(item)

    ranked.sort(key=lambda x: (float(x.get("rrf_score", 0.0)), float(x.get("dense_score", 0.0)), float(x.get("bm25_score", 0.0))), reverse=True)
    return ranked[: max(1, int(top_k or _DOCUMENT_TOP_K))]


def rank_documents_by_sketch_rrf(
    question: str,
    graph: Dict[str, Any],
    *,
    top_k: int = _DOCUMENT_TOP_K,
    question_vec: Optional[List[float]] = None,
    filters: Optional[Dict[str, Any]] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    all_docs = _build_doc_catalog(graph)
    active_filters = filters or {}

    strict_docs = [d for d in all_docs if _matches_filters(d, active_filters, relaxed=False)]
    if strict_docs:
        return _rank_documents_multifield_rrf(
            question, strict_docs, top_k=top_k, question_vec=question_vec, query_profile=query_profile, seed_stage="strict"
        )

    relaxed_docs = _filter_docs_relaxed(all_docs, active_filters) if active_filters else []
    if relaxed_docs:
        return _rank_documents_multifield_rrf(
            question, relaxed_docs, top_k=top_k, question_vec=question_vec, query_profile=query_profile, seed_stage="relaxed"
        )

    rescue_pool = _prefilter_docs_for_rescue(question, all_docs, active_filters, limit=max(32, 8 * max(1, int(top_k or _DOCUMENT_TOP_K))))
    if rescue_pool:
        return _rank_documents_multifield_rrf(
            question, rescue_pool, top_k=top_k, question_vec=question_vec, query_profile=query_profile, seed_stage="rescue_rrf"
        )

    return []


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
        if artifact == "evidence" and re.search(r"(?:^|\n)\s*(?:[a-zđ]\)|\d+\.)\s+", self_text):
            boost += 0.14
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
    context_sentences: List[str] = []

    article = str(md.get("article") or "").strip()
    if article:
        article_title = _article_bundle_title(doc_info, article, node_idx)
        if article_title:
            parts.append(f"[Neo cha]\n{article_title}")
        bundle_id = str((doc_info.get("article_to_bundle_id") or {}).get(article) or "")
        parent_node = node_idx.get(bundle_id) if bundle_id else None
        parent_text = _norm_space(str((parent_node or {}).get("text") or ""))
        if parent_text:
            context_sentences.append(_first_sentences(parent_text, max_sentences=1, max_chars=240))

    child_ids = [str(x) for x in (md.get("children_ids") or []) if str(x)]
    child_titles: List[str] = []
    for cid in child_ids[:2]:
        child_node = node_idx.get(cid) or {}
        child_md = dict(child_node.get("metadata") or {})
        title = _path_label(child_md)
        if title:
            child_titles.append(title)
        child_text = _norm_space(str(child_node.get("text") or ""))
        if child_text and len(context_sentences) < 2:
            context_sentences.append(_first_sentences(child_text, max_sentences=1, max_chars=220))
    if child_titles:
        parts.append("[Neo con]\n" + "\n".join(child_titles))

    if len(context_sentences) == 1:
        context_sentences[0] = _first_sentences(context_sentences[0], max_sentences=2, max_chars=320)
    if context_sentences:
        parts.append("[Ngữ cảnh cha-con]\n" + " ".join(context_sentences[:2]))
    return "\n\n".join(parts).strip()


def _build_passage(node: Dict[str, Any], doc_info: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(node.get("metadata") or {})
    artifact = _artifact_type(node)
    doc_title = _norm_space(str(doc_info.get("law_name") or md.get("official_title") or md.get("law_name") or doc_info.get("doc_key") or ""))
    official_title = _norm_space(str(md.get("official_title") or doc_info.get("official_title") or doc_title))
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
            f"[Official title] {official_title}" if official_title and official_title != doc_title else "",
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
            if query_profile.get("wants_list_answer"):
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
    doc_seed_stage = str((doc_candidates[0].get("doc_seed_stage") if doc_candidates else "none") or "none")

    final_limit = max(1, int(final_top_k or _FINAL_TOP_K))
    cross_limit = max(1, int(cross_top_k or (_PASSAGES_PER_DOC * max(1, len(doc_candidates)))))

    route = str(query_profile.get("route") or "factoid")
    if route == "heading_list":
        candidate_pool, final_passages = _heading_list_route(question, graph, doc_candidates, query_profile, question_vec, final_limit)
        pipeline = f"document_seed[{doc_seed_stage}] -> article_bundle_rank -> expand_same_article -> topk"
    elif route == "version_change":
        candidate_pool, final_passages = _version_change_route(question, graph, doc_candidates, query_profile, question_vec, cross_limit, final_limit)
        pipeline = f"document_seed[{doc_seed_stage}] -> change_candidates -> multifield_hybrid -> cross_rerank -> topk"
    else:
        candidate_pool, final_passages = _generic_route(question, graph, doc_candidates, query_profile, question_vec, cross_limit, final_limit)
        pipeline = f"document_seed[{doc_seed_stage}] -> provision/article candidates -> multifield_hybrid -> cross_rerank -> completion -> topk"

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
            "document_seed_stage": doc_seed_stage,
        },
    }
