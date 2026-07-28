from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.app.settings import settings
from src.rag.auto_filter import infer_query_profile
from src.rag.bm25_corpus import load_bm25_stats
from src.rag.hybrid import bm25_score, tokenize_vi
from src.rag.rerank_cross import cross_rerank
from src.storage.qdrant_store import get_qdrant_client, ensure_payload_indexes, search_qdrant

logger = logging.getLogger(__name__)

_QUESTION_EMBED_CACHE: Dict[str, List[float]] = {}
_DENSE_SEARCH_CACHE: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
_TOKEN_CACHE: Dict[str, List[str]] = {}
_BM25_RUNTIME: Optional[Dict[str, Any]] = None
REFERENCE_NODE_TYPES_EXTENDED = {
    "article",
    "clause",
    "point",
    "section",
    "appendix",
    "attachment",
    "roman_section",
    "alpha_section",
    "item",
    "decimal_item",
    "list_item",
    "bullet",
    "table",
    "table_row",
}

_CROSS_REF_RE = re.compile(
    r"(?:(điểm)\s+([a-zđ])\s+)?(?:(khoản)\s+(\d+)\s+)?(điều)\s+(\d+)",
    re.IGNORECASE,
)
_DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+\b", re.IGNORECASE)
_DOC_NUMBER_SHORT_RE = re.compile(r"\b\d{1,4}/[A-ZĐ\-]+\b", re.IGNORECASE)


def _norm_space(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _norm_key(text: Any) -> str:
    return _norm_space(text).lower()


def _norm_doc_number(text: Any) -> str:
    raw = _norm_space(text).upper().replace("Đ", "D")
    parts = raw.split("/")
    if parts and parts[0].isdigit():
        parts[0] = str(int(parts[0]))
    return "/".join(parts)


def _node_index(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx = graph.get("node_index", {}) or {}
    if idx:
        return idx
    out: Dict[str, Dict[str, Any]] = {}
    for node in list(graph.get("nodes") or []):
        node_id = str(node.get("node_id") or "").strip()
        if node_id:
            out[node_id] = node
    graph["node_index"] = out
    return out


def _node_metadata(node: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(node.get("metadata") or {})
    for key in ("node_id", "parent_id", "children_ids", "node_type", "text", "retrieval_text"):
        if key in node and key not in md:
            md[key] = node.get(key)
    return md


def _node_text(node: Dict[str, Any]) -> str:
    return _norm_space(node.get("retrieval_text") or node.get("text") or "")


def _candidate_id(item: Dict[str, Any]) -> str:
    return str(item.get("id") or item.get("node_id") or item.get("chunk_id") or "").strip()


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(_norm_space(text).encode("utf-8")).hexdigest()


def _tokens_for(text: str) -> List[str]:
    key = _text_cache_key(text)
    if key not in _TOKEN_CACHE:
        _TOKEN_CACHE[key] = tokenize_vi(text)
    return _TOKEN_CACHE[key]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = sum(float(x) * float(x) for x in a)
    nb = sum(float(y) * float(y) for y in b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def _question_embedding(question: str) -> List[float]:
    from src.rag.openai_clients import get_embedings

    key = _text_cache_key(question)
    if key not in _QUESTION_EMBED_CACHE:
        _QUESTION_EMBED_CACHE[key] = list(get_embedings().embed_query(_norm_space(question)))
    return _QUESTION_EMBED_CACHE[key]


def _question_embeddings(questions: List[str]) -> Dict[str, List[float]]:
    from src.rag.openai_clients import get_embedings

    normalized = [_norm_space(question) for question in questions if _norm_space(question)]
    out: Dict[str, List[float]] = {}
    misses: List[Tuple[str, str]] = []
    for question in normalized:
        key = _text_cache_key(question)
        if key in _QUESTION_EMBED_CACHE:
            out[question] = _QUESTION_EMBED_CACHE[key]
        else:
            misses.append((key, question))
    if misses:
        vectors = get_embedings().embed_documents([question for _, question in misses])
        for (key, question), vector in zip(misses, vectors):
            _QUESTION_EMBED_CACHE[key] = list(vector)
            out[question] = _QUESTION_EMBED_CACHE[key]
    return out


def _get_bm25_runtime() -> Dict[str, Any]:
    global _BM25_RUNTIME
    if _BM25_RUNTIME is not None:
        return _BM25_RUNTIME
    path = str(getattr(settings, "bm25_stats_path", "") or "").strip()
    if not path:
        _BM25_RUNTIME = {"idf": {}, "avgdl": 0.0}
        return _BM25_RUNTIME
    try:
        _BM25_RUNTIME = load_bm25_stats(path)
    except Exception as exc:
        logger.warning("Failed to load BM25 stats from %s: %s", path, exc)
        _BM25_RUNTIME = {"idf": {}, "avgdl": 0.0}
    return _BM25_RUNTIME


def _iter_graph_candidates(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node_id, node in _node_index(graph).items():
        md = _node_metadata(node)
        text = _node_text(node)
        if not text:
            continue
        artifact = str(md.get("artifact_type") or "evidence").strip().lower()
        node_type = str(md.get("node_type") or node.get("node_type") or "").strip().lower()
        if artifact not in {"evidence", "article_bundle", "section_bundle", "doc_sketch"} and node_type not in REFERENCE_NODE_TYPES_EXTENDED:
            continue
        out.append(
            {
                "id": node_id,
                "node_id": node_id,
                "chunk_id": str(md.get("chunk_id") or node_id),
                "doc_id": str(md.get("doc_id") or ""),
                "doc_key": str(md.get("doc_id") or ""),
                "text": text,
                "retrieval_text": _norm_space(node.get("retrieval_text") or text),
                "rerank_text_short": _norm_space(node.get("rerank_text") or node.get("retrieval_text") or text)[:1500],
                "metadata": md,
            }
        )
    return out


def analyze_query(question: str) -> Dict[str, Any]:
    profile = infer_query_profile(question) or {}
    filters = dict(profile.get("filters") or {})
    if not filters.get("doc_number"):
        m_short = _DOC_NUMBER_SHORT_RE.search(_norm_space(question).upper())
        if m_short:
            filters["doc_number"] = m_short.group(0).upper()
    query_type = "explicit_legal_reference" if profile.get("explicit_reference") else str(profile.get("route") or "factoid")
    out: Dict[str, Any] = {
        "question": _norm_space(question),
        "doc_number": filters.get("doc_number"),
        "law_type": filters.get("law_type"),
        "year": str(filters.get("year")) if filters.get("year") not in (None, "") else None,
        "article": filters.get("article"),
        "clause": filters.get("clause"),
        "point": filters.get("point"),
        "heading_terms": list(profile.get("heading_terms") or []),
        "change_terms": list(profile.get("change_terms") or []),
        "explicit_reference": bool(profile.get("explicit_reference")),
        "query_type": query_type,
        "route": profile.get("route"),
        "filters": filters,
    }
    return out


def _metadata_value(md: Dict[str, Any], key: str) -> Any:
    aliases = {
        "law_type": ("law_type", "doc_type"),
        "doc_number": ("doc_number",),
        "year": ("year",),
        "article": ("article",),
        "clause": ("clause",),
        "point": ("point",),
    }
    for candidate_key in aliases.get(key, (key,)):
        value = md.get(candidate_key)
        if value not in (None, ""):
            return value
    return None


def _matches_filter(item: Dict[str, Any], key: str, expected: Any) -> bool:
    if expected in (None, ""):
        return True
    md = dict(item.get("metadata") or {})
    actual = _metadata_value(md, key)
    if actual in (None, ""):
        return False
    if key == "doc_number":
        return _norm_doc_number(actual) == _norm_doc_number(expected)
    if key == "year":
        return str(actual) == str(expected)
    return _norm_key(actual) == _norm_key(expected)


def apply_metadata_filter(candidates: List[Dict[str, Any]], query_profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before_count = len(candidates)
    filters = dict(query_profile.get("filters") or {})
    for key in ("doc_number", "law_type", "year", "article", "clause", "point"):
        if query_profile.get(key) not in (None, ""):
            filters.setdefault(key, query_profile.get(key))

    qualifiers = [key for key in ("law_type", "year") if filters.get(key) not in (None, "")]
    levels: List[Tuple[str, List[str]]] = []
    if filters.get("doc_number"):
        for name, keys in (
            ("doc_number_article_clause_point", ["doc_number", "article", "clause", "point"]),
            ("doc_number_article_clause", ["doc_number", "article", "clause"]),
            ("doc_number_article", ["doc_number", "article"]),
            ("doc_number", ["doc_number"]),
        ):
            required = [key for key in keys if filters.get(key) not in (None, "")]
            levels.append((name, required))
    for name, keys in (
        ("article_clause_point", qualifiers + ["article", "clause", "point"]),
        ("article_clause", qualifiers + ["article", "clause"]),
        ("article", qualifiers + ["article"]),
        ("law_type_year", qualifiers),
    ):
        required = [key for key in keys if filters.get(key) not in (None, "")]
        if required:
            levels.append((name, required))

    seen_levels = set()
    deduped_levels: List[Tuple[str, List[str]]] = []
    for name, keys in levels:
        sig = tuple(keys)
        if not keys or sig in seen_levels:
            continue
        seen_levels.add(sig)
        deduped_levels.append((name, keys))

    for name, keys in deduped_levels:
        filtered = [item for item in candidates if all(_matches_filter(item, key, filters.get(key)) for key in keys)]
        if filtered:
            return filtered, {
                "applied_filters": keys,
                "fallback_level": name,
                "before_count": before_count,
                "after_count": len(filtered),
            }

    return candidates, {
        "applied_filters": [],
        "fallback_level": "no_hard_filter",
        "before_count": before_count,
        "after_count": len(candidates),
    }


def _qdrant_payload_to_passage(point: Any) -> Dict[str, Any]:
    payload = dict(getattr(point, "payload", None) or (point.get("payload", {}) if isinstance(point, dict) else {}) or {})
    md = dict(payload.get("metadata") or {})
    for key, value in payload.items():
        if key not in {"metadata", "vector"} and key not in md:
            md[key] = value
    point_id = str(getattr(point, "id", "") or (point.get("id", "") if isinstance(point, dict) else ""))
    node_id = str(payload.get("node_id") or md.get("node_id") or payload.get("chunk_id") or md.get("chunk_id") or point_id)
    text = _norm_space(payload.get("retrieval_text") or payload.get("text") or md.get("retrieval_text") or md.get("text") or "")
    score = float(getattr(point, "score", 0.0) or (point.get("score", 0.0) if isinstance(point, dict) else 0.0) or 0.0)
    return {
        "id": node_id,
        "node_id": node_id,
        "chunk_id": str(payload.get("chunk_id") or md.get("chunk_id") or node_id),
        "doc_id": str(payload.get("doc_id") or md.get("doc_id") or ""),
        "doc_key": str(payload.get("doc_id") or md.get("doc_id") or ""),
        "text": text,
        "retrieval_text": text,
        "rerank_text_short": _norm_space(payload.get("rerank_text") or md.get("rerank_text") or text)[:1500],
        "metadata": md,
        "dense_score": score,
    }


def _qdrant_filters_from_query(query_profile: Dict[str, Any], filter_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    filters = dict(query_profile.get("filters") or {})
    applied = list((filter_info or {}).get("applied_filters") or [])
    if not applied:
        applied = [key for key in ("doc_number", "law_type", "year", "article", "clause", "point") if filters.get(key) not in (None, "")]
    return {key: filters[key] for key in applied if filters.get(key) not in (None, "")}


def dense_search(
    question: str,
    candidates: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 50,
    *,
    query_profile: Optional[Dict[str, Any]] = None,
    filter_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    del candidates
    return dense_search_many(
        [question],
        top_k=top_k,
        query_profile=query_profile,
        filter_info=filter_info,
    ).get(_norm_space(question), [])


def _dense_cache_key(question: str, top_k: int, filters: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        hashlib.sha1(_norm_space(question).encode("utf-8")).hexdigest(),
        int(top_k or 0),
        json.dumps(filters or {}, ensure_ascii=False, sort_keys=True, default=str),
    )


def dense_search_many(
    questions: List[str],
    *,
    top_k: int = 50,
    query_profile: Optional[Dict[str, Any]] = None,
    filter_info: Optional[Dict[str, Any]] = None,
    max_workers: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    normalized: List[str] = []
    seen = set()
    for question in questions:
        clean = _norm_space(question)
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    if not normalized:
        return {}

    qdrant_filters = _qdrant_filters_from_query(query_profile or {}, filter_info)
    results: Dict[str, List[Dict[str, Any]]] = {}
    misses: List[str] = []
    for question in normalized:
        key = _dense_cache_key(question, top_k, qdrant_filters)
        cached = _DENSE_SEARCH_CACHE.get(key)
        if cached is not None:
            results[question] = [dict(item) for item in cached]
        else:
            misses.append(question)
    if not misses:
        return results

    try:
        vectors_by_question = _question_embeddings(misses)
        client = get_qdrant_client()
        if qdrant_filters:
            ensure_payload_indexes(client)

        def run_one(question: str) -> Tuple[str, List[Dict[str, Any]]]:
            points = search_qdrant(client, vectors_by_question[question], top_k=top_k, filters=qdrant_filters)
            ranked = [_qdrant_payload_to_passage(point) for point in points]
            ranked.sort(key=lambda x: float(x.get("dense_score", 0.0)), reverse=True)
            for rank, item in enumerate(ranked[:top_k], start=1):
                item["rank"] = rank
            return question, ranked[:top_k]

        worker_count = max(1, min(int(max_workers or 1), len(misses)))
        if worker_count == 1:
            completed = [run_one(question) for question in misses]
        else:
            completed = []
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(run_one, question) for question in misses]
                for future in as_completed(futures):
                    completed.append(future.result())

        for question, ranked in completed:
            key = _dense_cache_key(question, top_k, qdrant_filters)
            _DENSE_SEARCH_CACHE[key] = [dict(item) for item in ranked]
            results[question] = ranked
    except Exception as exc:
        logger.warning(
            "Dense search skipped after Qdrant/OpenAI failure: url=%s collection=%s top_k=%s filters=%s queries=%s error=%s",
            getattr(settings, "qdrant_url", ""),
            getattr(settings, "qdrant_collection", ""),
            top_k,
            qdrant_filters,
            len(misses),
            exc,
        )
        for question in misses:
            results.setdefault(question, [])

    return results


def bm25_search_vi(question: str, candidates: List[Dict[str, Any]], top_k: int = 50) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    stats = _get_bm25_runtime()
    query_tokens = _tokens_for(question)
    ranked: List[Dict[str, Any]] = []
    for item in candidates:
        text = str(item.get("retrieval_text") or item.get("text") or "")
        scored = dict(item)
        scored["bm25_score"] = bm25_score(
            query_tokens,
            _tokens_for(text),
            idf_map=dict(stats.get("idf") or {}),
            avgdl=float(stats.get("avgdl") or 0.0),
        )
        ranked.append(scored)
    ranked.sort(key=lambda x: float(x.get("bm25_score", 0.0)), reverse=True)
    for rank, item in enumerate(ranked[:top_k], start=1):
        item["rank"] = rank
    return ranked[:top_k]


def rrf_fusion(
    rank_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
    top_k: int = 30,
    *,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    weights = [dense_weight, bm25_weight] + [1.0] * max(0, len(rank_lists) - 2)
    buckets: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    ranks_by_source: Dict[str, Dict[str, int]] = {}

    for list_idx, rank_list in enumerate(rank_lists):
        weight = float(weights[list_idx] if list_idx < len(weights) else 1.0)
        for fallback_rank, item in enumerate(rank_list, start=1):
            item_id = _candidate_id(item)
            if not item_id:
                continue
            rank = int(item.get("rank") or fallback_rank)
            buckets.setdefault(item_id, dict(item))
            scores[item_id] = scores.get(item_id, 0.0) + weight * (1.0 / (float(rrf_k) + rank))
            source_key = "dense_rank" if list_idx == 0 else "bm25_rank" if list_idx == 1 else f"rank_{list_idx}"
            ranks_by_source.setdefault(item_id, {})[source_key] = rank

    fused: List[Dict[str, Any]] = []
    for item_id, item in buckets.items():
        out = dict(item)
        out["rrf_score"] = float(scores.get(item_id, 0.0))
        out.update(ranks_by_source.get(item_id, {}))
        fused.append(out)
    fused.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
    for rank, item in enumerate(fused[:top_k], start=1):
        item["rank"] = rank
    return fused[:top_k]


def _passage_from_node(node: Dict[str, Any], *, expansion_type: str, expanded_from: str, matched_reference: str = "") -> Dict[str, Any]:
    node_id = str(node.get("node_id") or "")
    md = _node_metadata(node)
    text = _node_text(node)
    out = {
        "id": node_id,
        "node_id": node_id,
        "chunk_id": str(md.get("chunk_id") or node_id),
        "doc_id": str(md.get("doc_id") or ""),
        "doc_key": str(md.get("doc_id") or ""),
        "text": text,
        "local_text": text,
        "shared_text": "",
        "snippet": text[:700],
        "retrieval_text": _norm_space(node.get("retrieval_text") or text),
        "rerank_text_short": _norm_space(node.get("rerank_text") or node.get("retrieval_text") or text)[:1500],
        "metadata": md,
        "expansion_type": expansion_type,
        "expanded_from": expanded_from,
    }
    if matched_reference:
        out["matched_reference"] = matched_reference
    return out


def expand_siblings(passages: List[Dict[str, Any]], graph: Dict[str, Any], window: int = 2, max_expanded: int = 20) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    out = [dict(p) for p in passages]
    seen = {_candidate_id(p) for p in out}
    expanded = 0

    for passage in passages:
        if expanded >= max_expanded:
            break
        node_id = str(passage.get("node_id") or "")
        node = node_idx.get(node_id)
        if not node:
            continue
        md = _node_metadata(node)
        parent_id = str(md.get("parent_id") or node.get("parent_id") or "").strip()
        siblings: List[Dict[str, Any]] = []
        if parent_id and parent_id in node_idx:
            parent_md = _node_metadata(node_idx[parent_id])
            child_ids = [str(x) for x in (parent_md.get("children_ids") or node_idx[parent_id].get("children_ids") or []) if str(x)]
            siblings = [node_idx[x] for x in child_ids if x in node_idx]
        if not siblings:
            article = _norm_key(md.get("article"))
            doc_id = _norm_key(md.get("doc_id"))
            siblings = [
                candidate
                for candidate in node_idx.values()
                if _norm_key(_node_metadata(candidate).get("doc_id")) == doc_id
                and _norm_key(_node_metadata(candidate).get("article")) == article
            ]
        if not siblings:
            continue
        siblings.sort(key=lambda n: int(_node_metadata(n).get("order_index") or 0))
        pos = next((idx for idx, candidate in enumerate(siblings) if str(candidate.get("node_id") or "") == node_id), -1)
        if pos < 0:
            continue
        for sibling in siblings[max(0, pos - window) : pos + window + 1]:
            sibling_id = str(sibling.get("node_id") or "")
            if not sibling_id or sibling_id in seen or sibling_id == node_id:
                continue
            text = _node_text(sibling)
            if not text:
                continue
            out.append(_passage_from_node(sibling, expansion_type="sibling", expanded_from=node_id))
            seen.add(sibling_id)
            expanded += 1
            if expanded >= max_expanded:
                break
    return out


def _extract_cross_refs(text: str) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    for match in _CROSS_REF_RE.finditer(_norm_space(text).lower()):
        point = match.group(2)
        clause = match.group(4)
        article = match.group(6)
        label_parts: List[str] = []
        if point:
            label_parts.append(f"Điểm {point}")
        if clause:
            label_parts.append(f"Khoản {clause}")
        label_parts.append(f"Điều {article}")
        refs.append(
            {
                "article": f"Điều {article}",
                "clause": f"Khoản {clause}" if clause else "",
                "point": f"Điểm {point}" if point else "",
                "label": " ".join(label_parts),
            }
        )
    return refs


def _resolve_reference(ref: Dict[str, str], graph: Dict[str, Any], source_md: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doc_number = _norm_key(source_md.get("doc_number"))
    doc_id = _norm_key(source_md.get("doc_id"))
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for node in _node_index(graph).values():
        md = _node_metadata(node)
        if doc_number and _norm_key(md.get("doc_number")) not in {"", doc_number}:
            continue
        if not doc_number and doc_id and _norm_key(md.get("doc_id")) != doc_id:
            continue
        if ref.get("article") and _norm_key(md.get("article")) != _norm_key(ref.get("article")):
            continue
        if ref.get("clause") and _norm_key(md.get("clause")) != _norm_key(ref.get("clause")):
            continue
        if ref.get("point") and _norm_key(md.get("point")) != _norm_key(ref.get("point")):
            continue
        specificity = int(bool(ref.get("article"))) + int(bool(ref.get("clause"))) + int(bool(ref.get("point")))
        candidates.append((specificity, node))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], int(_node_metadata(x[1]).get("order_index") or 0)), reverse=True)
    return candidates[0][1]


def expand_cross_refs(passages: List[Dict[str, Any]], graph: Dict[str, Any], max_refs: int = 10) -> List[Dict[str, Any]]:
    out = [dict(p) for p in passages]
    seen = {_candidate_id(p) for p in out}
    expanded = 0
    for passage in passages:
        if expanded >= max_refs:
            break
        source_id = str(passage.get("node_id") or "")
        source_md = dict(passage.get("metadata") or {})
        refs = _extract_cross_refs(str(passage.get("text") or passage.get("local_text") or ""))
        for ref in refs:
            node = _resolve_reference(ref, graph, source_md)
            if not node:
                continue
            node_id = str(node.get("node_id") or "")
            if not node_id or node_id in seen:
                continue
            out.append(
                _passage_from_node(
                    node,
                    expansion_type="cross_ref",
                    expanded_from=source_id,
                    matched_reference=str(ref.get("label") or ""),
                )
            )
            seen.add(node_id)
            expanded += 1
            if expanded >= max_refs:
                break
    return out


def graph_expansion_sibling_crossref(passages: List[Dict[str, Any]], graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    expanded = expand_siblings(passages, graph, window=2, max_expanded=20)
    return expand_cross_refs(expanded, graph, max_refs=10)


def optional_cpu_rerank(question: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bool(getattr(settings, "enable_reranker", False)):
        return passages
    input_top_k = max(1, int(getattr(settings, "rerank_input_top_k", 30) or 30))
    top_n = max(1, int(getattr(settings, "rerank_top_n", 5) or 5))
    reranked_head = cross_rerank(question, passages[:input_top_k], top_n=min(top_n, len(passages[:input_top_k])))
    remaining = [p for p in passages if _candidate_id(p) not in {_candidate_id(x) for x in reranked_head}]
    return reranked_head + remaining


def retrieve_with_graph_simple(
    question: str,
    graph: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    *,
    dense_top_k: int = 50,
    bm25_top_k: int = 50,
    fusion_top_k: int = 30,
    final_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    query_profile = analyze_query(question)
    if filters:
        merged = dict(query_profile.get("filters") or {})
        merged.update({key: value for key, value in filters.items() if value not in (None, "", [], {})})
        query_profile["filters"] = merged
        for key, value in merged.items():
            query_profile[key] = value

    candidates = _iter_graph_candidates(graph)
    filtered_candidates, filter_info = apply_metadata_filter(candidates, query_profile)
    dense_results = dense_search(
        question,
        top_k=dense_top_k,
        query_profile=query_profile,
        filter_info=filter_info,
    )
    bm25_results = bm25_search_vi(question, filtered_candidates, top_k=bm25_top_k)
    fused = rrf_fusion(
        [dense_results, bm25_results],
        rrf_k=int(getattr(settings, "rrf_k", 60) or 60),
        top_k=fusion_top_k,
    )
    expanded = graph_expansion_sibling_crossref(fused, graph)
    passages = optional_cpu_rerank(question, expanded)
    limit = int(final_top_k or getattr(settings, "final_top_k", 5) or 5)
    passages = passages[: max(1, limit)]

    return {
        "mode": "simple_rrf",
        "query_profile": query_profile,
        "filter_info": filter_info,
        "candidate_count": len(candidates),
        "filtered_candidate_count": len(filtered_candidates),
        "dense_results_count": len(dense_results),
        "bm25_results_count": len(bm25_results),
        "passages": passages,
        "reranker_enabled": bool(getattr(settings, "enable_reranker", False)),
        "insufficient_context": not bool(passages),
    }
