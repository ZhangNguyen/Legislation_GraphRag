from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.app.settings import settings
from src.rag.auto_filter import infer_filters
from src.rag.bm25_corpus import load_bm25_stats
from src.rag.hybrid import bm25_score, hybrid_rank, tokenize
from src.rag.openai_clients import get_embedings
from src.rag.rerank_cross import cross_rerank
from src.storage.qdrant_store import get_qdrant_client, search_qdrant

logger = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
POINT_RE = re.compile(r"điểm\s+([a-zđ])", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

_BM25_CACHE: Dict[str, Any] = {}


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def get_bm25_stats_cached(path: str) -> Dict[str, Any]:
    cached = _BM25_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        stats = load_bm25_stats(path)
    except Exception:
        stats = {"idf": {}, "avgdl": 0.0}
    _BM25_CACHE[path] = stats
    return stats


def _extract_node_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not payload:
        return None
    metadata = payload.get("metadata") or {}
    return metadata.get("node_id") or metadata.get("chunk_id") or payload.get("node_id") or payload.get("chunk_id")


def _point_to_candidate(point: Any) -> Dict[str, Any]:
    payload = getattr(point, "payload", {}) or {}
    metadata = dict(payload.get("metadata") or {})
    node_id = _extract_node_id_from_payload(payload)
    text = str(payload.get("text") or payload.get("snippet") or "").strip()
    retrieval_text = str(payload.get("retrieval_text") or text).strip()
    return {
        "node_id": str(node_id or ""),
        "text": text,
        "snippet": text[:600],
        "retrieval_text": retrieval_text,
        "rerank_text": str(payload.get("rerank_text") or retrieval_text),
        "metadata": metadata,
        "dense_score": float(getattr(point, "score", 0.0) or 0.0),
    }


def _fallback_query_profile(question: str) -> Dict[str, Any]:
    q = _norm_space(question)
    article_match = ARTICLE_RE.search(q)
    clause_match = CLAUSE_RE.search(q)
    point_match = POINT_RE.search(q)
    year_match = YEAR_RE.search(q)
    return {
        "mode": "general",
        "target_level": "point" if point_match else "clause" if clause_match else "article" if article_match else "document",
        "explicit_refs": {
            "article": f"Điều {article_match.group(1)}" if article_match else None,
            "clause": f"Khoản {clause_match.group(1)}" if clause_match else None,
            "point": f"Điểm {point_match.group(1).lower()}" if point_match else None,
            "year": int(year_match.group(1)) if year_match else None,
        },
    }


def retrieve_seed_candidates(
    question: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    merged_filters = dict(infer_filters(question))
    if filters:
        merged_filters.update({k: v for k, v in filters.items() if v not in (None, "")})
    profile = _fallback_query_profile(question)
    if profile["explicit_refs"].get("year") and "year" not in merged_filters:
        merged_filters["year"] = profile["explicit_refs"]["year"]

    client = get_qdrant_client()
    embeddings = get_embedings()
    vector = embeddings.embed_query(question)
    raw_points = search_qdrant(
        client=client,
        query_vector=vector,
        top_k=top_k or settings.qdrant_top_k,
        filters=merged_filters,
    )
    dense_candidates = [_point_to_candidate(p) for p in raw_points]
    bm25_stats = get_bm25_stats_cached(settings.bm25_stats_path)
    ranked = hybrid_rank(
        question,
        dense_candidates,
        dense_key="dense_score",
        text_key="retrieval_text",
        alpha=settings.hybrid_alpha,
        idf_map=bm25_stats.get("idf", {}),
        avgdl=bm25_stats.get("avgdl", 0.0),
    )
    out: List[Dict[str, Any]] = []
    for score, cand in ranked[: settings.bm25_top_k]:
        item = dict(cand)
        item["hybrid_score"] = float(score)
        item["query_profile"] = profile
        out.append(item)
    return out


def _query_overlap_score(question: str, text: str) -> float:
    q_tokens = tokenize(question)
    d_tokens = tokenize(text)
    if not q_tokens or not d_tokens:
        return 0.0
    overlap = len(set(q_tokens) & set(d_tokens)) / max(len(set(q_tokens)), 1)
    bm25 = bm25_score(q_tokens, d_tokens)
    return float(0.7 * overlap + 0.3 * min(bm25 / 10.0, 1.0))


def _scope_bonus(question: str, metadata: Dict[str, Any]) -> float:
    profile = _fallback_query_profile(question)
    bonus = 0.0
    refs = profile.get("explicit_refs") or {}
    if refs.get("article") and str(metadata.get("article") or "").lower() == str(refs["article"]).lower():
        bonus += 0.18
    if refs.get("clause") and str(metadata.get("clause") or "").lower() == str(refs["clause"]).lower():
        bonus += 0.16
    if refs.get("point") and str(metadata.get("point") or "").lower() == str(refs["point"]).lower():
        bonus += 0.14
    return bonus


def _select_related(node_ids: List[str], graph: Dict[str, Any], question: str, limit: int) -> List[str]:
    node_index = graph.get("node_index", {}) or {}
    scored: List[Tuple[float, str]] = []
    for node_id in node_ids:
        node = node_index.get(str(node_id))
        if not node:
            continue
        score = _query_overlap_score(question, node.get("text") or node.get("retrieval_text") or "")
        scored.append((score, str(node_id)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [node_id for _, node_id in scored[: max(limit, 0)] if node_id]


def _build_bundle_from_seed(seed: Dict[str, Any], graph: Dict[str, Any], question: str) -> Optional[Dict[str, Any]]:
    node_index = graph.get("node_index", {}) or {}
    seed_node_id = str(seed.get("node_id") or "")
    node = node_index.get(seed_node_id)
    if not node:
        return None

    md = dict(node.get("metadata") or seed.get("metadata") or {})
    doc_id = str(md.get("doc_id") or "")
    doc_meta = dict((graph.get("doc_index", {}) or {}).get(doc_id, {}))

    parent_id = md.get("parent_id")
    parent_node = node_index.get(str(parent_id)) if parent_id else None
    parent_ids: List[str] = []
    cur = parent_node
    hop = 0
    while cur is not None and hop < 2:
        parent_ids.append(str(cur["node_id"]))
        next_parent = cur.get("metadata", {}).get("parent_id")
        cur = node_index.get(str(next_parent)) if next_parent else None
        hop += 1

    child_ids = _select_related(list(md.get("children_ids") or []), graph, question, settings.bundle_max_children)
    sibling_ids = _select_related(list(md.get("sibling_ids") or []), graph, question, settings.bundle_max_siblings)

    sections: List[str] = []
    title = doc_meta.get("official_title") or md.get("official_title") or md.get("law_name") or ""
    doc_type = doc_meta.get("doc_type") or md.get("doc_type") or md.get("law_type") or "Unknown"
    lead_block = doc_meta.get("lead_block") or md.get("lead_block") or ""
    path_title = md.get("path_title") or ""

    sections.append(f"[DOC_TYPE] {doc_type}")
    sections.append(f"[DOC_TITLE] {title}")
    if lead_block:
        sections.append(f"[DOC_LEAD] {lead_block}")
    if path_title:
        sections.append(f"[PATH] {path_title}")

    for pid in reversed(parent_ids):
        parent = node_index.get(pid)
        if parent:
            sections.append(f"[PARENT] {parent.get('text')}")

    sections.append(f"[CENTER] {node.get('text')}")

    for cid in child_ids:
        child = node_index.get(cid)
        if child:
            sections.append(f"[CHILD] {child.get('text')}")

    for sid in sibling_ids:
        sibling = node_index.get(sid)
        if sibling:
            sections.append(f"[SIBLING] {sibling.get('text')}")

    bundle_text = "\n".join(_norm_space(x) for x in sections if _norm_space(x)).strip()
    bundle_overlap = _query_overlap_score(question, bundle_text)

    out = {
        "node_id": seed_node_id,
        "center_node_id": seed_node_id,
        "text": str(node.get("text") or "").strip(),
        "snippet": str(node.get("text") or "")[:600],
        "retrieval_text": bundle_text,
        "rerank_text": bundle_text,
        "bundle_text": bundle_text,
        "metadata": {**doc_meta, **md},
        "dense_score": _safe_float(seed.get("dense_score")),
        "hybrid_score": _safe_float(seed.get("hybrid_score")),
        "bundle_overlap": bundle_overlap,
        "scope_bonus": _scope_bonus(question, md),
    }
    return out


def retrieve_with_graph(
    question: str,
    *,
    graph: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    qdrant_top_k: Optional[int] = None,
    graph_hops: int = 2,
    max_graph_nodes: int = 80,
    final_top_k: Optional[int] = None,
    cross_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    _ = (graph_hops, max_graph_nodes)
    seed_candidates = retrieve_seed_candidates(question, filters=filters, top_k=qdrant_top_k)
    bundles: List[Dict[str, Any]] = []
    seen = set()
    for seed in seed_candidates:
        bundle = _build_bundle_from_seed(seed, graph, question)
        if not bundle:
            continue
        key = str(bundle.get("center_node_id") or "")
        if key in seen:
            continue
        seen.add(key)
        bundles.append(bundle)

    if not bundles:
        return {
            "mode": "general",
            "filters": dict(filters or {}),
            "qdrant_top_k": qdrant_top_k or settings.qdrant_top_k,
            "final_top_k": final_top_k or settings.final_top_k,
            "cross_top_k": cross_top_k or settings.cross_top_k,
            "passages": [],
        }

    cross_limit = min(cross_top_k or settings.cross_top_k, len(bundles))
    reranked = cross_rerank(question, bundles, top_n=cross_limit)
    reranked_map = {str(x.get("center_node_id") or x.get("node_id") or ""): x for x in reranked}

    scored: List[Dict[str, Any]] = []
    for item in bundles:
        key = str(item.get("center_node_id") or item.get("node_id") or "")
        cur = dict(item)
        cur["cross_score"] = _safe_float(reranked_map.get(key, {}).get("cross_score"))
        cur["final_score"] = (
            0.62 * cur["cross_score"]
            + 0.23 * _safe_float(cur.get("hybrid_score"))
            + 0.10 * _safe_float(cur.get("bundle_overlap"))
            + 0.05 * _safe_float(cur.get("scope_bonus"))
        )
        scored.append(cur)

    scored.sort(key=lambda x: _safe_float(x.get("final_score")), reverse=True)
    top_final = scored[: (final_top_k or settings.final_top_k)]

    return {
        "mode": "general",
        "filters": dict(filters or {}),
        "qdrant_top_k": qdrant_top_k or settings.qdrant_top_k,
        "final_top_k": final_top_k or settings.final_top_k,
        "cross_top_k": cross_top_k or settings.cross_top_k,
        "seed_candidates": seed_candidates,
        "passages": top_final,
    }
