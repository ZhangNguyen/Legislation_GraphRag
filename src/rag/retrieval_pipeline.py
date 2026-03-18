from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from src.app.settings import settings
from src.rag.auto_filter import infer_filters
from src.rag.bm25_corpus import load_bm25_stats
from src.rag.hybrid import hybrid_rank
from src.rag.legal_versioning import annotate_graph_with_versioning
from src.rag.openai_clients import get_embedings, get_llm
from src.rag.rerank_cross import cross_rerank
from src.storage.qdrant_store import get_qdrant_client, search_qdrant


STRUCTURAL_RELATIONS = {
    "HAS_CHILD",
    "SUMMARIZES",
    "HAS_CHILD_SUMMARY",
}

GENERAL_RELATIONS = {
    "REGULATES",
    "REFERS_TO",
    "APPLIES_TO",
    "REQUIRES",
    "PROHIBITS",
    "RESPONSIBLE_FOR",
    "PENALIZED_BY",
    "RELATED_IN_CONTEXT",
}

CHANGE_RELATIONS = {
    "AMENDS",
    "REPEALS",
    "REPLACES",
    "PARTIALLY_AMENDS",
    "PARTIALLY_REPEALS",
    "EFFECTIVE_FROM",
    "REFERS_TO",
}

SANCTION_KEYWORDS = [
    "bị gì",
    "xử phạt",
    "chế tài",
    "phạt",
    "truy cứu",
    "hình sự",
    "hành chính",
]

CHANGE_KEYWORDS = [
    "sửa đổi",
    "bổ sung",
    "bãi bỏ",
    "thay thế",
    "được sửa",
    "bị sửa",
    "bản nào",
]

EFFECTIVE_KEYWORDS = [
    "hiệu lực",
    "còn hiệu lực",
    "hết hiệu lực",
    "áp dụng",
    "hiện hành",
]

_BM25_CACHE: Dict[str, Any] = {}


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


def ensure_versioning_ready(graph: Dict[str, Any]) -> Dict[str, Any]:
    if graph.get("_versioning_ready"):
        return graph

    graph = annotate_graph_with_versioning(graph)
    graph["_versioning_ready"] = True
    return graph


def detect_query_mode(question: str) -> str:
    q = (question or "").lower()

    if any(k in q for k in CHANGE_KEYWORDS):
        return "change"
    if any(k in q for k in EFFECTIVE_KEYWORDS):
        return "effective"
    if any(k in q for k in SANCTION_KEYWORDS):
        return "sanction"
    return "general"


def allowed_relations_for_mode(mode: str) -> Set[str]:
    if mode == "change":
        return STRUCTURAL_RELATIONS | CHANGE_RELATIONS
    if mode == "effective":
        return STRUCTURAL_RELATIONS | CHANGE_RELATIONS | {"RESPONSIBLE_FOR", "APPLIES_TO"}
    if mode == "sanction":
        return STRUCTURAL_RELATIONS | GENERAL_RELATIONS | {"REGULATES"}
    return STRUCTURAL_RELATIONS | GENERAL_RELATIONS | CHANGE_RELATIONS


def compute_version_bonus(
    node: Dict[str, Any],
    *,
    question: str,
    mode: str,
) -> float:
    md = node.get("metadata", {}) or {}
    status = str(md.get("version_status") or "").strip().lower()
    event_count = int(md.get("version_event_count") or 0)

    legal_role = str(md.get("legal_role") or "").strip().lower()
    node_type = str(node.get("node_type") or "").strip().lower()

    bonus = 0.0

    if mode == "effective":
        if status == "repealed":
            bonus -= 0.35
        elif status == "replaced":
            bonus += 0.10
        elif status == "amended":
            bonus += 0.16
        elif status == "effective_info_found":
            bonus += 0.22

        if legal_role in {"effective", "responsibility", "transition"}:
            bonus += 0.10

    elif mode == "change":
        if status == "repealed":
            bonus += 0.20
        elif status == "replaced":
            bonus += 0.22
        elif status == "amended":
            bonus += 0.25
        elif status == "effective_info_found":
            bonus += 0.05

        if legal_role in {"amendment", "effective", "transition", "correction", "replacement"}:
            bonus += 0.10
        if node_type in {"amendment", "effective"}:
            bonus += 0.08

    else:
        if status == "repealed":
            bonus -= 0.12
        elif status in {"amended", "replaced", "effective_info_found"}:
            bonus += 0.04

    if event_count > 0:
        bonus += min(event_count * 0.02, 0.08)

    return float(bonus)


def _embed_query(question: str) -> List[float]:
    embeddings = get_embedings()
    return embeddings.embed_query(question)


def generate_query_variants(question: str, *, n: int = 3) -> List[str]:
    """
    Sinh thêm các biến thể của câu hỏi người dùng trước khi tính similarity.
    Fallback về câu hỏi gốc nếu LLM fail.
    """
    q = (question or "").strip()
    if not q:
        return []

    try:
        llm = get_llm()
        prompt = (
            f"Sinh {n} câu hỏi biến thể tiếng Việt có cùng ý nghĩa pháp lý với câu hỏi sau. "
            "Mỗi câu 1 dòng, không đánh số, không giải thích.\n\n"
            f"QUESTION: {q}"
        )
        response = llm.invoke(prompt)
        lines = [str(x).strip(" -\t") for x in str(getattr(response, "content", "") or "").splitlines()]
        variants = [x for x in lines if x]
    except Exception:
        variants = []

    merged = [q]
    for v in variants:
        if v.lower() not in {x.lower() for x in merged}:
            merged.append(v)
        if len(merged) >= n + 1:
            break

    return merged


def _extract_node_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not payload:
        return None

    md = payload.get("metadata")
    if isinstance(md, dict):
        node_id = md.get("node_id")
        if node_id:
            return str(node_id)

    for key in ["node_id", "chunk_id", "id"]:
        if payload.get(key):
            return str(payload[key])

    return None


def _point_to_candidate(point: Any) -> Dict[str, Any]:
    payload = getattr(point, "payload", {}) or {}
    text = payload.get("text") or payload.get("snippet") or ""
    metadata = payload.get("metadata") or {}

    node_id = _extract_node_id_from_payload(payload)
    if not node_id and isinstance(metadata, dict):
        node_id = metadata.get("chunk_id") or metadata.get("id")

    if isinstance(metadata, dict):
        for k in ["law_name", "law_type", "year", "source", "article", "clause", "point", "chunk_id", "node_id"]:
            if k not in metadata and payload.get(k) is not None:
                metadata[k] = payload.get(k)

    return {
        "node_id": node_id,
        "text": text,
        "snippet": text[:500],
        "metadata": metadata,
        "dense_score": float(getattr(point, "score", 0.0) or 0.0),
        "payload": payload,
    }


def retrieve_seed_candidates(
    question: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    merged_filters = dict(infer_filters(question))
    if filters:
        merged_filters.update({k: v for k, v in filters.items() if v is not None and v != ""})

    expanded_questions = generate_query_variants(question, n=3)
    embeddings = get_embedings()
    vectors = [embeddings.embed_query(q) for q in expanded_questions] or [embeddings.embed_query(question)]
    dims = len(vectors[0]) if vectors and vectors[0] else 0
    query_vector = [0.0] * dims
    for vec in vectors:
        for i, v in enumerate(vec):
            query_vector[i] += float(v)
    query_vector = [v / float(len(vectors)) for v in query_vector]

    client = get_qdrant_client()

    raw_points = search_qdrant(
        client=client,
        query_vector=query_vector,
        top_k=top_k or settings.qdrant_top_k,
        filters=merged_filters,
    )

    candidates = [_point_to_candidate(p) for p in raw_points]
    bm25_stats = get_bm25_stats_cached(settings.bm25_stats_path)

    ranked = hybrid_rank(
        question,
        candidates,
        dense_key="dense_score",
        text_key="text",
        idf_map=bm25_stats.get("idf", {}),
        avgdl=bm25_stats.get("avgdl", 0.0),
    )

    out: List[Dict[str, Any]] = []
    for rank_score, cand in ranked:
        item = dict(cand)
        item["hybrid_score"] = float(rank_score)
        item["query_variants"] = expanded_questions
        out.append(item)

    return out


def _node_index(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return graph.get("node_index", {}) or {}


def _adjacency(graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return graph.get("adjacency", {}) or {}


def _reverse_adjacency(graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return graph.get("reverse_adjacency", {}) or {}


def _is_evidence(node: Dict[str, Any]) -> bool:
    md = node.get("metadata", {}) or {}
    return md.get("artifact_type") == "evidence"


def _is_summary(node: Dict[str, Any]) -> bool:
    md = node.get("metadata", {}) or {}
    return md.get("artifact_type") == "summary"


def _graph_neighbors(
    graph: Dict[str, Any],
    node_id: str,
    allowed_relations: Set[str],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    for e in _adjacency(graph).get(node_id, []):
        rel = str(e.get("relation_type") or "").strip().upper()
        tgt = e.get("target_id")
        if tgt and rel in allowed_relations:
            out.append((str(tgt), "out"))

    for e in _reverse_adjacency(graph).get(node_id, []):
        rel = str(e.get("relation_type") or "").strip().upper()
        src = e.get("source_id")
        if src and rel in allowed_relations:
            out.append((str(src), "in"))

    return out


def expand_graph_from_seeds(
    graph: Dict[str, Any],
    seed_items: List[Dict[str, Any]],
    *,
    question: str,
    max_hops: int = 2,
    max_nodes: int = 80,
) -> Dict[str, float]:
    mode = detect_query_mode(question)
    allowed_relations = allowed_relations_for_mode(mode)
    node_idx = _node_index(graph)

    queue = deque()
    best_score: Dict[str, float] = {}

    for item in seed_items:
        node_id = item.get("node_id")
        if not node_id or node_id not in node_idx:
            continue

        seed_score = float(item.get("hybrid_score", item.get("dense_score", 0.0)) or 0.0)
        best_score[node_id] = max(best_score.get(node_id, 0.0), seed_score)
        queue.append((node_id, 0, seed_score))

    visited: Set[Tuple[str, int]] = set()

    while queue and len(best_score) < max_nodes:
        current_id, hop, current_score = queue.popleft()

        if (current_id, hop) in visited:
            continue
        visited.add((current_id, hop))

        if hop >= max_hops:
            continue

        neighbors = _graph_neighbors(
            graph=graph,
            node_id=current_id,
            allowed_relations=allowed_relations,
        )

        for neighbor_id, _direction in neighbors:
            if neighbor_id not in node_idx:
                continue

            next_score = current_score * (0.82 if hop == 0 else 0.68)

            node = node_idx[neighbor_id]
            if _is_evidence(node):
                next_score *= 1.05
            elif _is_summary(node):
                next_score *= 0.95

            if next_score > best_score.get(neighbor_id, -1.0):
                best_score[neighbor_id] = next_score
                queue.append((neighbor_id, hop + 1, next_score))

    return best_score


def collect_final_passages(
    graph: Dict[str, Any],
    seed_items: List[Dict[str, Any]],
    graph_scores: Dict[str, float],
    *,
    question: str,
    final_top_k: Optional[int] = None,
    cross_top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)

    seed_by_id = {
        item.get("node_id"): item
        for item in seed_items
        if item.get("node_id")
    }

    passages: List[Dict[str, Any]] = []
    mode = detect_query_mode(question)

    for node_id, g_score in graph_scores.items():
        node = node_idx.get(node_id)
        if not node:
            continue

        md = node.get("metadata", {}) or {}
        text = str(node.get("text") or "").strip()
        if not text:
            continue

        seed_item = seed_by_id.get(node_id, {})
        dense_score = float(seed_item.get("dense_score", 0.0) or 0.0)
        hybrid_score = float(seed_item.get("hybrid_score", 0.0) or 0.0)

        evidence_bonus = 0.12 if md.get("artifact_type") == "evidence" else 0.0
        version_bonus = compute_version_bonus(
            node,
            question=question,
            mode=mode,
        )

        final_score = (
            0.55 * hybrid_score
            + 0.45 * g_score
            + evidence_bonus
            + version_bonus
        )

        passages.append(
            {
                "node_id": node_id,
                "text": text,
                "snippet": text[:700],
                "metadata": md,
                "dense_score": dense_score,
                "hybrid_score": hybrid_score,
                "graph_score": float(g_score),
                "version_bonus": float(version_bonus),
                "version_status": md.get("version_status"),
                "version_event_count": int(md.get("version_event_count") or 0),
                "final_score": float(final_score),
            }
        )

    if not passages:
        for item in seed_items:
            text = str(item.get("text") or "").strip()
            if not text:
                continue

            md = item.get("metadata", {}) or {}
            pseudo_node = {
                "node_type": md.get("node_type"),
                "metadata": md,
            }
            version_bonus = compute_version_bonus(
                pseudo_node,
                question=question,
                mode=mode,
            )

            passages.append(
                {
                    "node_id": item.get("node_id"),
                    "text": text,
                    "snippet": item.get("snippet") or text[:700],
                    "metadata": md,
                    "dense_score": float(item.get("dense_score", 0.0) or 0.0),
                    "hybrid_score": float(item.get("hybrid_score", 0.0) or 0.0),
                    "graph_score": 0.0,
                    "version_bonus": float(version_bonus),
                    "version_status": md.get("version_status"),
                    "version_event_count": int(md.get("version_event_count") or 0),
                    "final_score": float(item.get("hybrid_score", 0.0) or 0.0) + float(version_bonus),
                }
            )

    passages.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)

    final_k = final_top_k or settings.final_top_k
    pre_cross_passages = passages[:final_k]

    cross_k = cross_top_k or settings.cross_top_k
    cross_k = min(cross_k, len(pre_cross_passages))

    if cross_k > 0:
        reranked_top = cross_rerank(question, pre_cross_passages[:cross_k], top_n=cross_k)
        remaining = pre_cross_passages[cross_k:]
        final_passages = reranked_top + remaining
    else:
        final_passages = pre_cross_passages

    for i, p in enumerate(final_passages):
        p["rank"] = i + 1

    return final_passages


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
    merged_filters = dict(infer_filters(question))
    if filters:
        merged_filters.update({k: v for k, v in filters.items() if v is not None and v != ""})

    graph = ensure_versioning_ready(graph)

    seeds = retrieve_seed_candidates(
        question,
        filters=merged_filters,
        top_k=qdrant_top_k or settings.qdrant_top_k,
    )

    graph_scores = expand_graph_from_seeds(
        graph=graph,
        seed_items=seeds,
        question=question,
        max_hops=graph_hops,
        max_nodes=max_graph_nodes,
    )

    passages = collect_final_passages(
        graph=graph,
        seed_items=seeds,
        graph_scores=graph_scores,
        question=question,
        final_top_k=final_top_k or settings.final_top_k,
        cross_top_k=cross_top_k or settings.cross_top_k,
    )

    return {
        "question": question,
        "filters": merged_filters,
        "mode": detect_query_mode(question),
        "qdrant_top_k": qdrant_top_k or settings.qdrant_top_k,
        "final_top_k": final_top_k or settings.final_top_k,
        "cross_top_k": cross_top_k or settings.cross_top_k,
        "seed_candidates": seeds,
        "graph_scores": graph_scores,
        "passages": passages,
    }
