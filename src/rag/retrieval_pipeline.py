from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.auto_filter import infer_filters
from src.rag.bm25_corpus import load_bm25_stats
from src.rag.hybrid import hybrid_rank
from src.rag.legal_versioning import annotate_graph_with_versioning
from src.rag.openai_clients import get_embedings, get_llm
from src.rag.rerank_cross import cross_rerank
from src.storage.qdrant_store import get_qdrant_client, search_qdrant

logger = logging.getLogger(__name__)


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

QUERY_EXPANSION_SYSTEM_PROMPT = """
Bạn là bộ mở rộng truy vấn pháp luật cho GraphRAG Việt Nam.

Nhiệm vụ:
- Sinh các câu hỏi tương đương/biến thể để tăng recall truy xuất vector.
- Giữ nguyên ý nghĩa pháp lý cốt lõi của câu hỏi gốc.
- Ưu tiên biến thể theo các góc nhìn: điều khoản áp dụng, chủ thể, hành vi, chế tài, hiệu lực/sửa đổi (nếu có liên quan).
- Không tự thêm thông tin mới không có trong câu hỏi gốc.

Trả về JSON hợp lệ, không markdown, theo schema:
{
  "queries": ["...", "..."]
}
""".strip()


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

    variants: List[str] = []
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Sinh tối đa {n} câu hỏi biến thể cho câu hỏi sau.\n"
                        f"Câu hỏi gốc: {q}"
                    )
                ),
            ]
        )
        raw = str(getattr(response, "content", "") or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            if lines and lines[0].strip().lower() == "json":
                lines = lines[1:]
            raw = "\n".join(lines).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            variants = [str(x).strip() for x in parsed["queries"] if str(x).strip()]
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


def _normalize_node_id(value: Any) -> str:
    return str(value or "").strip().lower()


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


_INT_RE = re.compile(r"\d+")
_LETTER_RE = re.compile(r"([a-zA-Z])")


def _extract_number(text: Any) -> int:
    raw = str(text or "")
    m = _INT_RE.search(raw)
    if not m:
        return 10**9
    try:
        return int(m.group(0))
    except Exception:
        return 10**9


def _extract_letter_rank(text: Any) -> int:
    raw = str(text or "").strip().lower()
    m = _LETTER_RE.search(raw)
    if not m:
        return 10**9
    ch = m.group(1).lower()
    if "a" <= ch <= "z":
        return ord(ch) - ord("a")
    return 10**9


def _node_sort_key(node: Dict[str, Any]) -> Tuple[int, int, int, str]:
    md = node.get("metadata", {}) or {}
    node_type = str(node.get("node_type") or md.get("node_type") or "").strip().lower()
    type_rank = {
        "article": 0,
        "clause": 1,
        "point": 2,
        "bullet": 3,
        "text": 4,
    }.get(node_type, 9)

    clause_rank = _extract_number(md.get("clause"))
    point_rank = _extract_letter_rank(md.get("point"))
    node_id = str(node.get("node_id") or "")
    return (type_rank, clause_rank, point_rank, node_id)


def _find_parent_article_node_id(graph: Dict[str, Any], start_node_id: str) -> str:
    node_idx = _node_index(graph)
    reverse = _reverse_adjacency(graph)
    current = str(start_node_id)
    visited: Set[str] = set()

    while current and current not in visited:
        visited.add(current)
        node = node_idx.get(current, {})
        node_type = str(node.get("node_type") or "").strip().lower()
        if node_type == "article":
            return current

        parent_id: Optional[str] = None
        for e in reverse.get(current, []):
            if str(e.get("relation_type") or "").strip().upper() == "HAS_CHILD":
                src = e.get("source_id")
                if src:
                    parent_id = str(src)
                    break
        if not parent_id:
            break
        current = parent_id

    return str(start_node_id)


def _node_level(node: Dict[str, Any]) -> Optional[str]:
    md = node.get("metadata", {}) or {}
    node_type = str(node.get("node_type") or md.get("node_type") or "").strip().lower()
    if node_type in {"article", "clause", "point"}:
        return node_type
    return None


def _parse_asked_level(question: str) -> Optional[str]:
    q = (question or "").lower()
    if "điểm" in q:
        return "point"
    if "khoản" in q:
        return "clause"
    if "điều" in q:
        return "article"
    return None


def _classify_relative_level(asked_level: Optional[str], hit_level: Optional[str]) -> Optional[str]:
    if not asked_level or not hit_level:
        return None
    rank = {"article": 1, "clause": 2, "point": 3}
    a = rank.get(asked_level)
    h = rank.get(hit_level)
    if a is None or h is None:
        return None
    if a == h:
        return "same_level"
    if a < h:
        return "upper_level"
    return "lower_level"


def analyze_hierarchy_scope(
    question: str,
    graph: Dict[str, Any],
    seed_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    asked_level = _parse_asked_level(question)
    if not asked_level:
        return {
            "is_hierarchy_query": False,
            "asked_level": None,
            "hit_level": None,
            "relative_level": None,
        }

    node_idx = _node_index(graph)
    hit_node: Optional[Dict[str, Any]] = None
    for s in seed_items:
        node_id = str(s.get("node_id") or "").strip()
        if node_id and node_id in node_idx:
            hit_node = node_idx[node_id]
            break

    hit_level = _node_level(hit_node or {})
    relative_level = _classify_relative_level(asked_level, hit_level)
    return {
        "is_hierarchy_query": True,
        "asked_level": asked_level,
        "hit_level": hit_level,
        "relative_level": relative_level,
    }


def _build_passage_from_node(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    node_id = str(node.get("node_id") or "").strip()
    text = str(node.get("text") or "").strip()
    if not node_id or not text:
        return None
    md = node.get("metadata", {}) or {}
    return {
        "node_id": node_id,
        "text": text,
        "snippet": text[:700],
        "metadata": md,
        "dense_score": 0.0,
        "hybrid_score": 0.0,
        "graph_score": 0.0,
        "version_bonus": 0.0,
        "version_status": md.get("version_status"),
        "version_event_count": int(md.get("version_event_count") or 0),
        "final_score": 0.0,
    }


def _inject_summary_passages(graph: Dict[str, Any], passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not passages:
        return passages

    node_idx = _node_index(graph)
    reverse = _reverse_adjacency(graph)
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for p in passages:
        node_id = str(p.get("node_id") or "")
        if node_id and node_id not in seen:
            out.append(p)
            seen.add(node_id)

        for e in reverse.get(node_id, []):
            rel = str(e.get("relation_type") or "").strip().upper()
            if rel != "SUMMARIZES":
                continue
            src = str(e.get("source_id") or "").strip()
            s_node = node_idx.get(src)
            if not s_node:
                continue
            md = s_node.get("metadata", {}) or {}
            if str(md.get("artifact_type") or "").strip().lower() != "summary":
                continue
            sp = _build_passage_from_node(s_node)
            if not sp:
                continue
            sid = str(sp.get("node_id") or "")
            if sid and sid not in seen:
                out.append(sp)
                seen.add(sid)
    return out


def _prune_passages_for_hierarchy_scope(
    graph: Dict[str, Any],
    passages: List[Dict[str, Any]],
    hierarchy_scope: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not passages or not hierarchy_scope or not hierarchy_scope.get("is_hierarchy_query"):
        return passages

    relative = str(hierarchy_scope.get("relative_level") or "")
    node_idx = _node_index(graph)

    if relative in {"same_level", "upper_level"}:
        # Trả lời ngắn: ưu tiên node cùng cây điều + summary của chúng
        compact: List[Dict[str, Any]] = []
        for p in passages:
            node = node_idx.get(str(p.get("node_id") or ""), {})
            level = _node_level(node)
            if level in {"article", "clause", "point"}:
                compact.append(p)
        compact = _inject_summary_passages(graph, compact)
        return compact if compact else passages

    if relative == "lower_level":
        # Trả lời chi tiết: giữ passage hiện có + bổ sung summary hỗ trợ
        detailed = _inject_summary_passages(graph, passages)
        return detailed if detailed else passages

    return passages


def reorder_passages_by_article_tree(
    graph: Dict[str, Any],
    passages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Nếu top hit rơi vào clause/point thì kéo parent article lên trước,
    rồi ưu tiên các node cùng cây article theo thứ tự cấu trúc.
    """
    if not passages:
        return passages

    node_idx = _node_index(graph)
    adjacency = _adjacency(graph)
    first_id = str(passages[0].get("node_id") or "")
    if not first_id or first_id not in node_idx:
        return passages

    article_root_id = _find_parent_article_node_id(graph, first_id)
    if not article_root_id or article_root_id not in node_idx:
        return passages

    # Duyệt cây article bằng HAS_CHILD để biết cụm node liên quan
    stack = [article_root_id]
    subtree_ids: Set[str] = set()
    while stack:
        nid = stack.pop()
        if nid in subtree_ids:
            continue
        subtree_ids.add(nid)
        for e in adjacency.get(nid, []):
            if str(e.get("relation_type") or "").strip().upper() != "HAS_CHILD":
                continue
            child = e.get("target_id")
            if child:
                stack.append(str(child))

    original_pos = {str(p.get("node_id") or ""): i for i, p in enumerate(passages)}
    in_subtree: List[Dict[str, Any]] = []
    out_subtree: List[Dict[str, Any]] = []

    # Luôn cố gắng đưa article root lên đầu nếu có text
    article_node = node_idx.get(article_root_id, {})
    article_text = str(article_node.get("text") or "").strip()
    if article_text and article_root_id not in original_pos:
        in_subtree.append(
            {
                "node_id": article_root_id,
                "text": article_text,
                "snippet": article_text[:700],
                "metadata": article_node.get("metadata", {}) or {},
                "dense_score": 0.0,
                "hybrid_score": 0.0,
                "graph_score": 0.0,
                "version_bonus": 0.0,
                "version_status": (article_node.get("metadata", {}) or {}).get("version_status"),
                "version_event_count": int((article_node.get("metadata", {}) or {}).get("version_event_count") or 0),
                "final_score": 0.0,
            }
        )

    for p in passages:
        node_id = str(p.get("node_id") or "")
        if node_id in subtree_ids:
            in_subtree.append(p)
        else:
            out_subtree.append(p)

    in_subtree.sort(
        key=lambda p: (
            _node_sort_key(node_idx.get(str(p.get("node_id") or ""), p)),
            original_pos.get(str(p.get("node_id") or ""), 10**9),
        )
    )
    out_subtree.sort(key=lambda p: original_pos.get(str(p.get("node_id") or ""), 10**9))
    merged = in_subtree + out_subtree

    # Gắn lại rank sau reorder
    for i, p in enumerate(merged):
        p["rank"] = i + 1
    return merged


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
    node_idx_by_norm: Dict[str, str] = {
        _normalize_node_id(node_id): str(node_id)
        for node_id in node_idx.keys()
    }
    chunk_id_index: Dict[str, str] = {}
    for real_node_id, node in node_idx.items():
        md = node.get("metadata", {}) or {}
        chunk_id = md.get("chunk_id")
        norm_chunk_id = _normalize_node_id(chunk_id)
        if norm_chunk_id and norm_chunk_id not in chunk_id_index:
            chunk_id_index[norm_chunk_id] = real_node_id

    queue = deque()
    best_score: Dict[str, float] = {}
    unmatched_seed_ids: List[str] = []
    matched_seed_count = 0

    for item in seed_items:
        raw_node_id = item.get("node_id")
        md = item.get("metadata", {}) or {}
        raw_chunk_id = md.get("chunk_id") if isinstance(md, dict) else None

        resolved_node_id: Optional[str] = None
        norm_node_id = _normalize_node_id(raw_node_id)
        if norm_node_id:
            resolved_node_id = node_idx_by_norm.get(norm_node_id)

        if not resolved_node_id:
            norm_chunk_id = _normalize_node_id(raw_chunk_id)
            if norm_chunk_id:
                resolved_node_id = chunk_id_index.get(norm_chunk_id)

        if not resolved_node_id or resolved_node_id not in node_idx:
            sample_id = str(raw_node_id or raw_chunk_id or "").strip()
            if sample_id:
                unmatched_seed_ids.append(sample_id)
            continue

        seed_score = float(item.get("hybrid_score", item.get("dense_score", 0.0)) or 0.0)
        best_score[resolved_node_id] = max(best_score.get(resolved_node_id, 0.0), seed_score)
        queue.append((resolved_node_id, 0, seed_score))
        matched_seed_count += 1

    adjacency_size = sum(len(v) for v in _adjacency(graph).values())
    reverse_adjacency_size = sum(len(v) for v in _reverse_adjacency(graph).values())
    logger.info(
        "Graph expansion seeds: total=%s matched=%s unmatched=%s mode=%s allowed_rel=%s adjacency_edges=%s reverse_edges=%s",
        len(seed_items),
        matched_seed_count,
        len(seed_items) - matched_seed_count,
        mode,
        sorted(list(allowed_relations)),
        adjacency_size,
        reverse_adjacency_size,
    )
    if unmatched_seed_ids:
        logger.debug(
            "Graph expansion unmatched seed IDs (sample): %s",
            unmatched_seed_ids[:10],
        )
    if adjacency_size == 0 and reverse_adjacency_size == 0:
        logger.warning("Graph adjacency is empty; expansion cannot traverse relations.")

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
    hierarchy_scope: Optional[Dict[str, Any]] = None,
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
    use_rerank = not bool((hierarchy_scope or {}).get("is_hierarchy_query"))

    if use_rerank:
        cross_k = cross_top_k or settings.cross_top_k
        cross_k = min(cross_k, len(pre_cross_passages))

        if cross_k > 0:
            reranked_top = cross_rerank(question, pre_cross_passages[:cross_k], top_n=cross_k)
            remaining = pre_cross_passages[cross_k:]
            final_passages = reranked_top + remaining
        else:
            final_passages = pre_cross_passages
    else:
        final_passages = pre_cross_passages

    final_passages = reorder_passages_by_article_tree(graph, final_passages)
    final_passages = _prune_passages_for_hierarchy_scope(graph, final_passages, hierarchy_scope)

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
    hierarchy_scope = analyze_hierarchy_scope(question, graph, seeds)
    logger.info("Hierarchy scope: %s", hierarchy_scope)

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
        hierarchy_scope=hierarchy_scope,
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
        "hierarchy_scope": hierarchy_scope,
    }
