
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
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

STRUCTURAL_RELATIONS = {"HAS_CHILD", "SUMMARIZES", "HAS_CHILD_SUMMARY"}
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

SANCTION_KEYWORDS = ["bị gì", "xử phạt", "chế tài", "phạt", "truy cứu", "hình sự", "hành chính"]
CHANGE_KEYWORDS = ["sửa đổi", "bổ sung", "bãi bỏ", "thay thế", "được sửa", "bị sửa", "bản nào"]
EFFECTIVE_KEYWORDS = ["hiệu lực", "còn hiệu lực", "hết hiệu lực", "áp dụng", "hiện hành"]

ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
POINT_RE = re.compile(r"điểm\s+([a-zđ])", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

LEVEL_ORDER = {"document": 0, "article": 1, "clause": 2, "point": 3}
NODE_TO_LEVEL = {
    "summary": "document",
    "article": "article",
    "section": "article",
    "subsection": "article",
    "clause": "clause",
    "point": "point",
    "bullet": "point",
    "text": "point",
    "amendment": "clause",
    "effective": "clause",
    "transition": "clause",
    "responsibility": "clause",
    "applicability": "clause",
}

_BM25_CACHE: Dict[str, Any] = {}

QUERY_ANALYSIS_SYSTEM_PROMPT = """
Bạn là bộ phân tích truy vấn cho GraphRAG pháp luật Việt Nam.

Mục tiêu:
- Hiểu đúng ý định pháp lý của câu hỏi.
- Sinh biến thể truy vấn để tăng recall.
- Không tự thêm thông tin pháp lý ngoài câu hỏi gốc.
- Ưu tiên đúng phạm vi: document / điều / khoản / điểm nếu có.
- Chỉ trả về JSON hợp lệ, không markdown.

Schema:
{
  "mode": "general | change | effective | sanction",
  "target_level": "document | article | clause | point",
  "needs_version_reasoning": true,
  "explicit_refs": {
    "article": "Điều 3",
    "clause": "Khoản 2",
    "point": "Điểm a",
    "year": 2020
  },
  "query_variants": ["...", "...", "..."]
}
""".strip()


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


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


def _fallback_query_profile(question: str, *, n_variants: int = 3) -> Dict[str, Any]:
    q = _norm_space(question)
    article_match = ARTICLE_RE.search(q)
    clause_match = CLAUSE_RE.search(q)
    point_match = POINT_RE.search(q)
    year_match = YEAR_RE.search(q)

    if point_match:
        target_level = "point"
    elif clause_match:
        target_level = "clause"
    elif article_match:
        target_level = "article"
    else:
        target_level = "document"

    return {
        "mode": detect_query_mode(q),
        "target_level": target_level,
        "needs_version_reasoning": detect_query_mode(q) in {"change", "effective"},
        "explicit_refs": {
            "article": f"Điều {article_match.group(1)}" if article_match else None,
            "clause": f"Khoản {clause_match.group(1)}" if clause_match else None,
            "point": f"Điểm {point_match.group(1).lower()}" if point_match else None,
            "year": int(year_match.group(1)) if year_match else None,
        },
        "query_variants": [q][: max(1, n_variants + 1)],
    }


def _extract_json_payload(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return json.loads(text)


def analyze_query_with_llm(question: str, *, n_variants: int = 3) -> Dict[str, Any]:
    fallback = _fallback_query_profile(question, n_variants=n_variants)
    q = _norm_space(question)
    if not q:
        return fallback

    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=QUERY_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Phân tích truy vấn sau và sinh tối đa {n_variants} biến thể truy vấn.\n"
                        f"Câu hỏi gốc: {q}"
                    )
                ),
            ]
        )
        parsed = _extract_json_payload(getattr(response, "content", "") or "")
        out = dict(fallback)

        mode = str(parsed.get("mode") or "").strip().lower()
        if mode in {"general", "change", "effective", "sanction"}:
            out["mode"] = mode

        target_level = str(parsed.get("target_level") or "").strip().lower()
        if target_level in {"document", "article", "clause", "point"}:
            out["target_level"] = target_level

        out["needs_version_reasoning"] = bool(
            parsed.get("needs_version_reasoning", out["needs_version_reasoning"])
        )

        explicit_refs = dict(out.get("explicit_refs") or {})
        llm_refs = parsed.get("explicit_refs") or {}
        if isinstance(llm_refs, dict):
            for key in ["article", "clause", "point", "year"]:
                value = llm_refs.get(key)
                if value not in (None, ""):
                    explicit_refs[key] = value
        out["explicit_refs"] = explicit_refs

        variants = []
        seen = set()
        for item in [q] + list(parsed.get("query_variants") or []):
            s = _norm_space(str(item))
            key = s.lower()
            if not s or key in seen:
                continue
            seen.add(key)
            variants.append(s)
            if len(variants) >= n_variants + 1:
                break
        out["query_variants"] = variants or fallback["query_variants"]

        # hard override for explicit legal refs detected by regex
        fallback_refs = fallback["explicit_refs"]
        for key, value in fallback_refs.items():
            if value not in (None, ""):
                out["explicit_refs"][key] = value

        # derive level from explicit refs if needed
        if out["explicit_refs"].get("point"):
            out["target_level"] = "point"
        elif out["explicit_refs"].get("clause"):
            out["target_level"] = "clause"
        elif out["explicit_refs"].get("article"):
            out["target_level"] = "article"

        if out["mode"] in {"change", "effective"}:
            out["needs_version_reasoning"] = True

        return out
    except Exception as exc:
        logger.warning("LLM query analysis failed, fallback to heuristic: %s", exc)
        return fallback


def allowed_relations_for_mode(mode: str) -> Set[str]:
    if mode == "change":
        return STRUCTURAL_RELATIONS | CHANGE_RELATIONS
    if mode == "effective":
        return STRUCTURAL_RELATIONS | CHANGE_RELATIONS | {"RESPONSIBLE_FOR", "APPLIES_TO"}
    if mode == "sanction":
        return STRUCTURAL_RELATIONS | GENERAL_RELATIONS | {"REGULATES"}
    return STRUCTURAL_RELATIONS | GENERAL_RELATIONS | CHANGE_RELATIONS


def compute_version_bonus(node: Dict[str, Any], *, mode: str) -> float:
    md = node.get("metadata", {}) or {}
    status = str(md.get("version_status") or "").strip().lower()
    event_count = int(md.get("version_event_count") or 0)
    legal_role = str(md.get("legal_role") or "").strip().lower()
    node_type = str(node.get("node_type") or md.get("node_type") or "").strip().lower()

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


def _extract_node_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not payload:
        return None
    md = payload.get("metadata")
    if isinstance(md, dict) and md.get("node_id"):
        return str(md["node_id"])
    for key in ["node_id", "chunk_id", "id"]:
        if payload.get(key):
            return str(payload[key])
    return None


def _candidate_doc_key(metadata: Dict[str, Any]) -> str:
    law_name = str(metadata.get("law_name") or "").strip().lower()
    if law_name:
        return law_name

    document_id = str(metadata.get("document_id") or "").strip().lower()
    if document_id:
        return document_id

    file_name = str(metadata.get("file_name") or metadata.get("filename") or "").strip().lower()
    if file_name:
        return file_name

    chunk_id = str(metadata.get("chunk_id") or "").strip().lower()
    if "::" in chunk_id:
        return chunk_id.split("::")[0]

    source_path = str(metadata.get("source_path") or metadata.get("file_path") or "").strip().lower()
    if source_path:
        return source_path

    source = str(metadata.get("source") or "").strip().lower()
    if source:
        return source

    return "unknown"

def _build_runtime_retrieval_text(text: str, metadata: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> str:
    payload = payload or {}
    existing = str(payload.get("retrieval_text") or "").strip()
    if existing:
        return existing

    parts: List[str] = []
    for key in [
        "law_name",
        "law_type",
        "year",
        "source",
        "section",
        "subsection",
        "article",
        "clause",
        "point",
        "node_type",
        "legal_role",
        "action",
        "target_article",
        "target_clause",
        "target_point",
        "path_title",
    ]:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    parts.append(_norm_space(text))
    return "\n".join(parts).strip()


def _point_to_candidate(point: Any) -> Dict[str, Any]:
    payload = getattr(point, "payload", {}) or {}
    text = str(payload.get("text") or payload.get("snippet") or "").strip()
    metadata = dict(payload.get("metadata") or {})
    node_id = _extract_node_id_from_payload(payload)

    if isinstance(metadata, dict):
        for key in [
            "law_name", "law_type", "year", "source", "section", "subsection",
            "article", "clause", "point", "chunk_id", "node_id", "node_type",
            "legal_role", "path_title"
        ]:
            if key not in metadata and payload.get(key) is not None:
                metadata[key] = payload.get(key)

    if not node_id:
        node_id = str(metadata.get("chunk_id") or metadata.get("node_id") or "")

    retrieval_text = _build_runtime_retrieval_text(text, metadata, payload)
    return {
        "node_id": node_id,
        "text": text,
        "snippet": text[:500],
        "metadata": metadata,
        "payload": payload,
        "retrieval_text": retrieval_text,
        "rerank_text": retrieval_text,
        "dense_score": float(getattr(point, "score", 0.0) or 0.0),
        "doc_key": _candidate_doc_key(metadata),
    }


def _rrf_fuse(rank_lists: List[List[Dict[str, Any]]], *, k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, item in enumerate(ranked, start=1):
            node_id = str(item.get("node_id") or "")
            if not node_id:
                continue
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return scores


def _merge_variant_candidates(rank_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    fused = _rrf_fuse(rank_lists)
    merged: Dict[str, Dict[str, Any]] = {}

    for variant_idx, ranked in enumerate(rank_lists):
        for rank, item in enumerate(ranked, start=1):
            node_id = str(item.get("node_id") or "")
            if not node_id:
                continue
            base = merged.get(node_id)
            if base is None:
                base = dict(item)
                base["variant_hits"] = 0
                base["variant_ranks"] = []
                base["max_dense_score"] = float(item.get("dense_score", 0.0) or 0.0)
                merged[node_id] = base

            base["variant_hits"] += 1
            base["variant_ranks"].append({"variant_index": variant_idx, "rank": rank})
            base["max_dense_score"] = max(
                float(base.get("max_dense_score", 0.0) or 0.0),
                float(item.get("dense_score", 0.0) or 0.0),
            )
            if len(str(item.get("text") or "")) > len(str(base.get("text") or "")):
                base["text"] = item.get("text")
                base["snippet"] = item.get("snippet")
                base["retrieval_text"] = item.get("retrieval_text")
                base["rerank_text"] = item.get("rerank_text")
                base["metadata"] = item.get("metadata")
                base["payload"] = item.get("payload")

    out = []
    for node_id, item in merged.items():
        item["rrf_score"] = float(fused.get(node_id, 0.0))
        item["dense_score"] = float(item.get("max_dense_score", 0.0) or 0.0)
        out.append(item)

    out.sort(key=lambda x: (float(x.get("rrf_score", 0.0)), float(x.get("dense_score", 0.0))), reverse=True)
    return out


def retrieve_seed_candidates(
    question: str,
    *,
    query_profile: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query_profile = query_profile or analyze_query_with_llm(question)
    merged_filters = dict(infer_filters(question))
    if filters:
        merged_filters.update({k: v for k, v in filters.items() if v is not None and v != ""})
    explicit_refs = dict(query_profile.get("explicit_refs") or {})
    if explicit_refs.get("year") is not None and "year" not in merged_filters:
        merged_filters["year"] = explicit_refs["year"]

    variants = list(query_profile.get("query_variants") or [_norm_space(question)])
    client = get_qdrant_client()
    embeddings = get_embedings()
    per_variant_lists: List[List[Dict[str, Any]]] = []

    for variant in variants:
        try:
            vector = embeddings.embed_query(variant)
            raw_points = search_qdrant(
                client=client,
                query_vector=vector,
                top_k=top_k or settings.qdrant_top_k,
                filters=merged_filters,
            )
            per_variant_lists.append([_point_to_candidate(p) for p in raw_points])
        except Exception as exc:
            logger.warning("Variant retrieval failed for %r: %s", variant, exc)

    if not per_variant_lists:
        return []

    merged_candidates = _merge_variant_candidates(per_variant_lists)
    bm25_stats = get_bm25_stats_cached(settings.bm25_stats_path)
    ranked = hybrid_rank(
        question,
        merged_candidates,
        dense_key="dense_score",
        text_key="retrieval_text",
        alpha=0.58,
        idf_map=bm25_stats.get("idf", {}),
        avgdl=bm25_stats.get("avgdl", 0.0),
    )

    out: List[Dict[str, Any]] = []
    for hybrid_score, cand in ranked:
        item = dict(cand)
        item["hybrid_score"] = float(hybrid_score)
        item["query_variants"] = variants
        item["query_profile"] = query_profile
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


def expand_graph_from_seeds(
    graph: Dict[str, Any],
    seed_items: List[Dict[str, Any]],
    *,
    mode: str,
    max_hops: int = 2,
    max_nodes: int = 80,
) -> Dict[str, float]:
    allowed_relations = allowed_relations_for_mode(mode)
    node_idx = _node_index(graph)
    node_idx_by_norm = {_normalize_node_id(node_id): str(node_id) for node_id in node_idx.keys()}
    chunk_id_index: Dict[str, str] = {}
    for real_node_id, node in node_idx.items():
        md = node.get("metadata", {}) or {}
        chunk_id = md.get("chunk_id")
        norm_chunk_id = _normalize_node_id(chunk_id)
        if norm_chunk_id and norm_chunk_id not in chunk_id_index:
            chunk_id_index[norm_chunk_id] = real_node_id

    queue = deque()
    best_score: Dict[str, float] = {}

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
            continue

        seed_score = float(item.get("hybrid_score", item.get("dense_score", 0.0)) or 0.0)
        best_score[resolved_node_id] = max(best_score.get(resolved_node_id, 0.0), seed_score)
        queue.append((resolved_node_id, 0, seed_score))

    visited: Set[Tuple[str, int]] = set()
    while queue and len(best_score) < max_nodes:
        current_id, hop, current_score = queue.popleft()
        if (current_id, hop) in visited:
            continue
        visited.add((current_id, hop))
        if hop >= max_hops:
            continue

        for edge in _adjacency(graph).get(current_id, []):
            rel = str(edge.get("relation_type") or "").strip().upper()
            nxt = edge.get("target_id")
            if not nxt or rel not in allowed_relations or nxt not in node_idx:
                continue
            next_score = current_score * (0.82 if hop == 0 else 0.68)
            node = node_idx[nxt]
            if _is_evidence(node):
                next_score *= 1.05
            elif _is_summary(node):
                next_score *= 0.95
            if next_score > best_score.get(str(nxt), -1.0):
                best_score[str(nxt)] = next_score
                queue.append((str(nxt), hop + 1, next_score))

        for edge in _reverse_adjacency(graph).get(current_id, []):
            rel = str(edge.get("relation_type") or "").strip().upper()
            nxt = edge.get("source_id")
            if not nxt or rel not in allowed_relations or nxt not in node_idx:
                continue
            next_score = current_score * (0.82 if hop == 0 else 0.68)
            node = node_idx[nxt]
            if _is_evidence(node):
                next_score *= 1.05
            elif _is_summary(node):
                next_score *= 0.95
            if next_score > best_score.get(str(nxt), -1.0):
                best_score[str(nxt)] = next_score
                queue.append((str(nxt), hop + 1, next_score))

    return best_score


def compute_scope_score(query_profile: Dict[str, Any], metadata: Dict[str, Any]) -> float:
    md = metadata or {}
    target_level = str(query_profile.get("target_level") or "document")
    node_type = str(md.get("node_type") or "").strip().lower()
    node_level = NODE_TO_LEVEL.get(node_type, "document")

    score = 0.0
    if node_level == target_level:
        score += 0.40
    elif LEVEL_ORDER.get(node_level, 0) > LEVEL_ORDER.get(target_level, 0):
        score += 0.20
    else:
        score += 0.05

    refs = dict(query_profile.get("explicit_refs") or {})
    q_article = refs.get("article")
    q_clause = refs.get("clause")
    q_point = refs.get("point")

    md_article = md.get("article")
    md_clause = md.get("clause")
    md_point = md.get("point")

    if q_article:
        if str(md_article or "").strip().lower() == str(q_article).lower():
            score += 0.25
        elif md_article:
            score -= 0.15
    if q_clause:
        if str(md_clause or "").strip().lower() == str(q_clause).lower():
            score += 0.20
        elif md_clause:
            score -= 0.12
    if q_point:
        if str(md_point or "").strip().lower() == str(q_point).lower():
            score += 0.15
        elif md_point:
            score -= 0.10

    mode = str(query_profile.get("mode") or "general")
    legal_role = str(md.get("legal_role") or "").strip().lower()
    if mode == "change" and legal_role in {"amendment", "replacement", "transition"}:
        score += 0.10
    elif mode == "effective" and legal_role in {"effective", "transition", "responsibility"}:
        score += 0.10
    elif mode == "sanction" and legal_role in {"responsibility", "clause", "point"}:
        score += 0.08

    return float(score)


def _build_passage_from_node(
    node_id: str,
    node: Dict[str, Any],
    seed_item: Optional[Dict[str, Any]],
    graph_score: float,
    query_profile: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    text = str(node.get("text") or "").strip()
    if not text:
        return None

    md = dict(node.get("metadata", {}) or {})
    retrieval_text = _build_runtime_retrieval_text(text, md)
    hybrid_score = float((seed_item or {}).get("hybrid_score", 0.0) or 0.0)
    dense_score = float((seed_item or {}).get("dense_score", 0.0) or 0.0)
    scope_score = compute_scope_score(query_profile, md)
    evidence_bonus = 0.12 if md.get("artifact_type") == "evidence" else 0.0
    version_bonus = compute_version_bonus(node, mode=str(query_profile.get("mode") or "general"))
    initial_score = 0.44 * hybrid_score + 0.26 * float(graph_score) + 0.18 * scope_score + evidence_bonus + version_bonus

    return {
        "node_id": node_id,
        "text": text,
        "snippet": text[:700],
        "metadata": md,
        "retrieval_text": retrieval_text,
        "rerank_text": retrieval_text,
        "dense_score": dense_score,
        "hybrid_score": hybrid_score,
        "graph_score": float(graph_score),
        "scope_score": float(scope_score),
        "cross_score": float((seed_item or {}).get("cross_score", 0.0) or 0.0),
        "version_bonus": float(version_bonus),
        "initial_score": float(initial_score),
        "doc_key": _candidate_doc_key(md),
    }


def aggregate_documents(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in passages:
        buckets[str(p.get("doc_key") or "unknown")].append(p)

    out: List[Dict[str, Any]] = []
    for doc_key, items in buckets.items():
        hs = sorted([float(x.get("hybrid_score", 0.0) or 0.0) for x in items], reverse=True)
        ss = sorted([float(x.get("scope_score", 0.0) or 0.0) for x in items], reverse=True)
        gs = sorted([float(x.get("graph_score", 0.0) or 0.0) for x in items], reverse=True)
        max_hybrid = hs[0] if hs else 0.0
        mean_top3 = (sum(hs[:3]) / max(1, min(3, len(hs)))) if hs else 0.0
        mean_scope = (sum(ss[:3]) / max(1, min(3, len(ss)))) if ss else 0.0
        max_graph = gs[0] if gs else 0.0
        doc_score = 0.50 * max_hybrid + 0.25 * mean_top3 + 0.15 * mean_scope + 0.10 * max_graph

        md = items[0].get("metadata", {}) or {}
        out.append(
            {
                "doc_key": doc_key,
                "doc_score": float(doc_score),
                "law_name": md.get("law_name"),
                "source": md.get("source"),
                "items": sorted(items, key=lambda x: float(x.get("initial_score", 0.0)), reverse=True),
            }
        )

    out.sort(key=lambda x: float(x.get("doc_score", 0.0)), reverse=True)
    return out


def build_final_passage_pool(
    doc_groups: List[Dict[str, Any]],
    all_passages: List[Dict[str, Any]],
    *,
    top_docs: int = 3,
    per_doc_base: int = 3,
    rescue_global: int = 4,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    chosen_docs = doc_groups[:max(1, top_docs)]

    top1 = float(chosen_docs[0].get("doc_score", 0.0)) if chosen_docs else 0.0
    top2 = float(chosen_docs[1].get("doc_score", 0.0)) if len(chosen_docs) > 1 else 0.0

    for idx, doc in enumerate(chosen_docs):
        doc_score = float(doc.get("doc_score", 0.0))
        items = list(doc.get("items") or [])

        # adaptive per-doc width
        if idx == 0 and top1 - top2 > 0.10:
            take_n = per_doc_base + 1
        elif idx > 0 and top1 - doc_score > 0.15:
            take_n = max(2, per_doc_base - 1)
        else:
            take_n = per_doc_base

        for p in items[:take_n]:
            nid = str(p.get("node_id") or "")
            if nid and nid not in seen:
                item = dict(p)
                item["doc_score"] = doc_score
                selected.append(item)
                seen.add(nid)

    for p in sorted(all_passages, key=lambda x: float(x.get("initial_score", 0.0)), reverse=True):
        nid = str(p.get("node_id") or "")
        if nid and nid not in seen:
            item = dict(p)
            item.setdefault("doc_score", 0.0)
            selected.append(item)
            seen.add(nid)
            rescue_global -= 1
            if rescue_global <= 0:
                break

    return selected


def reorder_passages_by_article_tree(graph: Dict[str, Any], passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not passages:
        return passages

    node_idx = _node_index(graph)
    adjacency = _adjacency(graph)
    reverse = _reverse_adjacency(graph)
    first_id = str(passages[0].get("node_id") or "")
    if not first_id or first_id not in node_idx:
        return passages

    current = first_id
    visited: Set[str] = set()
    article_root_id = first_id
    while current and current not in visited:
        visited.add(current)
        node = node_idx.get(current, {})
        node_type = str(node.get("node_type") or "").strip().lower()
        if node_type == "article":
            article_root_id = current
            break
        parent_id = None
        for edge in reverse.get(current, []):
            if str(edge.get("relation_type") or "").strip().upper() == "HAS_CHILD":
                parent_id = edge.get("source_id")
                break
        if not parent_id:
            break
        current = str(parent_id)

    stack = [article_root_id]
    subtree_ids: Set[str] = set()
    while stack:
        nid = stack.pop()
        if nid in subtree_ids:
            continue
        subtree_ids.add(nid)
        for edge in adjacency.get(nid, []):
            if str(edge.get("relation_type") or "").strip().upper() == "HAS_CHILD":
                child = edge.get("target_id")
                if child:
                    stack.append(str(child))

    article_first = [p for p in passages if str(p.get("node_id") or "") in subtree_ids]
    others = [p for p in passages if str(p.get("node_id") or "") not in subtree_ids]
    return article_first + others


def collect_final_passages(
    graph: Dict[str, Any],
    seed_items: List[Dict[str, Any]],
    graph_scores: Dict[str, float],
    *,
    query_profile: Dict[str, Any],
    final_top_k: Optional[int] = None,
    cross_top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    node_idx = _node_index(graph)
    seed_by_id = {str(item.get("node_id") or ""): item for item in seed_items if item.get("node_id")}
    candidate_map: Dict[str, Dict[str, Any]] = {}

    for node_id, node in node_idx.items():
        if node_id in graph_scores:
            passage = _build_passage_from_node(
                node_id=node_id,
                node=node,
                seed_item=seed_by_id.get(node_id),
                graph_score=float(graph_scores.get(node_id, 0.0)),
                query_profile=query_profile,
            )
            if passage:
                candidate_map[node_id] = passage

    if not candidate_map:
        for item in seed_items:
            node_id = str(item.get("node_id") or "")
            text = str(item.get("text") or "").strip()
            if not node_id or not text:
                continue
            md = dict(item.get("metadata", {}) or {})
            passage = {
                "node_id": node_id,
                "text": text,
                "snippet": item.get("snippet") or text[:700],
                "metadata": md,
                "retrieval_text": item.get("retrieval_text") or _build_runtime_retrieval_text(text, md),
                "rerank_text": item.get("rerank_text") or _build_runtime_retrieval_text(text, md),
                "dense_score": float(item.get("dense_score", 0.0) or 0.0),
                "hybrid_score": float(item.get("hybrid_score", 0.0) or 0.0),
                "graph_score": 0.0,
                "scope_score": compute_scope_score(query_profile, md),
                "cross_score": 0.0,
                "version_bonus": compute_version_bonus({"metadata": md, "node_type": md.get("node_type")}, mode=str(query_profile.get("mode") or "general")),
                "initial_score": float(item.get("hybrid_score", 0.0) or 0.0),
                "doc_key": _candidate_doc_key(md),
            }
            candidate_map[node_id] = passage

    all_passages = sorted(candidate_map.values(), key=lambda x: float(x.get("initial_score", 0.0)), reverse=True)
    doc_groups = aggregate_documents(all_passages)
    candidate_pool = build_final_passage_pool(
        doc_groups,
        all_passages,
        top_docs=min(4, len(doc_groups)),
        per_doc_base=3,
        rescue_global=4,
    )

    cross_limit = min(cross_top_k or settings.cross_top_k, len(candidate_pool))
    if cross_limit > 0:
        reranked = cross_rerank(
            question=str(query_profile.get("original_question") or ""),
            passages=candidate_pool[:cross_limit],
            top_n=cross_limit,
        )
        reranked_ids = {str(p.get("node_id") or "") for p in reranked}
        remaining = [p for p in candidate_pool[cross_limit:] if str(p.get("node_id") or "") not in reranked_ids]
        candidate_pool = reranked + remaining

    for p in candidate_pool:
        p["final_score"] = (
            0.35 * float(p.get("cross_score", 0.0) or 0.0)
            + 0.25 * float(p.get("hybrid_score", 0.0) or 0.0)
            + 0.20 * float(p.get("doc_score", 0.0) or 0.0)
            + 0.10 * float(p.get("graph_score", 0.0) or 0.0)
            + 0.10 * float(p.get("scope_score", 0.0) or 0.0)
            + float(p.get("version_bonus", 0.0) or 0.0)
        )

    candidate_pool.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    final_passages = reorder_passages_by_article_tree(graph, candidate_pool[: (final_top_k or settings.final_top_k)])
    for i, p in enumerate(final_passages, start=1):
        p["rank"] = i
    return final_passages, doc_groups, candidate_pool


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
    query_profile = analyze_query_with_llm(question, n_variants=3)
    query_profile["original_question"] = question

    merged_filters = dict(infer_filters(question))
    if filters:
        merged_filters.update({k: v for k, v in filters.items() if v is not None and v != ""})
    explicit_refs = dict(query_profile.get("explicit_refs") or {})
    if explicit_refs.get("year") is not None:
        merged_filters.setdefault("year", explicit_refs["year"])

    graph = ensure_versioning_ready(graph)

    seeds = retrieve_seed_candidates(
        question,
        query_profile=query_profile,
        filters=merged_filters,
        top_k=qdrant_top_k or settings.qdrant_top_k,
    )

    graph_scores = expand_graph_from_seeds(
        graph=graph,
        seed_items=seeds,
        mode=str(query_profile.get("mode") or "general"),
        max_hops=graph_hops,
        max_nodes=max_graph_nodes,
    )

    passages, doc_groups, candidate_pool = collect_final_passages(
        graph=graph,
        seed_items=seeds,
        graph_scores=graph_scores,
        query_profile=query_profile,
        final_top_k=final_top_k or settings.final_top_k,
        cross_top_k=cross_top_k or settings.cross_top_k,
    )

    return {
        "question": question,
        "query_profile": query_profile,
        "filters": merged_filters,
        "mode": str(query_profile.get("mode") or detect_query_mode(question)),
        "qdrant_top_k": qdrant_top_k or settings.qdrant_top_k,
        "final_top_k": final_top_k or settings.final_top_k,
        "cross_top_k": cross_top_k or settings.cross_top_k,
        "seed_candidates": seeds,
        "graph_scores": graph_scores,
        "doc_groups": doc_groups,
        "candidate_pool": candidate_pool,
        "passages": passages,
    }
