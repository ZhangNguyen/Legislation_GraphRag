from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.app.settings import settings
from src.rag.auto_filter import infer_query_profile
from src.rag.bm25_corpus import load_bm25_stats
from src.rag.graph_builder import collect_subtree_nodes, collect_subtree_text
from src.rag.hybrid import bm25_score, extract_topic_phrases, tokenize
from src.rag.openai_clients import get_embedings
from src.rag.rerank_cross import cross_rerank

logger = logging.getLogger(__name__)

_TEXT_EMBED_CACHE: Dict[str, List[float]] = {}
_QUESTION_EMBED_CACHE: Dict[str, List[float]] = {}
_BM25_RUNTIME: Optional[Dict[str, Any]] = None

_DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DOC_TYPE_PATTERNS = {
    "bộ luật": "Bộ luật",
    "luật": "Luật",
    "nghị định": "Nghị định",
    "thông tư": "Thông tư",
    "quyết định": "Quyết định",
    "chỉ thị": "Chỉ thị",
    "nghị quyết": "Nghị quyết",
    "công điện": "Công điện",
}
_CHANGE_CUES = ["sửa đổi", "bổ sung", "bãi bỏ", "thay thế", "có hiệu lực", "hết hiệu lực", "hiệu lực"]


# =========================
# Basic helpers
# =========================
def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(_norm_space(text).encode("utf-8")).hexdigest()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


def _compact_upper(text: str) -> str:
    return re.sub(r"[^A-Z0-9Đ]+", "", str(text or "").upper())


def _first_sentences(text: str, *, limit: int = 320) -> str:
    clean = _norm_space(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


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


def _node_md(node: Dict[str, Any]) -> Dict[str, Any]:
    return dict(node.get("metadata") or {})


def _node_type(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or _node_md(node).get("node_type") or "").strip().lower()


def _artifact_type(node: Dict[str, Any]) -> str:
    return str(_node_md(node).get("artifact_type") or "evidence").strip().lower()


def _node_text(node: Dict[str, Any]) -> str:
    return _norm_space(str(node.get("retrieval_text") or node.get("text") or ""))


def _path_label(md: Dict[str, Any]) -> str:
    for key in ["path_title", "title", "heading_title", "article", "clause", "point", "label"]:
        val = md.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "-"


# =========================
# Runtime BM25 + embeddings
# =========================
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


def _bm25_runtime_score(question: str, text: str) -> float:
    stats = _get_bm25_runtime()
    return float(
        bm25_score(
            tokenize(question),
            tokenize(text),
            idf_map=dict(stats.get("idf") or {}),
            avgdl=float(stats.get("avgdl") or 0.0),
        )
    )


def _question_embedding(question: str) -> List[float]:
    q = _norm_space(question)
    key = _text_cache_key(q)
    cached = _QUESTION_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = list(get_embedings().embed_query(q))
    _QUESTION_EMBED_CACHE[key] = vec
    return vec


def _batch_embed_texts(texts: List[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    misses: List[Tuple[str, str]] = []
    for text in texts:
        key = _text_cache_key(text)
        if key in _TEXT_EMBED_CACHE:
            out[key] = _TEXT_EMBED_CACHE[key]
        else:
            misses.append((key, text))
    if misses:
        vectors = get_embedings().embed_documents([text for _, text in misses])
        for (key, _), vec in zip(misses, vectors):
            _TEXT_EMBED_CACHE[key] = list(vec)
            out[key] = _TEXT_EMBED_CACHE[key]
    return out


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


# =========================
# Query understanding
# =========================
def _extract_query_signals(question: str) -> Dict[str, str]:
    q = _norm_space(question)
    q_lower = q.lower()

    doc_number = ""
    m = _DOC_NUMBER_RE.search(q.upper())
    if m:
        doc_number = _norm_space(m.group(0)).upper()

    doc_type = ""
    for raw, canonical in _DOC_TYPE_PATTERNS.items():
        if raw in q_lower:
            doc_type = canonical
            break

    agency_hint = doc_number.split("/")[-1].upper() if doc_number else ""

    year = ""
    m_year = _YEAR_RE.search(q)
    if m_year:
        year = m_year.group(1)

    change_cue = ""
    for cue in _CHANGE_CUES:
        if cue in q_lower:
            change_cue = cue
            break

    return {
        "doc_number": doc_number,
        "doc_type": doc_type,
        "agency_hint": agency_hint,
        "year": year,
        "change_cue": change_cue,
    }


def _metadata_exactish_score(query_signals: Dict[str, str], doc: Dict[str, Any]) -> Dict[str, float]:
    doc_number = str(doc.get("doc_number") or "").upper()
    doc_type = _norm_space(str(doc.get("doc_type") or ""))
    issuing_agency = _norm_space(str(doc.get("issuing_agency") or ""))
    year = str(doc.get("year") or "")

    q_doc_number = str(query_signals.get("doc_number") or "").upper()
    q_doc_type = _norm_space(str(query_signals.get("doc_type") or ""))
    q_agency = _compact_upper(query_signals.get("agency_hint") or "")
    q_year = str(query_signals.get("year") or "")

    doc_number_exact = 0.0
    if q_doc_number and doc_number:
        if q_doc_number == doc_number:
            doc_number_exact = 1.0
        elif q_doc_number in doc_number or doc_number in q_doc_number:
            doc_number_exact = 0.85

    doc_type_exact = 1.0 if q_doc_type and doc_type and q_doc_type.lower() == doc_type.lower() else 0.0

    agency_exact = 0.0
    if q_agency:
        doc_agency_compact = _compact_upper(issuing_agency)
        doc_number_suffix = _compact_upper(doc_number.split("/")[-1] if doc_number else "")
        if q_agency == doc_agency_compact or q_agency == doc_number_suffix:
            agency_exact = 1.0
        elif q_agency in doc_agency_compact or q_agency in doc_number_suffix:
            agency_exact = 0.7

    year_exact = 1.0 if q_year and year and q_year == year else 0.0

    return {
        "doc_number_exact": doc_number_exact,
        "doc_type_exact": doc_type_exact,
        "agency_exact": agency_exact,
        "year_exact": year_exact,
    }


def _metadata_prior(query_signals: Dict[str, str], doc: Dict[str, Any]) -> float:
    scores = _metadata_exactish_score(query_signals, doc)
    return float(
        0.60 * scores["doc_number_exact"]
        + 0.18 * scores["agency_exact"]
        + 0.12 * scores["doc_type_exact"]
        + 0.10 * scores["year_exact"]
    )


def _phrase_signal(question: str, text: str) -> Tuple[float, List[str]]:
    phrases = extract_topic_phrases(question)
    if not phrases:
        return 0.0, []
    hay = _norm_space(text).lower()
    hits = [p for p in phrases if p and p in hay]
    return float(len(hits) / max(1, len(phrases))), hits


def _hybrid_score(dense: float, bm25_raw: float) -> float:
    alpha = float(getattr(settings, "hybrid_alpha", 0.60) or 0.60)
    bm25_norm = min(1.0, float(bm25_raw) / 8.0)
    return float(alpha * float(dense) + (1.0 - alpha) * bm25_norm)


def _reference_bonus(question: str, md: Dict[str, Any]) -> float:
    q = _norm_space(question).lower()
    bonus = 0.0
    article = _norm_space(str(md.get("article") or "")).lower()
    clause = _norm_space(str(md.get("clause") or "")).lower()
    point = _norm_space(str(md.get("point") or "")).lower()
    path = _norm_space(str(md.get("path_title") or md.get("title") or "")).lower()
    if article and article in q:
        bonus += 0.15
    if clause and clause in q:
        bonus += 0.18
    if point and point in q:
        bonus += 0.18
    if path and path in q:
        bonus += 0.08
    return bonus


def _structure_bonus(profile: Dict[str, Any], md: Dict[str, Any]) -> float:
    score = 0.0
    path = _norm_space(str(md.get("path_title") or md.get("title") or "")).lower()
    full = f"{path}\n{_norm_space(str(md.get('heading_title') or ''))}".lower()
    for term in list(profile.get("heading_terms") or []):
        if term and term in full:
            score += 0.10
    if list(profile.get("change_terms") or []):
        version_status = _norm_space(str(md.get("version_status") or "")).lower()
        event_count = _safe_int(md.get("version_event_count"), 0)
        if version_status and version_status != "base":
            score += 0.10
        if event_count > 0:
            score += min(0.12, 0.03 * event_count)
    return min(0.22, score)


def _reference_exact_match_score(md: Dict[str, Any], filters: Dict[str, Any]) -> float:
    article_q = _norm_space(str(filters.get("article") or "")).lower()
    clause_q = _norm_space(str(filters.get("clause") or "")).lower()
    point_q = _norm_space(str(filters.get("point") or "")).lower()
    article = _norm_space(str(md.get("article") or "")).lower()
    clause = _norm_space(str(md.get("clause") or "")).lower()
    point = _norm_space(str(md.get("point") or "")).lower()
    path = _norm_space(str(md.get("path_title") or md.get("title") or "")).lower()

    score = 0.0
    if article_q:
        if article == article_q:
            score += 0.50
        elif article_q in path:
            score += 0.20
        else:
            return 0.0
    if clause_q:
        if clause == clause_q:
            score += 0.30
        elif clause_q in path:
            score += 0.10
        else:
            return 0.0
    if point_q:
        if point == point_q:
            score += 0.20
        elif point_q in path:
            score += 0.08
        else:
            return 0.0
    return min(1.0, score)


def _should_use_dense(query_profile: Dict[str, Any], mode: str) -> bool:
    if mode in {"exact", "passage"} and bool(query_profile.get("explicit_reference")):
        return False
    return True


# =========================
# Doc catalog
# =========================
def _pick_best_doc_metadata(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not nodes:
        return {}

    def sort_key(node: Dict[str, Any]) -> Tuple[int, int]:
        ntype = _node_type(node)
        art = _artifact_type(node)
        md = _node_md(node)
        order = _safe_int(md.get("order_index"), 0)
        if ntype == "document":
            return (0, order)
        if art == "doc_sketch":
            return (1, order)
        if art in {"article_bundle", "section_bundle"}:
            return (2, order)
        if art == "evidence":
            return (3, order)
        return (9, order)

    chosen = sorted(nodes, key=sort_key)[0]
    return _node_md(chosen)


def _build_doc_catalog(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    buckets: Dict[str, Dict[str, Any]] = {}

    for node_id, node in node_idx.items():
        md = _node_md(node)
        doc_id = str(md.get("doc_id") or "").strip()
        if not doc_id:
            continue
        bucket = buckets.setdefault(
            doc_id,
            {
                "id": doc_id,
                "doc_id": doc_id,
                "nodes": [],
                "member_node_ids": [],
                "reference_node_ids": [],
                "article_bundle_ids": [],
                "section_bundle_ids": [],
                "evidence_node_ids": [],
                "doc_sketch_node_id": "",
            },
        )
        bucket["nodes"].append(node)
        bucket["member_node_ids"].append(node_id)
        art = _artifact_type(node)
        ntype = _node_type(node)
        if art == "doc_sketch":
            bucket["doc_sketch_node_id"] = node_id
        elif art == "article_bundle":
            bucket["article_bundle_ids"].append(node_id)
            bucket["evidence_node_ids"].append(node_id)
        elif art == "section_bundle":
            bucket["section_bundle_ids"].append(node_id)
            bucket["evidence_node_ids"].append(node_id)
        elif art == "evidence":
            bucket["evidence_node_ids"].append(node_id)
        if ntype in {"article", "clause", "point", "section", "item", "bullet"}:
            bucket["reference_node_ids"].append(node_id)

    docs: List[Dict[str, Any]] = []
    for doc_id, bucket in buckets.items():
        chosen_md = _pick_best_doc_metadata(bucket["nodes"])
        doc_type = _norm_space(str(chosen_md.get("doc_type") or chosen_md.get("law_type") or ""))
        official_title = _norm_space(str(chosen_md.get("official_title") or chosen_md.get("law_name") or doc_id))
        issuing_agency = _norm_space(str(chosen_md.get("issuing_agency") or chosen_md.get("source") or ""))
        doc_sketch = ""
        if bucket["doc_sketch_node_id"] and bucket["doc_sketch_node_id"] in node_idx:
            doc_sketch = _node_text(node_idx[bucket["doc_sketch_node_id"]])
        docs.append(
            {
                "id": doc_id,
                "doc_key": doc_id,
                "doc_id": doc_id,
                "official_title": official_title,
                "law_name": official_title,
                "doc_type": doc_type or "Unknown",
                "law_type": doc_type or "Unknown",
                "issuing_agency": issuing_agency,
                "source": issuing_agency or "LocalFile",
                "doc_number": _norm_space(str(chosen_md.get("doc_number") or "")).upper(),
                "year": _safe_int(chosen_md.get("year"), 0),
                "date_raw": _norm_space(str(chosen_md.get("date_raw") or "")),
                "doc_sketch": doc_sketch,
                "member_node_ids": list(dict.fromkeys(bucket["member_node_ids"])),
                "reference_node_ids": list(dict.fromkeys(bucket["reference_node_ids"])),
                "article_bundle_ids": list(dict.fromkeys(bucket["article_bundle_ids"])),
                "section_bundle_ids": list(dict.fromkeys(bucket["section_bundle_ids"])),
                "evidence_node_ids": list(dict.fromkeys(bucket["evidence_node_ids"])),
            }
        )
    docs.sort(key=lambda x: str(x.get("doc_id") or ""))
    return docs


# =========================
# Candidate builders
# =========================
def _iter_bundle_candidates(graph: Dict[str, Any], *, allowed_doc_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    for node_id, node in node_idx.items():
        md = _node_md(node)
        doc_id = str(md.get("doc_id") or "").strip()
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        if _artifact_type(node) not in {"article_bundle", "section_bundle"}:
            continue
        out.append(
            {
                "id": node_id,
                "node_id": node_id,
                "doc_id": doc_id,
                "doc_key": doc_id,
                "text": _node_text(node),
                "snippet": _first_sentences(_node_text(node), limit=320),
                "retrieval_text": _norm_space(str(node.get("retrieval_text") or _node_text(node))),
                "rerank_text_short": _norm_space(str(node.get("rerank_text") or node.get("retrieval_text") or _node_text(node))),
                "metadata": md,
            }
        )
    return out


def _iter_reference_candidates(graph: Dict[str, Any], *, allowed_doc_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    for node_id, node in node_idx.items():
        md = _node_md(node)
        doc_id = str(md.get("doc_id") or "").strip()
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        ntype = _node_type(node)
        if ntype not in {"article", "clause", "point", "section", "item", "bullet"}:
            continue
        text = _node_text(node)
        out.append(
            {
                "id": node_id,
                "node_id": node_id,
                "doc_id": doc_id,
                "doc_key": doc_id,
                "text": text,
                "snippet": _first_sentences(text, limit=320),
                "retrieval_text": _norm_space(str(node.get("retrieval_text") or text)),
                "rerank_text_short": _norm_space(str(node.get("rerank_text") or node.get("retrieval_text") or text)),
                "metadata": md,
            }
        )
    return out


def _score_candidates(
    question: str,
    candidates: List[Dict[str, Any]],
    docs_by_id: Dict[str, Dict[str, Any]],
    query_signals: Dict[str, str],
    query_profile: Dict[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    use_dense = _should_use_dense(query_profile, mode)
    q_vec = _question_embedding(question) if use_dense else []
    texts = [str(c.get("retrieval_text") or c.get("text") or "") for c in candidates]
    vecs = _batch_embed_texts(texts) if use_dense else {}
    q_phrases = extract_topic_phrases(question)
    filters = dict(query_profile.get("filters") or {})
    scored: List[Dict[str, Any]] = []

    for item, text in zip(candidates, texts):
        dense = _cosine(q_vec, vecs.get(_text_cache_key(text), [])) if use_dense else 0.0
        bm25_val = _bm25_runtime_score(question, text)
        hybrid = _hybrid_score(dense, bm25_val) if use_dense else min(1.0, bm25_val / 8.0)
        phrase_score, matched_phrases = _phrase_signal(question, text)
        md = dict(item.get("metadata") or {})
        doc = docs_by_id.get(str(item.get("doc_id") or ""), {})
        metadata_prior = _metadata_prior(query_signals, doc) #Xem document có khớp với tín hiệu câu hỏi không
        structure_bonus = _structure_bonus(query_profile, md) #Điểm thưởng nếu candidate nằm đúng phần cấu trúc mà câu hỏi cần.
        reference_bonus = _reference_bonus(question, md)
        reference_exact = _reference_exact_match_score(md, filters)

        phrase_gate_penalty = 0.0
        phrase_gate_bonus = 0.0
        if q_phrases:
            hit_count = len(matched_phrases)
            coverage = hit_count / max(1, len(q_phrases))
            phrase_gate_bonus = min(0.24, 0.24 * coverage)
            if mode == "content":
                if hit_count == 0:
                    phrase_gate_penalty = 0.22
                elif hit_count == 1 and len(q_phrases) >= 2:
                    phrase_gate_penalty = 0.06

        if mode == "exact":
            score = 0.34 * reference_exact + 0.24 * hybrid + 0.12 * phrase_score + 0.12 * metadata_prior + 0.10 * structure_bonus + 0.08 * reference_bonus + phrase_gate_bonus
        elif mode == "passage" and bool(query_profile.get("explicit_reference")):
            score = 0.40 * reference_exact + 0.22 * hybrid + 0.14 * phrase_score + 0.10 * metadata_prior + 0.08 * structure_bonus + 0.06 * reference_bonus + phrase_gate_bonus
        else:
            score = 0.42 * hybrid + 0.20 * phrase_score + 0.12 * structure_bonus + 0.10 * metadata_prior + 0.08 * reference_bonus + phrase_gate_bonus - phrase_gate_penalty

        enriched = dict(item)
        enriched.update(
            {
                "dense": float(dense),
                "bm25": float(bm25_val),
                "hybrid": float(hybrid),
                "phrase_score": float(phrase_score),
                "matched_topic_phrases": matched_phrases,
                "metadata_prior": float(metadata_prior),
                "structure_bonus": float(structure_bonus),
                "reference_bonus": float(reference_bonus),
                "reference_exact": float(reference_exact),
                "phrase_gate_bonus": float(phrase_gate_bonus),
                "phrase_gate_penalty": float(phrase_gate_penalty),
                "candidate_score": float(score),
                "mode": mode,
            }
        )
        scored.append(enriched)

    scored.sort(key=lambda x: float(x.get("candidate_score", 0.0)), reverse=True)
    return scored


# =========================
# Aggregation and rerank
# =========================
def _aggregate_candidates_to_docs(
    question: str,
    candidates: List[Dict[str, Any]],
    docs_by_id: Dict[str, Dict[str, Any]],
    query_signals: Dict[str, str],
) -> List[Dict[str, Any]]:
    bucketed: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        doc_id = str(item.get("doc_id") or "")
        if doc_id:
            bucketed.setdefault(doc_id, []).append(item)

    q_phrases = extract_topic_phrases(question)
    docs_ranked: List[Dict[str, Any]] = []
    for doc_id, items in bucketed.items():
        base_doc = dict(docs_by_id.get(doc_id) or {})
        items.sort(key=lambda x: float(x.get("candidate_score", 0.0)), reverse=True)
        top_scores = [float(x.get("candidate_score", 0.0)) for x in items[:2]]
        max_bundle = top_scores[0] if top_scores else 0.0
        mean_top2 = sum(top_scores) / max(1, len(top_scores)) if top_scores else 0.0
        phrase_hits = sorted({p for item in items[:4] for p in list(item.get("matched_topic_phrases") or [])})
        if q_phrases:
            phrase_coverage = len(phrase_hits) / max(1, len(q_phrases))
        else:
            phrase_coverage = max((float(x.get("phrase_score", 0.0)) for x in items[:2]), default=0.0)
        metadata_prior = _metadata_prior(query_signals, base_doc)
        version_bonus = max((float(x.get("structure_bonus", 0.0)) for x in items[:3]), default=0.0)
        doc_score = 0.45 * max_bundle + 0.25 * mean_top2 + 0.20 * phrase_coverage + 0.10 * max(metadata_prior, version_bonus)
        base_doc.update(
            {
                "doc_score": float(doc_score),
                "max_bundle_score": float(max_bundle),
                "mean_top2_bundle_score": float(mean_top2),
                "phrase_coverage": float(phrase_coverage),
                "matched_topic_phrases": phrase_hits,
                "metadata_prior": float(metadata_prior),
                "top_bundle_candidates": items[:3],
            }
        )
        docs_ranked.append(base_doc)

    docs_ranked.sort(key=lambda x: float(x.get("doc_score", 0.0)), reverse=True)
    return docs_ranked


def _build_doc_evidence_pack(doc: Dict[str, Any]) -> str:
    lines: List[str] = []
    if _norm_space(str(doc.get("official_title") or "")):
        lines.append(f"Văn bản: {doc.get('official_title')}")
    if _norm_space(str(doc.get("doc_number") or "")):
        lines.append(f"Số văn bản: {doc.get('doc_number')}")
    if _norm_space(str(doc.get("doc_type") or "")):
        lines.append(f"Loại văn bản: {doc.get('doc_type')}")
    if _norm_space(str(doc.get("doc_sketch") or "")):
        lines.append(f"Mô tả văn bản: {_first_sentences(str(doc.get('doc_sketch') or ''), limit=450)}")
    for idx, item in enumerate(list(doc.get("top_bundle_candidates") or [])[:2], start=1):
        md = dict(item.get("metadata") or {})
        lines.append(f"[Bằng chứng {idx}] {_path_label(md)}")
        lines.append(_first_sentences(str(item.get("text") or item.get("snippet") or ""), limit=420))
    return "\n".join(x for x in lines if _norm_space(x)).strip()


def _rerank_docs(question: str, docs: List[Dict[str, Any]], *, top_n: int) -> List[Dict[str, Any]]:
    if not docs:
        return []
    candidates: List[Dict[str, Any]] = []
    for doc in docs:
        item = dict(doc)
        item["rerank_text_short"] = _build_doc_evidence_pack(doc)
        item["retrieval_text"] = item["rerank_text_short"]
        candidates.append(item)
    reranked = cross_rerank(question, candidates, top_n=min(max(1, top_n), len(candidates)))
    for item in reranked:
        cross_score = float(item.get("cross_score", 0.0) or 0.0)
        item["doc_final_score"] = 0.65 * cross_score + 0.35 * float(item.get("doc_score", 0.0) or 0.0)
    reranked.sort(key=lambda x: float(x.get("doc_final_score", 0.0)), reverse=True)
    return reranked


# =========================
# Exact reference mode
# =========================
def _exact_reference_node_candidates(graph: Dict[str, Any], *, allowed_doc_ids: Set[str], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    article_q = _norm_space(str(filters.get("article") or "")).lower()
    clause_q = _norm_space(str(filters.get("clause") or "")).lower()
    point_q = _norm_space(str(filters.get("point") or "")).lower()
    out: List[Dict[str, Any]] = []
    for node_id, node in node_idx.items():
        md = _node_md(node)
        doc_id = str(md.get("doc_id") or "").strip()
        if doc_id not in allowed_doc_ids:
            continue
        node_type = _node_type(node)
        article = _norm_space(str(md.get("article") or "")).lower()
        clause = _norm_space(str(md.get("clause") or "")).lower()
        point = _norm_space(str(md.get("point") or "")).lower()
        if article_q and article != article_q:
            continue
        if clause_q and clause != clause_q:
            continue
        if point_q and point != point_q:
            continue
        if point_q and node_type != "point":
            continue
        if clause_q and not point_q and node_type != "clause":
            continue
        if article_q and not clause_q and not point_q and node_type != "article":
            continue
        out.append({
            "node_id": node_id,
            "doc_id": doc_id,
            "metadata": md,
            "node_type": node_type,
            "text": _node_text(node),
            "path_title": _path_label(md),
        })
    return out


def _build_exact_reference_passages(
    graph: Dict[str, Any],
    docs: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    filters = dict(query_profile.get("filters") or {})
    if not filters or not query_profile.get("explicit_reference"):
        return []
    allowed_doc_ids = {str(d.get("doc_id") or "") for d in docs if str(d.get("doc_id") or "")}
    if not allowed_doc_ids:
        return []
    matches = _exact_reference_node_candidates(graph, allowed_doc_ids=allowed_doc_ids, filters=filters)
    if not matches:
        return []
    docs_by_id = {str(d.get("doc_id") or ""): d for d in docs}
    matches.sort(
        key=lambda item: (
            _reference_exact_match_score(dict(item.get("metadata") or {}), filters),
            float((docs_by_id.get(str(item.get("doc_id") or ""), {}) or {}).get("doc_final_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    final_passages: List[Dict[str, Any]] = []
    node_idx = _node_index(graph)
    for item in matches[: max(1, limit)]:
        node_id = str(item.get("node_id") or "")
        node = node_idx.get(node_id)
        if not node:
            continue
        md = dict(item.get("metadata") or {})
        subtree_nodes = collect_subtree_nodes(graph, node_id)
        subtree_text = collect_subtree_text(graph, node_id) or _node_text(node)
        final_passages.append(
            {
                "id": node_id,
                "node_id": node_id,
                "doc_id": str(item.get("doc_id") or md.get("doc_id") or ""),
                "doc_key": str(item.get("doc_id") or md.get("doc_id") or ""),
                "metadata": md,
                "text": subtree_text,
                "local_text": subtree_text,
                "shared_text": f"Vị trí pháp lý: {_path_label(md)}" if _path_label(md) != "-" else "",
                "snippet": _first_sentences(subtree_text, limit=320),
                "final_score": float(_reference_exact_match_score(md, filters)),
                "final_context_node_ids": [str(n.get("node_id") or "") for n in subtree_nodes if str(n.get("node_id") or "")],
                "final_context_paths": [_path_label(_node_md(n)) for n in subtree_nodes if _path_label(_node_md(n)) != "-"],
            }
        )
    return final_passages


# =========================
# Passage mode
# =========================
def _build_passage_candidates_from_docs(
    question: str,
    graph: Dict[str, Any],
    docs: List[Dict[str, Any]],
    query_signals: Dict[str, str],
    query_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    allowed_doc_ids = {str(d.get("doc_id") or "") for d in docs if str(d.get("doc_id") or "")}
    node_candidates = _iter_reference_candidates(graph, allowed_doc_ids=allowed_doc_ids)
    docs_by_id = {str(d.get("doc_id") or ""): d for d in docs}
    scored = _score_candidates(question, node_candidates, docs_by_id, query_signals, query_profile, mode="passage")
    for item in scored:
        doc_prior = float((docs_by_id.get(str(item.get("doc_id") or ""), {}) or {}).get("doc_final_score", 0.0) or 0.0)
        item["final_candidate_score"] = 0.70 * float(item.get("candidate_score", 0.0)) + 0.30 * doc_prior
    scored.sort(key=lambda x: float(x.get("final_candidate_score", 0.0)), reverse=True)
    return scored


def _build_final_passage(item: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    node_idx = _node_index(graph)
    node_id = str(item.get("node_id") or "")
    node = node_idx.get(node_id)
    md = dict(item.get("metadata") or {})
    local_text = _norm_space(str(item.get("text") or item.get("snippet") or ""))
    shared_text = ""
    subtree_nodes: List[Dict[str, Any]] = []
    if node is not None:
        subtree_nodes = collect_subtree_nodes(graph, node_id)
        subtree_text = collect_subtree_text(graph, node_id)
        if subtree_text:
            local_text = subtree_text
        path = _path_label(md)
        if path and path != "-":
            shared_text = f"Vị trí pháp lý: {path}"
    out = dict(item)
    out["text"] = local_text
    out["local_text"] = local_text
    out["shared_text"] = shared_text
    out["snippet"] = _first_sentences(local_text, limit=320)
    out["final_score"] = float(item.get("cross_score", item.get("final_candidate_score", item.get("candidate_score", 0.0))) or 0.0)
    out["doc_key"] = str(item.get("doc_id") or md.get("doc_id") or "")
    out["final_context_node_ids"] = [str(n.get("node_id") or "") for n in subtree_nodes if str(n.get("node_id") or "")]
    out["final_context_paths"] = [_path_label(_node_md(n)) for n in subtree_nodes if _path_label(_node_md(n)) != "-"]
    return out


def _compact_doc_debug(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": str(doc.get("doc_id") or ""),
        "official_title": str(doc.get("official_title") or doc.get("law_name") or ""),
        "doc_number": str(doc.get("doc_number") or ""),
        "doc_type": str(doc.get("doc_type") or ""),
        "doc_score": float(doc.get("doc_score", doc.get("doc_final_score", 0.0)) or 0.0),
        "doc_final_score": float(doc.get("doc_final_score", 0.0) or 0.0),
        "phrase_hits": list(doc.get("matched_topic_phrases") or []),
    }


def _compact_passage_debug(item: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(item.get("metadata") or {})
    return {
        "node_id": str(item.get("node_id") or ""),
        "doc_id": str(item.get("doc_id") or md.get("doc_id") or ""),
        "path": _path_label(md),
        "article": str(md.get("article") or ""),
        "clause": str(md.get("clause") or ""),
        "point": str(md.get("point") or ""),
        "score": float(item.get("final_score", item.get("candidate_score", 0.0)) or 0.0),
    }


# =========================
# Compatibility helper
# =========================
def rank_top_documents_by_official_title(question: str, graph: Dict[str, Any], *, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    docs = _build_doc_catalog(graph)
    query_signals = _extract_query_signals(question)
    phrases = extract_topic_phrases(question)
    docs.sort(key=lambda d: _metadata_prior(query_signals, d), reverse=True)
    return docs[: max(1, int(top_k or getattr(settings, "doc_top_k", 3) or 3))], query_signals, phrases


# =========================
# Main entrypoint
# =========================
def retrieve_with_graph(
    question: str,
    graph: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    qdrant_top_k: Optional[int] = None,
    graph_hops: int = 2,
    max_graph_nodes: int = 80,
    final_top_k: Optional[int] = None,
    cross_top_k: Optional[int] = None,
    doc_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    # compatibility: router cũ vẫn truyền các tham số này
    del qdrant_top_k, graph_hops, max_graph_nodes

    final_top_k = int(final_top_k or getattr(settings, "final_top_k", 5) or 5)
    cross_top_k = int(cross_top_k or getattr(settings, "cross_top_k", 8) or 8)
    doc_top_k = int(doc_top_k or getattr(settings, "doc_top_k", 3) or 3)

    docs = _build_doc_catalog(graph)
    if not docs:
        return {
            "mode": "no_docs",
            "route_case": None,
            "query_profile": {},
            "topic_phrases": [],
            "top_docs": [],
            "top_docs_initial": [],
            "candidate_pool": [],
            "passages": [],
            "rerank_veto": True,
            "veto_reason": "no_docs",
            "insufficient_context": True,
            "query_signals": {},
            "final_context_paths": [],
            "final_context_node_ids": [],
            "reference_threshold": None,
            "max_reference_hybrid": None,
        }

    query_signals = _extract_query_signals(question)
    query_profile = infer_query_profile(question) or {}
    topic_phrases = extract_topic_phrases(question)

    merged_filters = dict(query_profile.get("filters") or {})
    if filters:
        merged_filters.update({k: v for k, v in filters.items() if v not in (None, "", [], {})})

    # đồng bộ query_profile với filters từ API
    query_profile["filters"] = merged_filters
    if any(k in merged_filters for k in ("article", "clause", "point")):
        query_profile["explicit_reference"] = True
    if merged_filters.get("doc_number"):
        query_profile["has_doc_number"] = True

    route_case = "TH2" if bool(
        query_profile.get("explicit_reference")
        or query_profile.get("has_doc_number")
        or query_profile.get("change_terms")
    ) else "TH1"

    docs_by_id = {str(doc.get("doc_id") or ""): doc for doc in docs}

    def _build_exact_subtree_passages(selected_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        exact_filters = dict(query_profile.get("filters") or {})
        article_q = _norm_space(str(exact_filters.get("article") or "")).lower()
        clause_q = _norm_space(str(exact_filters.get("clause") or "")).lower()
        point_q = _norm_space(str(exact_filters.get("point") or "")).lower()

        if not (article_q or clause_q or point_q):
            return []

        node_idx = _node_index(graph)
        allowed_doc_ids = {
            str(d.get("doc_id") or "")
            for d in selected_docs
            if str(d.get("doc_id") or "")
        }
        if not allowed_doc_ids:
            return []

        matched_nodes: List[Tuple[float, Dict[str, Any]]] = []
        for node_id, node in node_idx.items():
            md = _node_md(node)
            doc_id = str(md.get("doc_id") or "").strip()
            if doc_id not in allowed_doc_ids:
                continue

            ntype = _node_type(node)
            article = _norm_space(str(md.get("article") or "")).lower()
            clause = _norm_space(str(md.get("clause") or "")).lower()
            point = _norm_space(str(md.get("point") or "")).lower()

            if article_q and article != article_q:
                continue
            if clause_q and clause != clause_q:
                continue
            if point_q and point != point_q:
                continue

            if point_q and ntype != "point":
                continue
            if clause_q and not point_q and ntype != "clause":
                continue
            if article_q and not clause_q and not point_q and ntype != "article":
                continue

            doc_score = 0.0
            for d in selected_docs:
                if str(d.get("doc_id") or "") == doc_id:
                    doc_score = float(d.get("doc_final_score", d.get("doc_score", 0.0)) or 0.0)
                    break
            matched_nodes.append((doc_score, node))

        matched_nodes.sort(key=lambda x: x[0], reverse=True)
        if not matched_nodes:
            return []

        passages: List[Dict[str, Any]] = []
        for rank, (_, node) in enumerate(matched_nodes[: max(1, final_top_k)], start=1):
            md = _node_md(node)
            doc_id = str(md.get("doc_id") or "").strip()
            subtree_nodes = collect_subtree_nodes(graph, str(node.get("node_id") or ""))
            subtree_text = collect_subtree_text(graph, str(node.get("node_id") or ""))
            if not _norm_space(subtree_text):
                subtree_text = _node_text(node)

            path_title = _path_label(md)
            item = {
                "id": str(node.get("node_id") or ""),
                "node_id": str(node.get("node_id") or ""),
                "doc_id": doc_id,
                "doc_key": doc_id,
                "text": subtree_text,
                "snippet": _first_sentences(subtree_text, limit=700),
                "shared_text": "",
                "local_text": subtree_text,
                "retrieval_text": subtree_text,
                "rerank_text_short": subtree_text[:1200],
                "metadata": md,
                "final_score": float(max(0.01, 1.0 - 0.01 * (rank - 1))),
                "candidate_score": float(max(0.01, 1.0 - 0.01 * (rank - 1))),
                "path_title": path_title,
                "subtree_node_ids": [str(n.get("node_id") or "") for n in subtree_nodes],
            }
            passages.append(item)

        return passages

    # =========================
    # Stage 1: candidate retrieval
    # =========================
    if route_case == "TH2":
        shortlist = sorted(docs, key=lambda d: _metadata_prior(query_signals, d), reverse=True)
        allowed_doc_ids = {
            str(d.get("doc_id") or "")
            for d in shortlist[: max(8, doc_top_k * 3)]
            if _metadata_prior(query_signals, d) > 0.0
        }
        raw_candidates = _iter_reference_candidates(graph, allowed_doc_ids=allowed_doc_ids or None)
        candidate_mode = "exact"
    else:
        raw_candidates = _iter_bundle_candidates(graph)
        candidate_mode = "content"

    candidate_pool = _score_candidates(
        question,
        raw_candidates,
        docs_by_id,
        query_signals,
        query_profile,
        mode=candidate_mode,
    )

    if not candidate_pool:
        return {
            "mode": "no_candidates",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "top_docs": [],
            "top_docs_initial": [],
            "candidate_pool": [],
            "passages": [],
            "rerank_veto": True,
            "veto_reason": "no_candidates",
            "insufficient_context": True,
            "query_signals": query_signals,
            "final_context_paths": [],
            "final_context_node_ids": [],
            "reference_threshold": None,
            "max_reference_hybrid": max((float(x.get("hybrid", 0.0) or 0.0) for x in candidate_pool), default=None),
        }

    # =========================
    # Stage 2: doc aggregation + rerank
    # =========================
    doc_ranked = _aggregate_candidates_to_docs(
        question,
        candidate_pool[: max(24, cross_top_k * 4)],
        docs_by_id,
        query_signals,
    )
    top_docs_initial = doc_ranked[: max(6, doc_top_k * 2)]

    reranked_docs = _rerank_docs(
        question,
        top_docs_initial,
        top_n=max(doc_top_k, min(len(top_docs_initial), cross_top_k)),
    )

    if not reranked_docs:
        return {
            "mode": "doc_rerank_empty",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "top_docs": [],
            "top_docs_initial": [_compact_doc_debug(x) for x in top_docs_initial[:6]],
            "candidate_pool": [_compact_passage_debug(x) for x in candidate_pool[:12]],
            "passages": [],
            "rerank_veto": True,
            "veto_reason": "doc_rerank_empty",
            "insufficient_context": True,
            "query_signals": query_signals,
            "final_context_paths": [],
            "final_context_node_ids": [],
            "reference_threshold": float(getattr(settings, "cross_rerank_min_score", 0.05) or 0.05),
            "max_reference_hybrid": max((float(x.get("hybrid", 0.0) or 0.0) for x in candidate_pool), default=None),
        }

    best_doc_score = float(reranked_docs[0].get("doc_final_score", 0.0) or 0.0)
    min_accept = float(getattr(settings, "cross_rerank_min_score", 0.05) or 0.05)
    if best_doc_score < min_accept:
        return {
            "mode": "doc_rerank_low_confidence",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "top_docs": [_compact_doc_debug(x) for x in reranked_docs[:doc_top_k]],
            "top_docs_initial": [_compact_doc_debug(x) for x in top_docs_initial[:6]],
            "candidate_pool": [_compact_passage_debug(x) for x in candidate_pool[:12]],
            "passages": [],
            "rerank_veto": True,
            "veto_reason": "doc_rerank_low_confidence",
            "insufficient_context": True,
            "query_signals": query_signals,
            "final_context_paths": [],
            "final_context_node_ids": [],
            "reference_threshold": min_accept,
            "max_reference_hybrid": max((float(x.get("hybrid", 0.0) or 0.0) for x in candidate_pool), default=None),
        }

    docs_for_passages = reranked_docs[:1]
    if len(reranked_docs) >= 2:
        margin = float(reranked_docs[0].get("doc_final_score", 0.0) or 0.0) - float(reranked_docs[1].get("doc_final_score", 0.0) or 0.0)
        if margin < 0.08 and not bool(query_profile.get("explicit_reference")):
            docs_for_passages = reranked_docs[: min(2, len(reranked_docs))]

    # =========================
    # Stage 3: exact subtree mode for Điều/Khoản/Điểm
    # =========================
    exact_passages = _build_exact_subtree_passages(docs_for_passages)
    if exact_passages:
        final_context_paths = [
            _path_label(dict(p.get("metadata") or {}))
            for p in exact_passages
        ]
        final_context_node_ids = [
            str(p.get("node_id") or "")
            for p in exact_passages
        ]
        return {
            "mode": "exact_subtree",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "top_docs": [_compact_doc_debug(x) for x in reranked_docs[:doc_top_k]],
            "top_docs_initial": [_compact_doc_debug(x) for x in top_docs_initial[:6]],
            "candidate_pool": [_compact_passage_debug(x) for x in candidate_pool[:12]],
            "passages": exact_passages[:final_top_k],
            "rerank_veto": False,
            "veto_reason": "",
            "insufficient_context": False,
            "query_signals": query_signals,
            "final_context_paths": final_context_paths[:final_top_k],
            "final_context_node_ids": final_context_node_ids[:final_top_k],
            "reference_threshold": min_accept,
            "max_reference_hybrid": max((float(x.get("hybrid", 0.0) or 0.0) for x in candidate_pool), default=None),
        }

    # =========================
    # Stage 4: passage selection fallback
    # =========================
    allowed_doc_ids = {
        str(d.get("doc_id") or "")
        for d in docs_for_passages
        if str(d.get("doc_id") or "")
    }
    raw_passages = _iter_reference_candidates(graph, allowed_doc_ids=allowed_doc_ids or None)

    passage_pool = _score_candidates(
        question,
        raw_passages,
        docs_by_id,
        query_signals,
        query_profile,
        mode="passage",
    )

    if not passage_pool:
        return {
            "mode": "no_passages",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "top_docs": [_compact_doc_debug(x) for x in reranked_docs[:doc_top_k]],
            "top_docs_initial": [_compact_doc_debug(x) for x in top_docs_initial[:6]],
            "candidate_pool": [_compact_passage_debug(x) for x in candidate_pool[:12]],
            "passages": [],
            "rerank_veto": True,
            "veto_reason": "no_passages",
            "insufficient_context": True,
            "query_signals": query_signals,
            "final_context_paths": [],
            "final_context_node_ids": [],
            "reference_threshold": min_accept,
            "max_reference_hybrid": max((float(x.get("hybrid", 0.0) or 0.0) for x in candidate_pool), default=None),
        }

    final_passages: List[Dict[str, Any]] = []
    for item in passage_pool[:final_top_k]:
        md = dict(item.get("metadata") or {})
        text = str(item.get("text") or item.get("snippet") or "")
        final_item = dict(item)
        final_item["shared_text"] = ""
        final_item["local_text"] = text
        final_item["final_score"] = float(item.get("candidate_score", 0.0) or 0.0)
        final_passages.append(final_item)

    final_context_paths = [
        _path_label(dict(p.get("metadata") or {}))
        for p in final_passages
    ]
    final_context_node_ids = [
        str(p.get("node_id") or "")
        for p in final_passages
    ]

    return {
        "mode": "passage_fallback",
        "route_case": route_case,
        "query_profile": query_profile,
        "topic_phrases": topic_phrases,
        "top_docs": [_compact_doc_debug(x) for x in reranked_docs[:doc_top_k]],
        "top_docs_initial": [_compact_doc_debug(x) for x in top_docs_initial[:6]],
        "candidate_pool": [_compact_passage_debug(x) for x in candidate_pool[:12]],
        "passages": final_passages,
        "rerank_veto": False,
        "veto_reason": "",
        "insufficient_context": False,
        "query_signals": query_signals,
        "final_context_paths": final_context_paths,
        "final_context_node_ids": final_context_node_ids,
        "reference_threshold": min_accept,
        "max_reference_hybrid": max((float(x.get("hybrid", 0.0) or 0.0) for x in candidate_pool), default=None),
    }