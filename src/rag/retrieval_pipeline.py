from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.app.settings import settings
from src.rag.auto_filter import infer_query_profile
from src.rag.graph_builder import (
    collect_amendment_source_nodes,
    collect_subtree_nodes,
    collect_subtree_text,
)
from src.rag.hybrid import bm25_score, extract_topic_phrases, tokenize
from src.rag.openai_clients import get_embedings
from src.rag.rerank_cross import cross_rerank

logger = logging.getLogger(__name__)

_DOC_EMBED_CACHE: Dict[str, List[float]] = {}
_TEXT_EMBED_CACHE: Dict[str, List[float]] = {}
_QUESTION_EMBED_CACHE: Dict[str, List[float]] = {}


# =========================
# Basic helpers
# =========================
def _sanitize_official_title(doc_type: str, official_title: str, issuing_agency: str) -> str:
    title = _norm_space(official_title)
    dtype = _norm_space(doc_type).upper()
    agency = _norm_space(issuing_agency)

    if not title:
        return ""

    if dtype and title.upper() == dtype:
        return ""

    if agency and title.upper().endswith(agency.upper()):
        title = _norm_space(title[: -len(agency)].strip(" -,:;"))

    title = re.sub(r"\s{2,}", " ", title).strip()
    return title


def _sanitize_issuing_agency(issuing_agency: str, official_title: str) -> str:
    agency = _norm_space(issuing_agency)
    title = _norm_space(official_title)

    if not agency:
        return ""

    if title and agency.upper() == title.upper():
        return ""

    if title and agency.upper() in title.upper():
        return ""

    return agency

def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(_norm_space(text).encode("utf-8")).hexdigest()


def _question_embedding(question: str) -> List[float]:
    q = _norm_space(question)
    key = _text_cache_key(q)
    cached = _QUESTION_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = list(get_embedings().embed_query(q))
    _QUESTION_EMBED_CACHE[key] = vec
    return vec


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
        vectors = get_embedings().embed_documents([text for _, text in misses])
        for (key, _), vec in zip(misses, vectors):
            cache[key] = list(vec)
            out[key] = cache[key]

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


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _rrf_merge(rank_lists: List[List[str]], *, k: int = 60) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, item_id in enumerate(ranked, start=1):
            if not item_id:
                continue
            out[item_id] = out.get(item_id, 0.0) + 1.0 / (k + rank)
    return out


def _sorted_ids_by_score(items: List[Dict[str, Any]], score_key: str) -> List[str]:
    ranked = sorted(items, key=lambda x: float(x.get(score_key, 0.0) or 0.0), reverse=True)
    out: List[str] = []
    for item in ranked:
        score = float(item.get(score_key, 0.0) or 0.0)
        if score <= 0.0:
            continue
        item_id = str(item.get("id") or item.get("node_id") or "")
        if item_id:
            out.append(item_id)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


# =========================
# Query signal extraction
# =========================

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

_DATE_INTENT_TERMS = [
    "có hiệu lực",
    "hiệu lực",
    "từ ngày nào",
    "ngày nào",
    "ban hành ngày nào",
    "ngày ban hành",
    "ngày ký",
]


def _compact_upper(text: str) -> str:
    return re.sub(r"[^A-Z0-9Đ]+", "", str(text or "").upper())

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

    agency_hint = ""
    if doc_number:
        suffix = doc_number.split("/")[-1].upper()
        agency_hint = suffix

        if not doc_type:
            if suffix.startswith("TT-"):
                doc_type = "Thông tư"
            elif suffix.startswith("QĐ-"):
                doc_type = "Quyết định"
            elif suffix.startswith("NQ-"):
                doc_type = "Nghị quyết"
            elif suffix.startswith("CĐ-"):
                doc_type = "Công điện"
            elif suffix.startswith("NĐ-"):
                doc_type = "Nghị định"
            elif suffix.startswith("CT-"):
                doc_type = "Chỉ thị"

    year = ""
    y = _YEAR_RE.search(q)
    if y:
        year = y.group(1)

    date_intent = ""
    for term in _DATE_INTENT_TERMS:
        if term in q_lower:
            date_intent = term
            break

    return {
        "doc_number": doc_number,
        "doc_type": doc_type,
        "agency_hint": agency_hint,
        "year": year,
        "date_intent": date_intent,
    }

# =========================
# Graph/node helpers
# =========================

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


def _node_text(node: Dict[str, Any]) -> str:
    return _norm_space(str(node.get("text") or node.get("retrieval_text") or ""))


def _artifact_type(node: Dict[str, Any]) -> str:
    return str(_node_md(node).get("artifact_type") or "evidence").strip().lower()


def _path_label(md: Dict[str, Any]) -> str:
    for key in ["path_title", "title", "heading_title", "article", "clause", "point", "label"]:
        val = md.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "-"


def _first_sentences(text: str, *, limit: int = 320) -> str:
    clean = _norm_space(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _to_node_id_list(values: Any) -> List[str]:
    out: List[str] = []
    for v in list(values or []):
        if isinstance(v, dict):
            nid = str(v.get("node_id") or v.get("id") or "").strip()
            if nid:
                out.append(nid)
        else:
            nid = str(v or "").strip()
            if nid:
                out.append(nid)
    return out


# =========================
# Doc catalog
# =========================

def _pick_best_doc_metadata(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ưu tiên metadata theo đúng cấp document:
    1) node_type=document
    2) artifact_type=doc_sketch
    3) evidence/article_bundle đầu tiên còn lại
    """
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
        if art == "evidence":
            return (2, order)
        if art == "article_bundle":
            return (3, order)
        return (9, order)

    chosen = sorted(nodes, key=sort_key)[0]
    return _node_md(chosen)


def _build_doc_catalog(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)

    # Chỉ bucket theo doc_id thật. Không dùng official_title/file_stem/law_name để đoán.
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
                "doc_key": doc_id,
                "doc_id": doc_id,
                "nodes": [],
                "reference_node_ids_strict": [],
                "reference_node_ids": [],
                "evidence_node_ids": [],
                "member_node_ids": [],
                "doc_sketch_node_id": "",
            },
        )

        bucket["nodes"].append(node)
        bucket["member_node_ids"].append(node_id)

        ntype = _node_type(node)
        art = _artifact_type(node)

        if art == "doc_sketch":
            bucket["doc_sketch_node_id"] = node_id

        if art == "evidence":
            bucket["evidence_node_ids"].append(node_id)

        if ntype in {"article", "clause", "point"}:
            bucket["reference_node_ids_strict"].append(node_id)
            bucket["reference_node_ids"].append(node_id)
        elif ntype in {"section", "item"}:
            bucket["reference_node_ids"].append(node_id)

    docs: List[Dict[str, Any]] = []

    for doc_id, bucket in buckets.items():
        chosen_md = _pick_best_doc_metadata(bucket["nodes"])

        doc_type = _norm_space(str(chosen_md.get("doc_type") or chosen_md.get("law_type") or ""))
        issuing_agency = _norm_space(str(chosen_md.get("issuing_agency") or chosen_md.get("source") or ""))
        official_title = _norm_space(str(chosen_md.get("official_title") or chosen_md.get("law_name") or ""))

        official_title = _sanitize_official_title(doc_type, official_title, issuing_agency)
        issuing_agency = _sanitize_issuing_agency(issuing_agency, official_title)

        law_name = official_title or doc_id
        doc_number = _norm_space(str(chosen_md.get("doc_number") or "")).upper()
        year = _safe_int(chosen_md.get("year"), 0)
        date_raw = _norm_space(str(chosen_md.get("date_raw") or ""))

        doc_sketch = ""
        if bucket["doc_sketch_node_id"]:
            doc_sketch_node = node_idx.get(bucket["doc_sketch_node_id"])
            if doc_sketch_node:
                doc_sketch = _node_text(doc_sketch_node)

        docs.append(
            {
                "id": doc_id,
                "doc_key": doc_id,
                "doc_id": doc_id,
                "official_title": official_title,
                "law_name": law_name,
                "doc_type": doc_type,
                "law_type": doc_type,
                "source": issuing_agency,
                "issuing_agency": issuing_agency,
                "doc_number": doc_number,
                "year": year,
                "date_raw": date_raw,
                "doc_sketch": doc_sketch,
                "reference_node_ids_strict": list(dict.fromkeys(bucket["reference_node_ids_strict"])),
                "reference_node_ids": list(dict.fromkeys(bucket["reference_node_ids"])),
                "evidence_node_ids": list(dict.fromkeys(bucket["evidence_node_ids"])),
                "member_node_ids": list(dict.fromkeys(bucket["member_node_ids"])),
                "doc_sketch_node_id": bucket["doc_sketch_node_id"],
            }
        )

    docs.sort(key=lambda x: str(x.get("doc_id") or ""))
    return docs


# =========================
# Metadata-first shortlist
# =========================

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
        else:
            q_parts = q_doc_number.split("/")
            d_parts = doc_number.split("/")
            if len(q_parts) >= 3 and len(d_parts) >= 3:
                # cùng năm + cùng hậu tố văn bản
                if q_parts[1] == d_parts[1] and q_parts[2] == d_parts[2]:
                    doc_number_exact = 0.35

    doc_type_exact = 0.0
    if q_doc_type and doc_type and q_doc_type.lower() == doc_type.lower():
        doc_type_exact = 1.0

    agency_exact = 0.0
    if q_agency:
        doc_agency_compact = _compact_upper(issuing_agency)
        doc_number_suffix = _compact_upper(doc_number.split("/")[-1] if doc_number else "")
        if q_agency == doc_agency_compact or q_agency == doc_number_suffix:
            agency_exact = 1.0
        elif q_agency and (q_agency in doc_agency_compact or q_agency in doc_number_suffix):
            agency_exact = 0.7

    year_exact = 0.0
    if q_year and year and q_year == year:
        year_exact = 1.0

    return {
        "doc_number_exact": doc_number_exact,
        "doc_type_exact": doc_type_exact,
        "agency_exact": agency_exact,
        "year_exact": year_exact,
    }


def _metadata_shortlist_docs(question: str, docs: List[Dict[str, Any]], *, shortlist_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    query_signals = _extract_query_signals(question)
    rescored: List[Dict[str, Any]] = []

    for doc in docs:
        item = dict(doc)
        scores = _metadata_exactish_score(query_signals, item)
        item.update(scores)
        item["metadata_shortlist_score"] = (
            3.0 * scores["doc_number_exact"]
            + 1.5 * scores["agency_exact"]
            + 0.8 * scores["doc_type_exact"]
            + 0.3 * scores["year_exact"]
        )
        item["query_signals"] = query_signals
        rescored.append(item)

    rescored.sort(key=lambda x: float(x.get("metadata_shortlist_score", 0.0)), reverse=True)

    if not rescored:
        return [], query_signals

    if float(rescored[0].get("metadata_shortlist_score", 0.0)) <= 0.0:
        return rescored[: max(1, shortlist_k)], query_signals

    return rescored[: max(1, shortlist_k)], query_signals


# =========================
# Top-doc ranking
# =========================

def _doc_title_text(doc: Dict[str, Any]) -> str:
    return _norm_space(str(doc.get("official_title") or doc.get("law_name") or ""))


def rank_top_documents_by_official_title(question: str, graph: Dict[str, Any], *, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    docs = _build_doc_catalog(graph)
    if not docs:
        return [], {}, []

    top_k = int(top_k or getattr(settings, "doc_top_k", 3) or 3)
    shortlist_k = max(top_k * 8, 20)

    metadata_shortlist, query_signals = _metadata_shortlist_docs(question, docs, shortlist_k=shortlist_k)
    topic_phrases = extract_topic_phrases(question)

    title_texts = [_doc_title_text(d) for d in metadata_shortlist]
    q_vec = _question_embedding(question)
    title_vecs = _batch_embed_texts(title_texts, cache=_DOC_EMBED_CACHE)

    scored: List[Dict[str, Any]] = []
    for doc, title_text in zip(metadata_shortlist, title_texts):
        item = dict(doc)
        title_key = _text_cache_key(title_text)
        title_dense = _cosine(q_vec, title_vecs.get(title_key, []))
        title_bm25 = bm25_score(tokenize(question), tokenize(title_text))

        doc_sketch = _norm_space(str(item.get("doc_sketch") or ""))
        matched_topic_phrases = [
            p for p in topic_phrases
            if p and (p in title_text.lower() or p in doc_sketch.lower())
        ]
        topic_gate_bonus = 0.12 if matched_topic_phrases else 0.0

        item["title_dense"] = float(title_dense)
        item["title_bm25"] = float(title_bm25)
        item["matched_topic_phrases"] = matched_topic_phrases
        item["topic_gate_bonus"] = topic_gate_bonus
        scored.append(item)

    dense_norm = _minmax([float(x["title_dense"]) for x in scored])
    bm25_norm = _minmax([float(x["title_bm25"]) for x in scored])

    for item, dnorm, bnorm in zip(scored, dense_norm, bm25_norm):
        item["title_dense_norm"] = dnorm
        item["title_bm25_norm"] = bnorm

    rank_doc_number = _sorted_ids_by_score(scored, "doc_number_exact")
    rank_agency = _sorted_ids_by_score(scored, "agency_exact")
    rank_title_dense = _sorted_ids_by_score(scored, "title_dense_norm")
    rank_title_bm25 = _sorted_ids_by_score(scored, "title_bm25_norm")

    rrf = _rrf_merge(
        [rank_doc_number, rank_agency, rank_title_dense, rank_title_bm25],
        k=60,
    )

    for item in scored:
        item["title_rrf"] = float(rrf.get(str(item.get("id") or ""), 0.0))
        item["doc_score"] = (
            float(item.get("title_rrf", 0.0))
            + float(item.get("topic_gate_bonus", 0.0))
            + 0.03 * float(item.get("metadata_shortlist_score", 0.0))
        )

    scored.sort(key=lambda x: float(x.get("doc_score", 0.0)), reverse=True)
    return scored[: max(1, top_k)], query_signals, topic_phrases




def _hybrid_score(dense: float, bm25: float) -> float:
    alpha = float(getattr(settings, "hybrid_alpha", 0.55) or 0.55)
    bm25_norm = min(1.0, float(bm25) / 3.0)
    return float(alpha * float(dense) + (1.0 - alpha) * bm25_norm)


def _route_case(question: str, query_signals: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    profile = infer_query_profile(question)
    explicit_reference = bool(profile.get("explicit_reference"))
    has_doc_number = bool(profile.get("has_doc_number")) or bool(query_signals.get("doc_number"))
    route = str(profile.get("route") or "")

    if explicit_reference or has_doc_number or route in {"direct_reference", "document_focus", "version_change"}:
        return "TH2", profile
    return "TH1", profile


def _score_docs_via_doc_sketch(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return []

    q_vec = _question_embedding(question)
    texts: List[str] = []
    items: List[Dict[str, Any]] = []

    for doc in docs:
        sketch = _norm_space(str(doc.get("doc_sketch") or ""))
        descriptor = "\n".join(
            x
            for x in [
                f"Số văn bản: {doc.get('doc_number')}",
                f"Loại văn bản: {doc.get('doc_type')}",
                f"Tiêu đề: {doc.get('official_title') or doc.get('law_name')}",
                f"Cơ quan ban hành: {doc.get('issuing_agency') or doc.get('source')}",
                f"Tóm tắt cấu trúc: {sketch}",
            ]
            if _norm_space(x)
        ).strip()
        if not descriptor:
            continue
        item = dict(doc)
        item["doc_sketch_descriptor"] = descriptor
        items.append(item)
        texts.append(descriptor)

    vecs = _batch_embed_texts(texts, cache=_TEXT_EMBED_CACHE)
    ranked: List[Dict[str, Any]] = []
    for item, descriptor in zip(items, texts):
        key = _text_cache_key(descriptor)
        dense = _cosine(q_vec, vecs.get(key, []))
        bm25 = bm25_score(tokenize(question), tokenize(descriptor))
        item["doc_sketch_dense"] = float(dense)
        item["doc_sketch_bm25"] = float(bm25)
        item["doc_sketch_hybrid"] = _hybrid_score(dense, bm25)
        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            float(x.get("doc_sketch_hybrid", 0.0) or 0.0),
            float(x.get("doc_score", 0.0) or 0.0),
            float(x.get("metadata_shortlist_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ranked


def _select_doc_for_general_question(question: str, docs: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], float]:
    ranked = _score_docs_via_doc_sketch(question, docs)
    threshold = float(getattr(settings, "doc_sketch_hybrid_threshold", 0.82) or 0.82)
    if not ranked:
        return None, [], threshold
    top_doc = ranked[0]
    if float(top_doc.get("doc_sketch_hybrid", 0.0) or 0.0) >= threshold:
        return top_doc, ranked, threshold
    return None, ranked, threshold


def _score_docs_via_bm25_exact(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    q_tokens = tokenize(question)
    for doc in docs:
        descriptor = "\n".join(
            x
            for x in [
                f"Số văn bản: {doc.get('doc_number')}",
                f"Loại văn bản: {doc.get('doc_type')}",
                f"Tiêu đề: {doc.get('official_title') or doc.get('law_name')}",
                f"Cơ quan ban hành: {doc.get('issuing_agency') or doc.get('source')}",
            ]
            if _norm_space(x)
        ).strip()
        item = dict(doc)
        item["exact_doc_descriptor"] = descriptor
        item["exact_doc_bm25"] = float(bm25_score(q_tokens, tokenize(descriptor)))
        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            float(x.get("exact_doc_bm25", 0.0) or 0.0),
            float(x.get("doc_number_exact", 0.0) or 0.0),
            float(x.get("agency_exact", 0.0) or 0.0),
            float(x.get("metadata_shortlist_score", 0.0) or 0.0),
            float(x.get("doc_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ranked


def _iter_single_doc_passage_nodes(doc: Dict[str, Any], graph: Dict[str, Any]) -> List[str]:
    node_idx = _node_index(graph)
    out: List[str] = []
    seen: Set[str] = set()
    for node_id in list(doc.get("member_node_ids") or []):
        node = node_idx.get(str(node_id or ""))
        if not node:
            continue
        art = _artifact_type(node)
        ntype = _node_type(node)
        if art == "doc_sketch":
            continue
        if art == "article_bundle" or ntype in {"article", "clause", "point", "item", "bullet", "section"}:
            nid = str(node_id)
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
    return out


def _build_single_doc_passage_candidates(
    question: str,
    graph: Dict[str, Any],
    doc: Dict[str, Any],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    q_vec = _question_embedding(question)
    candidates: List[Dict[str, Any]] = []

    for node_id in _iter_single_doc_passage_nodes(doc, graph):
        node = node_idx.get(node_id)
        if not node:
            continue

        md = _node_md(node)
        parent_id = str(md.get("parent_id") or "").strip()
        children_ids = [str(x).strip() for x in list(md.get("children_ids") or []) if str(x).strip()][:8]

        source_node_ids: List[str] = []
        if parent_id and parent_id in node_idx:
            source_node_ids.append(parent_id)
        source_node_ids.append(node_id)
        for cid in children_ids:
            if cid in node_idx and cid not in source_node_ids:
                source_node_ids.append(cid)

        text_parts: List[str] = []
        for sid in source_node_ids:
            src = node_idx.get(sid)
            if not src:
                continue
            st = _node_text(src)
            if st:
                text_parts.append(st)

        merged_text = "\n".join(t for t in text_parts if _norm_space(t)).strip()
        if not merged_text:
            continue

        path_title = _path_label(md)
        retrieval_text = "\n".join(
            x
            for x in [
                f"Văn bản: {doc.get('official_title') or doc.get('law_name')}",
                f"Số văn bản: {doc.get('doc_number')}",
                f"Vị trí pháp lý: {path_title}",
                f"Ngữ cảnh cha-con: {_first_sentences(merged_text, limit=900)}",
            ]
            if _norm_space(x)
        ).strip()

        key = _text_cache_key(retrieval_text)
        vec = _batch_embed_texts([retrieval_text], cache=_TEXT_EMBED_CACHE).get(key, [])
        dense = _cosine(q_vec, vec)
        bm25 = bm25_score(tokenize(question), tokenize(retrieval_text))
        hybrid = _hybrid_score(dense, bm25)

        candidates.append(
            {
                "id": f"{doc.get('doc_id')}::{node_id}",
                "node_id": node_id,
                "doc_id": str(doc.get("doc_id") or ""),
                "doc_key": str(doc.get("doc_id") or ""),
                "doc_score": float(doc.get("doc_score", 0.0) or 0.0),
                "text": merged_text,
                "snippet": _first_sentences(merged_text, limit=320),
                "retrieval_text": retrieval_text,
                "rerank_text_short": retrieval_text,
                "metadata": md,
                "path_title": path_title,
                "source_node_ids": source_node_ids,
                "content_dense": dense,
                "content_bm25": bm25,
                "content_hybrid": hybrid,
            }
        )

    candidates.sort(key=lambda x: float(x.get("content_hybrid", 0.0) or 0.0), reverse=True)
    return candidates[: max(1, top_k)]


def _exact_doc_route(
    question: str,
    graph: Dict[str, Any],
    docs: List[Dict[str, Any]],
    *,
    cross_top_k: int,
    final_top_k: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool, Optional[str], List[Dict[str, Any]]]:
    ranked_docs = _score_docs_via_bm25_exact(question, docs)
    if not ranked_docs:
        return None, [], [], True, "no_exact_doc_match", []

    chosen_doc = ranked_docs[0]
    passage_top_k = int(getattr(settings, "th2_passage_top_k", max(cross_top_k * 3, 12)) or max(cross_top_k * 3, 12))
    passage_candidates = _build_single_doc_passage_candidates(
        question,
        graph,
        chosen_doc,
        top_k=passage_top_k,
    )
    if not passage_candidates:
        return chosen_doc, [], [], True, "no_exact_doc_passages", ranked_docs

    reranked = cross_rerank(
        question,
        passage_candidates[: max(1, cross_top_k * 3)],
        top_n=min(max(1, cross_top_k), len(passage_candidates)),
    )
    if not reranked:
        return chosen_doc, passage_candidates, [], True, "cross_empty", ranked_docs

    best_cross = float(reranked[0].get("cross_score", 0.0) or 0.0)
    min_cross = float(getattr(settings, "cross_rerank_min_score", 0.05) or 0.05)
    if best_cross < min_cross:
        return chosen_doc, passage_candidates, [], True, "cross_score_too_low", ranked_docs

    final_passages = [
        _build_final_bundle_from_grouped_candidate(item, graph)
        for item in reranked[: max(1, final_top_k)]
    ]
    return chosen_doc, passage_candidates, final_passages, False, None, ranked_docs

# =========================
# Candidate building
# =========================

def _collect_reference_candidates(question: str, graph: Dict[str, Any], docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    q_vec = _question_embedding(question)
    out: List[Dict[str, Any]] = []

    texts_for_embed: List[str] = []
    temp_items: List[Dict[str, Any]] = []

    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        for node_id in list(doc.get("reference_node_ids") or []):
            node = node_idx.get(node_id)
            if not node:
                continue
            md = _node_md(node)
            node_type = _node_type(node)
            path_title = _path_label(md)
            self_text = _first_sentences(_node_text(node), limit=360)

            anchor_text = "\n".join(
                x for x in [
                    f"Văn bản: {doc.get('official_title')}",
                    f"Vị trí: {path_title}",
                    self_text,
                ] if _norm_space(x)
            ).strip()

            item = {
                "id": node_id,
                "node_id": node_id,
                "doc_id": doc_id,
                "doc_key": doc_id,
                "doc_score": float(doc.get("doc_score", 0.0)),
                "doc_title": doc.get("official_title"),
                "doc_number": doc.get("doc_number"),
                "text": _node_text(node),
                "snippet": self_text,
                "metadata": md,
                "node_type": node_type,
                "path_title": path_title,
                "retrieval_text": anchor_text,
                "rerank_text_short": anchor_text,
                "source_node_ids": [node_id],
            }
            temp_items.append(item)
            texts_for_embed.append(anchor_text)

    vecs = _batch_embed_texts(texts_for_embed, cache=_TEXT_EMBED_CACHE)

    for item, text in zip(temp_items, texts_for_embed):
        key = _text_cache_key(text)
        dense = _cosine(q_vec, vecs.get(key, []))
        bm25 = bm25_score(tokenize(question), tokenize(text))
        hybrid = 0.55 * dense + 0.45 * min(1.0, bm25 / 3.0)

        item["reference_dense"] = dense
        item["reference_bm25"] = bm25
        item["reference_hybrid"] = hybrid
        out.append(item)

    out.sort(key=lambda x: float(x.get("reference_hybrid", 0.0)), reverse=True)
    return out


def _group_content_candidates_by_siblings(
    question: str,
    graph: Dict[str, Any],
    docs: List[Dict[str, Any]],
    *,
    per_doc_top_k: int,
) -> List[Dict[str, Any]]:
    node_idx = _node_index(graph)
    q_vec = _question_embedding(question)
    grouped: List[Dict[str, Any]] = []

    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        seen_groups: Set[Tuple[str, ...]] = set()
        candidates: List[Dict[str, Any]] = []

        for node_id in list(doc.get("evidence_node_ids") or []):
            node = node_idx.get(node_id)
            if not node:
                continue

            md = _node_md(node)
            sibling_ids = list(md.get("sibling_ids") or [])
            group_ids = tuple(sorted(set([node_id] + sibling_ids)))
            if group_ids in seen_groups:
                continue
            seen_groups.add(group_ids)

            texts: List[str] = []
            path_title = ""
            group_nodes: List[Dict[str, Any]] = []

            for gid in group_ids:
                gnode = node_idx.get(gid)
                if not gnode:
                    continue
                group_nodes.append(gnode)
                gmd = _node_md(gnode)
                if not path_title:
                    path_title = _path_label(gmd)
                gtext = _node_text(gnode)
                if gtext:
                    texts.append(gtext)

            if not texts:
                continue

            merged_text = "\n".join(t for t in texts if _norm_space(t)).strip()
            retrieval_text = "\n".join(
                x for x in [
                    f"Văn bản: {doc.get('official_title')}",
                    f"Vị trí chung: {path_title}",
                    f"Nhóm sibling: {_first_sentences(merged_text, limit=700)}",
                ] if _norm_space(x)
            ).strip()

            dense_key = _text_cache_key(retrieval_text)
            vec = _batch_embed_texts([retrieval_text], cache=_TEXT_EMBED_CACHE).get(dense_key, [])
            dense = _cosine(q_vec, vec)
            bm25 = bm25_score(tokenize(question), tokenize(retrieval_text))
            hybrid = 0.55 * dense + 0.45 * min(1.0, bm25 / 3.0)

            candidates.append(
                {
                    "id": "::".join(group_ids),
                    "node_id": group_ids[0],
                    "doc_id": doc_id,
                    "doc_key": doc_id,
                    "doc_score": float(doc.get("doc_score", 0.0)),
                    "text": merged_text,
                    "snippet": _first_sentences(merged_text, limit=320),
                    "retrieval_text": retrieval_text,
                    "rerank_text_short": retrieval_text,
                    "metadata": dict(_node_md(group_nodes[0])) if group_nodes else {},
                    "path_title": path_title,
                    "source_node_ids": list(group_ids),
                    "content_dense": dense,
                    "content_bm25": bm25,
                    "content_hybrid": hybrid,
                }
            )

        candidates.sort(key=lambda x: float(x.get("content_hybrid", 0.0)), reverse=True)
        grouped.extend(candidates[: max(1, per_doc_top_k)])

    return grouped


# =========================
# Late expansion
# =========================

def _build_final_bundle_from_anchor(candidate: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    node_idx = _node_index(graph)
    node_id = str(candidate.get("node_id") or "")
    node = node_idx.get(node_id)
    md = dict(candidate.get("metadata") or {})

    if not node:
        out = dict(candidate)
        out["expanded_node_ids"] = [node_id] if node_id else []
        out["local_text"] = str(candidate.get("text") or "")
        out["shared_text"] = f"Văn bản: {md.get('official_title') or md.get('law_name') or ''}"
        out["final_score"] = float(
            candidate.get("cross_score", candidate.get("reference_hybrid", candidate.get("content_hybrid", 0.0))) or 0.0
        )
        return out

    subtree_ids_raw = collect_subtree_nodes(graph, node_id)
    amendment_sources_raw = collect_amendment_source_nodes(graph, node_id)

    subtree_ids = _to_node_id_list(subtree_ids_raw)
    amendment_source_ids = _to_node_id_list(amendment_sources_raw)
    subtree_text = collect_subtree_text(graph, node_id)

    amendment_texts: List[str] = []
    for sid in amendment_source_ids:
        src_node = node_idx.get(sid)
        if not src_node:
            continue
        st = _node_text(src_node)
        if st:
            amendment_texts.append(st)

    local_parts: List[str] = []
    if subtree_text:
        local_parts.append(subtree_text)
    if amendment_texts:
        local_parts.append("Nguồn sửa đổi/bổ sung/bãi bỏ liên quan:\n" + "\n".join(amendment_texts))

    expanded_node_ids: List[str] = []
    seen: Set[str] = set()
    for nid in [node_id] + subtree_ids + amendment_source_ids:
        nid = str(nid or "").strip()
        if nid and nid not in seen:
            seen.add(nid)
            expanded_node_ids.append(nid)

    out = dict(candidate)
    out["expanded_node_ids"] = expanded_node_ids
    out["local_text"] = "\n\n".join(x for x in local_parts if _norm_space(x)).strip()
    out["shared_text"] = "\n".join(
        x for x in [
            f"Văn bản: {md.get('official_title') or md.get('law_name') or ''}",
            f"Vị trí: {candidate.get('path_title') or _path_label(md)}",
        ] if _norm_space(x)
    ).strip()
    out["final_score"] = float(
        candidate.get("cross_score", candidate.get("reference_hybrid", candidate.get("content_hybrid", 0.0))) or 0.0
    )
    return out

def _build_final_bundle_from_grouped_candidate(candidate: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    node_idx = _node_index(graph)
    md = dict(candidate.get("metadata") or {})

    source_node_ids = _to_node_id_list(candidate.get("source_node_ids") or [])
    source_node_ids = [nid for nid in source_node_ids if nid in node_idx]

    grouped_text = _norm_space(str(candidate.get("text") or ""))
    grouped_snippet = _norm_space(str(candidate.get("snippet") or ""))
    path_title = _norm_space(str(candidate.get("path_title") or _path_label(md) or ""))

    amendment_source_ids: List[str] = []
    seen_amend: Set[str] = set()
    for nid in source_node_ids:
        for sid in _to_node_id_list(collect_amendment_source_nodes(graph, nid)):
            if sid and sid not in seen_amend:
                seen_amend.add(sid)
                amendment_source_ids.append(sid)

    amendment_texts: List[str] = []
    for sid in amendment_source_ids:
        src_node = node_idx.get(sid)
        if not src_node:
            continue
        st = _node_text(src_node)
        if st:
            amendment_texts.append(st)

    local_parts: List[str] = []
    if grouped_text:
        local_parts.append(grouped_text)
    elif grouped_snippet:
        local_parts.append(grouped_snippet)

    if amendment_texts:
        local_parts.append("Nguồn sửa đổi/bổ sung/bãi bỏ liên quan:\n" + "\n".join(amendment_texts))

    expanded_node_ids: List[str] = []
    seen: Set[str] = set()
    for nid in source_node_ids + amendment_source_ids:
        nid = str(nid or "").strip()
        if nid and nid not in seen:
            seen.add(nid)
            expanded_node_ids.append(nid)

    out = dict(candidate)
    out["expanded_node_ids"] = expanded_node_ids
    out["local_text"] = "\n\n".join(x for x in local_parts if _norm_space(x)).strip()
    out["shared_text"] = "\n".join(
        x
        for x in [
            f"Văn bản: {md.get('official_title') or md.get('law_name') or ''}",
            f"Vị trí: {path_title}",
        ]
        if _norm_space(x)
    ).strip()
    out["final_score"] = float(candidate.get("cross_score", candidate.get("content_hybrid", 0.0)) or 0.0)
    return out
# =========================
# Routes
# =========================

def _reference_route(
    question: str,
    graph: Dict[str, Any],
    docs: List[Dict[str, Any]],
    *,
    cross_top_k: int,
    final_top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool, Optional[str], float]:
    ref_candidates = _collect_reference_candidates(question, graph, docs)
    if not ref_candidates:
        return [], [], True, "no_reference_candidates", 0.0

    max_reference_hybrid = float(ref_candidates[0].get("reference_hybrid", 0.0) or 0.0)

    reranked = cross_rerank(
        question,
        ref_candidates[: max(1, cross_top_k * 3)],
        top_n=min(max(1, cross_top_k), len(ref_candidates)),
    )

    if not reranked:
        return ref_candidates, [], True, "cross_empty", max_reference_hybrid

    best_cross = float(reranked[0].get("cross_score", 0.0) or 0.0)
    min_cross = float(getattr(settings, "cross_rerank_min_score", 0.05) or 0.05)
    if best_cross < min_cross:
        return ref_candidates, [], True, "cross_score_too_low", max_reference_hybrid

    final_passages = [_build_final_bundle_from_anchor(item, graph) for item in reranked[: max(1, final_top_k)]]
    return ref_candidates, final_passages, False, None, max_reference_hybrid

def _content_route(
    question: str,
    graph: Dict[str, Any],
    docs: List[Dict[str, Any]],
    *,
    cross_top_k: int,
    final_top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool, Optional[str]]:
    per_doc_top_k = int(getattr(settings, "content_top_k_per_doc", 3) or 3)
    content_candidates = _group_content_candidates_by_siblings(
        question,
        graph,
        docs,
        per_doc_top_k=per_doc_top_k,
    )

    if not content_candidates:
        return [], [], True, "no_content_candidates"

    reranked = cross_rerank(
        question,
        content_candidates[: max(1, cross_top_k * 3)],
        top_n=min(max(1, cross_top_k), len(content_candidates)),
    )

    if not reranked:
        return content_candidates, [], True, "cross_empty"

    best_cross = float(reranked[0].get("cross_score", 0.0) or 0.0)
    min_cross = float(getattr(settings, "cross_rerank_min_score", 0.05) or 0.05)
    if best_cross < min_cross:
        return content_candidates, [], True, "cross_score_too_low"

    # GIỮ NGUYÊN grouped sibling candidate sau rerank, không rebuild lại theo anchor node
    final_passages = [
        _build_final_bundle_from_grouped_candidate(item, graph)
        for item in reranked[: max(1, final_top_k)]
    ]
    return content_candidates, final_passages, False, None

# =========================
# Public API
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
) -> Dict[str, Any]:
    del filters, qdrant_top_k, graph_hops, max_graph_nodes

    final_top_k = int(final_top_k or getattr(settings, "final_top_k", 5) or 5)
    cross_top_k = int(cross_top_k or getattr(settings, "cross_top_k", 8) or 8)

    top_docs, query_signals, topic_phrases = rank_top_documents_by_official_title(
        question,
        graph,
        top_k=int(getattr(settings, "doc_top_k", 3) or 3),
    )

    route_case, query_profile = _route_case(question, query_signals)

    if not top_docs:
        return {
            "mode": "no_top_docs",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "doc_sketch_hybrid_threshold": float(getattr(settings, "doc_sketch_hybrid_threshold", 0.82) or 0.82),
            "rerank_veto": True,
            "veto_reason": "no_top_docs",
            "top_docs": [],
            "candidate_pool": [],
            "passages": [],
            "insufficient_context": True,
            "query_signals": query_signals,
        }

    if route_case == "TH2":
        chosen_doc, candidate_pool, final_passages, veto, veto_reason, exact_doc_ranked = _exact_doc_route(
            question,
            graph,
            top_docs,
            cross_top_k=cross_top_k,
            final_top_k=final_top_k,
        )
        return {
            "mode": "th2_exact_doc_bm25_route",
            "route_case": route_case,
            "query_profile": query_profile,
            "topic_phrases": topic_phrases,
            "rerank_veto": veto,
            "veto_reason": veto_reason,
            "top_docs": [chosen_doc] if chosen_doc else [],
            "top_docs_initial": top_docs,
            "exact_doc_ranked": exact_doc_ranked,
            "candidate_pool": candidate_pool,
            "passages": final_passages,
            "insufficient_context": veto or not final_passages,
            "query_signals": query_signals,
        }

    selected_doc, doc_sketch_ranked, doc_sketch_threshold = _select_doc_for_general_question(question, top_docs)
    docs_for_passages = [selected_doc] if selected_doc else top_docs
    mode = "th1_doc_sketch_locked_route" if selected_doc else "th1_doc_sketch_fallback_route"

    candidate_pool, final_passages, veto, veto_reason = _content_route(
        question,
        graph,
        docs_for_passages,
        cross_top_k=cross_top_k,
        final_top_k=final_top_k,
    )

    return {
        "mode": mode,
        "route_case": route_case,
        "query_profile": query_profile,
        "topic_phrases": topic_phrases,
        "doc_sketch_hybrid_threshold": doc_sketch_threshold,
        "doc_sketch_ranked": doc_sketch_ranked,
        "top_docs": docs_for_passages,
        "top_docs_initial": top_docs,
        "candidate_pool": candidate_pool,
        "passages": final_passages,
        "rerank_veto": veto,
        "veto_reason": veto_reason,
        "insufficient_context": veto or not final_passages,
        "query_signals": query_signals,
    }
