from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.settings import settings
from src.rag.auto_filter import infer_query_profile, is_likely_document_year_filter
from src.rag.hybrid import bm25_score, extract_topic_phrases, regex_tokenize, tokenize_vi
from src.rag.openai_clients import get_llm
from src.rag.rerank_cross import cross_rerank
from src.rag.retrieval_pipeline_simple import apply_metadata_filter, dense_search_many, rrf_fusion

logger = logging.getLogger(__name__)
_SUMMARY_TOKEN_CACHE: Dict[str, List[str]] = {}
_SUMMARY_CANDIDATE_CACHE: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
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
    "doc_sketch",
}
TREE_EXCLUDED_ARTIFACTS = {"article_bundle", "section_bundle"}
ALLOWED_ANSWER_SCOPES = {"exact", "definition", "summary", "list", "comparison", "procedure"}
VALID_LAW_TYPES = {
    "bộ luật",
    "luật",
    "nghị định",
    "thông tư",
    "quyết định",
    "chỉ thị",
    "công điện",
    "nghị quyết",
    "văn bản hợp nhất",
    "unknown",
}
_CROSS_REF_RE = re.compile(
    r"(?:(?:diem|điểm)\s+([a-zđ])\s+)?(?:(?:khoan|khoản)\s+(\d+)\s+)?(?:dieu|điều)\s+(\d+)",
    re.IGNORECASE,
)


def _repair_mojibake(text: Any) -> str:
    clean = str(text or "")
    markers = ("Ã", "Ä", "Â", "áº", "á»", "�")
    for _ in range(2):
        if not any(marker in clean for marker in markers):
            break
        try:
            repaired = clean.encode("latin1").decode("utf-8")
        except Exception:
            break
        if repaired == clean:
            break
        clean = repaired
    return clean


def _norm_space(text: Any) -> str:
    return " ".join(_repair_mojibake(text).split()).strip()


def _norm_key(text: Any) -> str:
    return _norm_space(text).lower()


def _ascii_key(text: Any) -> str:
    normalized = _norm_key(text).replace("đ", "d").replace("Đ", "D")
    raw = unicodedata.normalize("NFD", normalized)
    raw = "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")
    raw = raw.replace("đ", "d").replace("Đ", "D").replace("Ä‘", "d").replace("Ä", "D")
    return raw


def _infer_answer_scope(question: str, query_profile: Optional[Dict[str, Any]] = None) -> str:
    query_profile = dict(query_profile or {})
    q = _ascii_key(question)
    wants_list = bool(query_profile.get("wants_list_answer"))
    if any(term in q for term in ("so sanh", "khac nhau", "giong nhau", "doi chieu")):
        return "comparison"
    if any(term in q for term in ("thu tuc", "trinh tu", "ho so", "thoi han", "dieu kien", "quy trinh", "cac buoc")):
        return "procedure"
    if wants_list or any(
        term in q
        for term in (
            "gom nhung gi",
            "bao gom",
            "liet ke",
            "danh sach",
            "cac noi dung",
            "noi dung chu yeu",
            "cac muc",
            "nhung muc",
            "cac chi tieu",
            "nhung chi tieu",
            "cac nhom",
            "nhung nhom",
            "cac loai",
            "nhung loai",
        )
    ):
        return "list"
    if any(term in q for term in ("la gi", "la ai", "duoc hieu la", "dinh nghia", "khai niem")):
        return "definition"
    if any(
        term in q
        for term in (
            "van ban nao",
            "dia phuong nao",
            "co hieu luc",
            "hieu luc tu",
            "khi nao",
            "ngay nao",
            "so may",
            "so bao nhieu",
            "phe duyet gi",
            "ban hanh gi",
            "ban hanh kem theo gi",
        )
    ):
        return "exact"
    if query_profile.get("heading_terms") or query_profile.get("doc_number") or query_profile.get("document_match"):
        return "summary"
    return "exact"


def _normalize_answer_scope(value: Any, question: str = "", query_profile: Optional[Dict[str, Any]] = None) -> str:
    scope = str(value or "").strip().lower()
    if scope in ALLOWED_ANSWER_SCOPES:
        return scope
    return _infer_answer_scope(question, query_profile)


def _truncate(text: Any, limit: int = 320) -> str:
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
    md = dict(node.get("metadata") or {})
    for key in ("node_id", "node_type", "text", "retrieval_text", "rerank_text"):
        if key in node and key not in md:
            md[key] = node.get(key)
    return md


def _node_type(node: Dict[str, Any]) -> str:
    md = _node_md(node)
    return str(node.get("node_type") or md.get("node_type") or "").strip().lower()


def _artifact_type(node: Dict[str, Any]) -> str:
    md = _node_md(node)
    return str(md.get("artifact_type") or "evidence").strip().lower()


def _is_tree_excluded_artifact(node: Dict[str, Any]) -> bool:
    return _artifact_type(node) in TREE_EXCLUDED_ARTIFACTS


def _node_text(node: Dict[str, Any]) -> str:
    return _norm_space(node.get("text") or _node_md(node).get("text") or "")


def _summary_candidate_id(item: Dict[str, Any]) -> str:
    return str(item.get("node_id") or item.get("id") or item.get("chunk_id") or "").strip()


def _summary_token_key(item: Dict[str, Any], text: str) -> str:
    item_id = _summary_candidate_id(item)
    if item_id:
        return item_id
    return hashlib.sha1(_norm_space(text).encode("utf-8")).hexdigest()


def _summary_tokens(item: Dict[str, Any], text: str) -> List[str]:
    key = _summary_token_key(item, text)
    if key not in _SUMMARY_TOKEN_CACHE:
        _SUMMARY_TOKEN_CACHE[key] = regex_tokenize(text)
    return _SUMMARY_TOKEN_CACHE[key]


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


def _normalize_reference_filter(key: str, value: Any) -> Any:
    clean = _norm_space(value)
    if clean in {"", "null", "None"}:
        return None
    if key == "law_type":
        return clean if clean.lower() in VALID_LAW_TYPES else None
    low = clean.lower()
    if key == "article":
        m = re.search(r"(\d+[a-z]?)", low, flags=re.IGNORECASE)
        return f"Điều {m.group(1)}" if m else clean
    if key == "clause":
        m = re.search(r"(\d+)", low)
        return f"Khoản {m.group(1)}" if m else clean
    if key == "point":
        m = re.search(r"([a-zđ])", low, flags=re.IGNORECASE)
        return f"Điểm {m.group(1).lower()}" if m else clean
    return value


def _metadata_matches(md: Dict[str, Any], key: str, expected: Any) -> bool:
    if expected in (None, ""):
        return True
    expected = _normalize_reference_filter(key, expected)
    actual = _metadata_value(md, key)
    if actual in (None, ""):
        return False
    if key == "year":
        return str(actual) == str(expected)
    return _ascii_key(actual) == _ascii_key(expected)


def _json_loads_from_llm(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def _merge_profile(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    filters = dict(out.get("filters") or {})
    for key in ("doc_number", "law_type", "year", "article", "clause", "point"):
        value = extra.get(key)
        if value not in (None, "", [], {}):
            value = _normalize_reference_filter(key, value)
            if value in (None, "", [], {}):
                continue
            filters[key] = value
            out[key] = value
    if filters:
        out["filters"] = filters
    for key in ("intent", "route", "needs_reasoning", "should_deepen", "confidence"):
        value = extra.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    expanded = extra.get("expanded_queries")
    if isinstance(expanded, list):
        out["expanded_queries"] = [_norm_space(x) for x in expanded if _norm_space(x)]
    return out


def _sanitize_unsafe_year_filter(question: str, query_profile: Dict[str, Any], filters: Dict[str, Any]) -> None:
    year = filters.get("year")
    if year in (None, "", [], {}):
        return
    if is_likely_document_year_filter(question, year):
        return

    filters.pop("year", None)
    query_profile.pop("year", None)
    removed = list(query_profile.get("removed_filters") or [])
    removed.append(
        {
            "key": "year",
            "value": year,
            "reason": "not_a_document_year_context",
        }
    )
    query_profile["removed_filters"] = removed


def plan_query_with_llm(question: str, *, enable_llm: bool = True) -> Dict[str, Any]:
    """Return a structured query plan, using LLM when available and regex as fallback."""
    base = infer_query_profile(question) or {}
    base.setdefault("question", _norm_space(question))
    base.setdefault("expanded_queries", [])

    if not enable_llm or _rule_profile_is_sufficient_for_planning(base):
        base.setdefault("planner_source", "rule")
        base.setdefault("confidence", 0.82)
        return base

    prompt = f"""
Extract a structured search plan for a Vietnamese legal RAG system.
Return only JSON with these keys:
- intent: one of direct_reference, document_focus, doc_summary, procedure, penalty, condition_list, version_change, compare, scenario_reasoning, factoid, unknown
- doc_number, law_type, year, article, clause, point: string or null
- expanded_queries: 2 to 5 short Vietnamese legal search queries
- needs_reasoning: boolean
- should_deepen: boolean
- confidence: number from 0 to 1

Question: {question}
""".strip()
    try:
        resp = get_llm().invoke(
            [
                SystemMessage(content="You are a precise query planner. Output valid JSON only."),
                HumanMessage(content=prompt),
            ]
        )
        parsed = _json_loads_from_llm(getattr(resp, "content", "") or "")
    except Exception as exc:
        logger.info("LLM query planner skipped: %s", exc)
        parsed = None

    if not parsed:
        filters = dict(base.get("filters") or {})
        _sanitize_unsafe_year_filter(question, base, filters)
        base["filters"] = filters
        return base
    merged = _merge_profile(base, parsed)
    merged["planner_source"] = "llm"
    filters = dict(merged.get("filters") or {})
    _sanitize_unsafe_year_filter(question, merged, filters)
    merged["filters"] = filters
    return merged


def _rule_profile_is_sufficient_for_planning(profile: Dict[str, Any]) -> bool:
    filters = dict(profile.get("filters") or {})
    route = str(profile.get("route") or "")
    has_structured_target = any(filters.get(key) not in (None, "", [], {}) for key in ("article", "clause", "point"))
    has_doc_filter = any(filters.get(key) not in (None, "", [], {}) for key in ("doc_number", "law_type", "year"))
    has_heading = bool(profile.get("heading_terms"))
    if has_structured_target:
        return True
    if has_doc_filter and route in {"direct_reference", "document_focus", "heading_list", "version_change"}:
        return True
    if has_heading and route in {"heading_list", "document_focus"}:
        return True
    if bool(profile.get("change_terms")) and has_doc_filter:
        return True
    return False


def _fallback_expanded_queries(question: str, query_profile: Dict[str, Any]) -> List[str]:
    out = [_norm_space(question)]
    filters = dict(query_profile.get("filters") or {})
    bits = [
        str(filters.get("doc_number") or ""),
        str(filters.get("law_type") or ""),
        str(filters.get("article") or ""),
        str(filters.get("clause") or ""),
        str(filters.get("point") or ""),
    ]
    joined = _norm_space(" ".join(x for x in bits if x))
    if joined:
        out.append(joined)
    if not bool(query_profile.get("explicit_reference")):
        for phrase in extract_topic_phrases(question)[:3]:
            out.append(phrase)
    for item in list(query_profile.get("expanded_queries") or []):
        if _norm_space(item):
            out.append(_norm_space(item))
    seen = set()
    uniq: List[str] = []
    for item in out:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    max_queries = 3 if bool(query_profile.get("explicit_reference")) else 5
    return uniq[:max_queries]


def _summary_text_for_node(node: Dict[str, Any]) -> str:
    md = _node_md(node)
    title = _norm_space(md.get("official_title") or md.get("law_name") or md.get("file_stem") or "")
    path = _norm_space(md.get("path_title") or md.get("title") or md.get("heading_title") or "")
    heading = _norm_space(md.get("heading_title") or md.get("title") or "")
    doc_number = _norm_space(md.get("doc_number") or "")
    doc_type = _norm_space(md.get("doc_type") or md.get("law_type") or "")
    agency = _norm_space(md.get("issuing_agency") or md.get("source") or "")
    node_type = _node_type(node) or _norm_space(md.get("node_type") or "")
    artifact = _artifact_type(node)
    text = _node_text(node) or _norm_space(node.get("retrieval_text") or "")
    child_count = len([x for x in (md.get("children_ids") or []) if str(x).strip()])

    parts = [
        f"Document: {title}" if title else "",
        f"Doc number: {doc_number}" if doc_number else "",
        f"Doc type: {doc_type}" if doc_type else "",
        f"Agency/source: {agency}" if agency else "",
        f"Path: {path}" if path else "",
        f"Heading: {heading}" if heading and heading != path else "",
        f"Node type: {node_type}" if node_type else "",
        f"Artifact: {artifact}" if artifact else "",
        f"Children: {child_count}" if child_count else "Children: 0",
        f"Short summary: {_truncate(text, 260)}" if text else "",
    ]
    return "\n".join(p for p in parts if p).strip()


def _summary_from_node(node: Dict[str, Any], *, reason: str = "", score: float = 0.0) -> Dict[str, Any]:
    md = _node_md(node)
    node_id = str(node.get("node_id") or md.get("node_id") or md.get("chunk_id") or "").strip()
    summary = _summary_text_for_node(node)
    return {
        "id": node_id,
        "node_id": node_id,
        "chunk_id": str(md.get("chunk_id") or node_id),
        "doc_id": str(md.get("doc_id") or ""),
        "doc_key": str(md.get("doc_id") or ""),
        "metadata": md,
        "summary_text": summary,
        "text": summary,
        "snippet": _truncate(summary, 700),
        "retrieval_text": summary,
        "rerank_text_short": summary[:1500],
        "node_type": _node_type(node),
        "artifact_type": _artifact_type(node),
        "summary_score": float(score or 0.0),
        "reason": reason,
    }


def _node_heading_text(node: Dict[str, Any]) -> str:
    md = _node_md(node)
    parts = [
        _norm_space(md.get("official_title") or md.get("law_name") or ""),
        _norm_space(md.get("path_title") or md.get("title") or md.get("heading_title") or ""),
    ]
    text = _norm_space(_node_text(node))
    if text and text not in parts:
        parts.append(text)
    seen = set()
    out = []
    for part in parts:
        key = _ascii_key(part)
        if part and key not in seen:
            seen.add(key)
            out.append(part)
    return "\n".join(out).strip()


def _node_preview(node: Dict[str, Any], *, limit: int = 220) -> Dict[str, Any]:
    md = _node_md(node)
    return {
        "node_id": str(node.get("node_id") or md.get("node_id") or ""),
        "node_type": _node_type(node),
        "title": _truncate(md.get("heading_title") or md.get("title") or md.get("path_title") or _node_text(node), limit),
        "path_title": _truncate(md.get("path_title") or md.get("title") or md.get("heading_title") or "", limit),
        "summary": _truncate(_node_text(node), limit),
        "children_count": len([x for x in (md.get("children_ids") or []) if str(x).strip()]),
    }


def _related_node_previews(
    graph: Dict[str, Any],
    node: Dict[str, Any],
    relation: str,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    md = _node_md(node)
    max_items = max(1, int(limit or getattr(settings, "tree_neighbor_preview_limit", 5) or 5))
    nodes: List[Dict[str, Any]] = []
    if relation == "children":
        for child_id in [str(x) for x in (md.get("children_ids") or []) if str(x).strip()]:
            child = idx.get(child_id)
            if child and not _is_tree_excluded_artifact(child):
                nodes.append(child)
    elif relation == "siblings":
        parent = idx.get(str(md.get("parent_id") or ""))
        sibling_ids = []
        if parent:
            sibling_ids = [str(x) for x in (_node_md(parent).get("children_ids") or []) if str(x).strip()]
        else:
            sibling_ids = [str(node.get("node_id") or "")]
            sibling_ids.extend([str(x) for x in (md.get("sibling_ids") or []) if str(x).strip()])
        for sibling_id in sibling_ids:
            sibling = idx.get(sibling_id)
            if sibling and not _is_tree_excluded_artifact(sibling):
                nodes.append(sibling)
    elif relation == "parent":
        parent = idx.get(str(md.get("parent_id") or ""))
        if parent and not _is_tree_excluded_artifact(parent):
            nodes.append(parent)
    return [_node_preview(node) for node in _sort_nodes_by_order(nodes)[:max_items]]


def _enriched_candidate_for_judge(graph: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(item.get("metadata") or {})
    node_id = str(item.get("node_id") or md.get("node_id") or "")
    node = _node_index(graph).get(node_id)
    children_ids = [str(x) for x in (md.get("children_ids") or []) if str(x).strip()]
    parent_id = str(md.get("parent_id") or "").strip()
    sibling_count = 0
    if parent_id and parent_id in _node_index(graph):
        sibling_count = len([x for x in (_node_md(_node_index(graph)[parent_id]).get("children_ids") or []) if str(x).strip()])
    elif md.get("sibling_ids"):
        sibling_count = len([x for x in (md.get("sibling_ids") or []) if str(x).strip()]) + 1
    return {
        "node_id": node_id,
        "node_type": str(item.get("node_type") or md.get("node_type") or ""),
        "artifact_type": str(item.get("artifact_type") or md.get("artifact_type") or ""),
        "doc_number": str(md.get("doc_number") or ""),
        "official_title": _truncate(md.get("official_title") or md.get("law_name") or "", 220),
        "path_title": _truncate(md.get("path_title") or md.get("title") or md.get("heading_title") or "", 280),
        "own_text": _truncate(md.get("text") or item.get("summary_text") or item.get("text") or "", 340),
        "children_count": len(children_ids),
        "sibling_count": sibling_count,
        "parent_preview": _related_node_previews(graph, node, "parent", limit=1) if node else [],
        "children_preview": _related_node_previews(graph, node, "children") if node else [],
        "siblings_preview": _related_node_previews(graph, node, "siblings") if node else [],
        "answer_source_options": ["own_text", "title_or_heading", "parent_heading", "children", "siblings", "parent"],
        "answer_scope_options": sorted(ALLOWED_ANSWER_SCOPES),
        "score": float(item.get("cross_score", item.get("rrf_score", item.get("summary_score", 0.0))) or 0.0),
    }


def _iter_summary_candidates(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    cache_key = (id(graph), len(list(graph.get("nodes") or [])))
    cached = _SUMMARY_CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return [dict(item) for item in cached]

    out: List[Dict[str, Any]] = []
    allowed_artifacts = {"evidence", "doc_sketch"}
    for node in _node_index(graph).values():
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        artifact = _artifact_type(node)
        node_type = _node_type(node)
        if artifact in TREE_EXCLUDED_ARTIFACTS:
            continue
        if artifact not in allowed_artifacts and node_type not in REFERENCE_NODE_TYPES_EXTENDED:
            continue
        if not _summary_text_for_node(node):
            continue
        out.append(_summary_from_node(node, reason="graph_summary"))
    _SUMMARY_CANDIDATE_CACHE.clear()
    _SUMMARY_CANDIDATE_CACHE[cache_key] = [dict(item) for item in out]
    return out


DOC_TITLE_STOPWORDS = {
    "trong",
    "cua",
    "cho",
    "voi",
    "theo",
    "nay",
    "cac",
    "nhung",
    "den",
    "nam",
    "tam",
    "nhin",
    "quyet",
    "dinh",
    "phe",
    "duyet",
    "ve",
    "va",
    "la",
    "gi",
    "noi",
    "dung",
    "muc",
    "tai",
    "tren",
    "duoc",
    "van",
    "ban",
    "quy",
    "hoach",
    "chung",
}


def _important_doc_tokens(text: Any) -> List[str]:
    tokens = regex_tokenize(_ascii_key(text))
    return [tok for tok in tokens if len(tok) >= 3 and tok not in DOC_TITLE_STOPWORDS]


def _infer_document_filter_from_question(graph: Dict[str, Any], question: str) -> Dict[str, Any]:
    q_key = _ascii_key(question)
    q_tokens = set(_important_doc_tokens(question))
    if not q_tokens and not q_key:
        return {}

    docs: Dict[str, Dict[str, Any]] = {}
    for node in _node_index(graph).values():
        md = _node_md(node)
        doc_number = _norm_space(md.get("doc_number") or "")
        doc_id = _norm_space(md.get("doc_id") or "")
        key = _ascii_key(doc_number or doc_id)
        if not key or key in docs:
            continue
        title = _norm_space(md.get("official_title") or md.get("law_name") or md.get("file_stem") or "")
        if not title and not doc_number:
            continue
        docs[key] = {"doc_number": doc_number, "doc_id": doc_id, "title": title, "year": md.get("year")}

    best: Dict[str, Any] = {}
    best_score = 0.0
    for doc in docs.values():
        doc_number = _norm_space(doc.get("doc_number") or "")
        doc_number_key = _ascii_key(doc_number)
        if doc_number_key and doc_number_key in q_key:
            return {
                "doc_number": doc_number,
                "matched_by": "doc_number_in_question",
                "score": 1.0,
                "title": doc.get("title") or "",
                "year": doc.get("year"),
            }

        title_tokens = _important_doc_tokens(" ".join(str(doc.get(k) or "") for k in ("title", "doc_number", "doc_id")))
        if not title_tokens:
            continue
        hits = [tok for tok in title_tokens if tok in q_tokens]
        hit_count = len(set(hits))
        score = hit_count / max(1, min(len(set(title_tokens)), len(q_tokens)))
        if hit_count >= 5 and score > best_score:
            best_score = score
            best = {
                "doc_number": doc_number,
                "matched_by": "title_overlap_in_question",
                "score": round(score, 4),
                "hit_count": hit_count,
                "title": doc.get("title") or "",
                "year": doc.get("year"),
            }

    if best and best_score >= 0.45 and best.get("doc_number"):
        return best
    return {}


def _target_node_type(filters: Dict[str, Any]) -> str:
    if filters.get("point"):
        return "point"
    if filters.get("clause"):
        return "clause"
    if filters.get("article"):
        return "article"
    return ""


def _find_direct_target_summaries(graph: Dict[str, Any], query_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    filters = dict(query_profile.get("filters") or {})
    target_type = _target_node_type(filters)
    if not target_type and not filters.get("doc_number"):
        return []

    matches: List[Dict[str, Any]] = []
    if target_type:
        for node in _node_index(graph).values():
            md = _node_md(node)
            if _node_type(node) != target_type:
                continue
            ok = True
            for key in ("doc_number", "law_type", "year", "article", "clause", "point"):
                if filters.get(key) not in (None, "") and not _metadata_matches(md, key, filters.get(key)):
                    ok = False
                    break
            if ok:
                matches.append(_summary_from_node(node, reason="direct_filter_match", score=1.0))

        if matches:
            has_doc_discriminator = any(filters.get(key) not in (None, "", [], {}) for key in ("doc_number", "law_type", "year"))
            if len(matches) > 1 and not has_doc_discriminator:
                return []
            matches.sort(key=lambda x: int((x.get("metadata") or {}).get("order_index") or 0))
            return matches[: max(1, int(getattr(settings, "tree_direct_top_k", 8) or 8))]

    if filters.get("doc_number") and not target_type:
        for node in _node_index(graph).values():
            md = _node_md(node)
            if not _metadata_matches(md, "doc_number", filters.get("doc_number")):
                continue
            if _is_tree_excluded_artifact(node):
                continue
            if _artifact_type(node) == "doc_sketch" or _node_type(node) in {"article", "section", "appendix", "attachment", "roman_section", "alpha_section"}:
                matches.append(_summary_from_node(node, reason="document_filter_match", score=0.8))
        return matches[: max(1, int(getattr(settings, "tree_summary_top_k", 12) or 12))]

    return []


def _iter_nodes_for_heading_search(graph: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    nodes = list(_node_index(graph).values())
    filters = dict(filters or {})
    keys = [key for key in ("doc_number", "law_type", "year") if filters.get(key) not in (None, "", [], {})]
    if not keys:
        return nodes
    return [node for node in nodes if all(_metadata_matches(_node_md(node), key, filters.get(key)) for key in keys)]


def _find_heading_match_summaries(
    graph: Dict[str, Any],
    question: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    q_key = _ascii_key(question)
    q_tokens = set(_important_doc_tokens(question))
    q_years = set(re.findall(r"\b(?:19|20)\d{2}\b", q_key))
    q_phrases = [phrase for phrase in extract_topic_phrases(question) if len(phrase) >= 8]
    requested_heading_phrases = [phrase for phrase in ("tren the gioi", "trong nuoc") if phrase in q_key]
    if len(q_tokens) < 3:
        return []

    def requested_phrase_is_heading(md: Dict[str, Any], node: Dict[str, Any], phrase: str) -> bool:
        path_key = _ascii_key(md.get("path_title") or "")
        heading_key = _ascii_key(md.get("heading_title") or md.get("title") or "")
        opening_key = _ascii_key(_node_text(node)[:90])
        heading_surface = _norm_space(f"{path_key} {heading_key[:90]} {opening_key}")
        pattern = rf"(?:^|>\s*|\b\d+\.\s*|\b[ivxlcdm]+\.\s*){re.escape(phrase)}\b"
        return bool(re.search(pattern, heading_surface, flags=re.IGNORECASE))

    ranked: List[Dict[str, Any]] = []
    for node in _iter_nodes_for_heading_search(graph, filters):
        if _is_tree_excluded_artifact(node):
            continue
        md = _node_md(node)
        heading_bits = " ".join(
            str(md.get(k) or "") for k in ("path_title", "heading_title", "title")
        )
        own_heading_bits = " ".join(str(md.get(k) or "") for k in ("heading_title", "title"))
        heading_key = _ascii_key(heading_bits)
        if not heading_key:
            continue
        heading_tokens = set(_important_doc_tokens(heading_bits))
        if not heading_tokens:
            continue
        hits = q_tokens & heading_tokens
        hit_count = len(hits)
        overlap = hit_count / max(1, min(len(q_tokens), len(heading_tokens)))
        contains_bonus = 0.0
        heading_subheading_match = False
        heading_own_match = False
        if q_key and (q_key in heading_key or heading_key in q_key):
            contains_bonus = 1.0
        phrase_hits = [phrase for phrase in q_phrases if _ascii_key(phrase) in heading_key]
        contains_bonus += 0.18 * len(phrase_hits)
        own_heading_key = _ascii_key(own_heading_bits)
        own_phrase_hits = [phrase for phrase in q_phrases if _ascii_key(phrase) in own_heading_key]
        own_heading_tokens = set(_important_doc_tokens(own_heading_bits))
        own_hits = q_tokens & own_heading_tokens
        own_overlap = len(own_hits) / max(1, min(len(q_tokens), len(own_heading_tokens)))
        if own_phrase_hits or (len(own_hits) >= 2 and own_overlap >= 0.45):
            heading_own_match = True
            contains_bonus += 0.7
        for requested_phrase in requested_heading_phrases:
            if requested_phrase_is_heading(md, node, requested_phrase):
                heading_subheading_match = True
                contains_bonus += 0.8
            elif requested_phrase in _ascii_key(_node_text(node)):
                contains_bonus += 0.25
            if requested_phrase in own_heading_key:
                heading_own_match = True
                contains_bonus += 0.4
        if hit_count < 3 and contains_bonus <= 0:
            continue

        heading_years = set(re.findall(r"\b(?:19|20)\d{2}\b", heading_key))
        year_penalty = 0.0
        if q_years:
            if not q_years <= heading_years:
                year_penalty += 0.35
            extra_years = heading_years - q_years
            if extra_years:
                year_penalty += min(0.35, 0.18 * len(extra_years))
        child_penalty = 0.08 if [x for x in (md.get("children_ids") or []) if str(x).strip()] else 0.0
        depth_bonus = min(float(md.get("level") or 0) * 0.03, 0.15)
        score = overlap + contains_bonus + depth_bonus - child_penalty - year_penalty
        if score < 0.55:
            continue
        item = _summary_from_node(node, reason="heading_exact_match", score=score)
        item["heading_match_score"] = score
        item["heading_subheading_match"] = heading_subheading_match
        item["heading_own_match"] = heading_own_match
        ranked.append(item)

    if requested_heading_phrases and any(bool(item.get("heading_subheading_match")) for item in ranked):
        ranked = [item for item in ranked if bool(item.get("heading_subheading_match"))]

    ranked.sort(
        key=lambda item: (
            bool(item.get("heading_own_match")),
            float(item.get("heading_match_score") or 0.0),
            int((item.get("metadata") or {}).get("level") or 0),
            -len([x for x in ((item.get("metadata") or {}).get("children_ids") or []) if str(x).strip()]),
        ),
        reverse=True,
    )
    return _dedupe_summaries(ranked)[: max(1, top_k)]


def _filter_summaries_by_metadata(items: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not filters:
        return items
    keys = [key for key in ("doc_number", "law_type", "year") if filters.get(key) not in (None, "", [], {})]
    if not keys:
        return items
    out: List[Dict[str, Any]] = []
    for item in items:
        md = dict(item.get("metadata") or {})
        if all(_metadata_matches(md, key, filters.get(key)) for key in keys):
            out.append(item)
    return out


def _promote_heading_matches_to_ancestors(
    graph: Dict[str, Any],
    items: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    terms = [_ascii_key(term) for term in (query_profile.get("heading_terms") or []) if _ascii_key(term)]
    if not terms:
        return items
    idx = _node_index(graph)
    promoted: List[Dict[str, Any]] = []
    for item in items:
        node_id = _summary_candidate_id(item)
        node = idx.get(node_id)
        parent_id = str((_node_md(node or {}).get("parent_id") if node else "") or "")
        replacement: Optional[Dict[str, Any]] = None
        while parent_id:
            parent = idx.get(parent_id)
            if not parent:
                break
            md = _node_md(parent)
            own_heading_key = _ascii_key(
                " ".join(str(md.get(k) or "") for k in ("heading_title", "title")) + " " + _node_text(parent)[:160]
            )
            if any(term in own_heading_key for term in terms):
                replacement = _summary_from_node(
                    parent,
                    reason="heading_parent_match",
                    score=float(item.get("heading_match_score") or item.get("score") or 0.0) + 0.4,
                )
                replacement["promoted_from_node_id"] = node_id
                break
            parent_id = str(md.get("parent_id") or "")
        promoted.append(replacement or item)
    return _dedupe_summaries(promoted)


def _bm25_rank_summaries(question: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    query_tokens = regex_tokenize(question)
    topic_phrases = extract_topic_phrases(question)
    q_key = _ascii_key(question)
    wants_table = any(term in q_key for term in ("bang", "bieu", "phu luc", "danh muc", "danh sach", "table"))
    ranked: List[Dict[str, Any]] = []
    for item in candidates:
        scored = dict(item)
        text = str(item.get("summary_text") or item.get("retrieval_text") or item.get("text") or "")
        md = dict(item.get("metadata") or {})
        node_type = str(md.get("node_type") or item.get("node_type") or "").strip().lower()
        path_bits = " ".join(
            str(md.get(k) or "") for k in ("path_title", "heading_title", "title", "official_title", "doc_number")
        )
        hay = _norm_space(f"{path_bits} {text}").lower()
        score = bm25_score(query_tokens, _summary_tokens(item, text))
        score += 4.0 * sum(1 for phrase in topic_phrases if phrase and phrase in hay)
        if query_tokens and all(tok in hay for tok in query_tokens[:4]):
            score += 2.0
        if not wants_table and node_type in {"table", "table_row"}:
            score -= 10.0
        scored["bm25_score"] = score
        ranked.append(scored)
    ranked.sort(key=lambda x: float(x.get("bm25_score", 0.0)), reverse=True)
    for rank, item in enumerate(ranked[:top_k], start=1):
        item["rank"] = rank
    return ranked[:top_k]


def _map_dense_summary_results(dense: List[Dict[str, Any]], candidates_by_id: Dict[str, Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for rank, item in enumerate(dense, start=1):
        md = dict(item.get("metadata") or {})
        keys = [
            str(item.get("node_id") or ""),
            str(item.get("chunk_id") or ""),
            str(md.get("node_id") or ""),
            str(md.get("chunk_id") or ""),
        ]
        summary = next((candidates_by_id[k] for k in keys if k in candidates_by_id), None)
        if not summary:
            continue
        node_id = _summary_candidate_id(summary)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        scored = dict(summary)
        scored["dense_score"] = float(item.get("dense_score", 0.0) or 0.0)
        scored["rank"] = rank
        out.append(scored)
    return out[:top_k]


def _dense_rank_summaries_batch(
    questions: List[str],
    candidates_by_id: Dict[str, Dict[str, Any]],
    query_profile: Dict[str, Any],
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if not bool(getattr(settings, "enable_dense_summary_search", True)):
        return {question: [] for question in questions}
    try:
        dense_by_query = dense_search_many(questions, top_k=top_k, query_profile=query_profile)
    except Exception as exc:
        logger.info("Dense summary retrieval skipped: %s", exc)
        return {question: [] for question in questions}
    return {
        question: _map_dense_summary_results(dense_by_query.get(_norm_space(question), []), candidates_by_id, top_k)
        for question in questions
    }


def retrieve_summary_candidates(
    question: str,
    graph: Dict[str, Any],
    query_profile: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    timings_ms: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    timings_ms = timings_ms if timings_ms is not None else {}

    def mark(name: str, started: float) -> float:
        now = time.perf_counter()
        timings_ms[name] = timings_ms.get(name, 0.0) + round((now - started) * 1000, 2)
        return now

    t_stage = time.perf_counter()
    top_k = int(top_k or getattr(settings, "tree_summary_top_k", 12) or 12)
    all_candidates = _iter_summary_candidates(graph)
    t_stage = mark("summary_iter_candidates", t_stage)
    if not all_candidates:
        return []

    filtered_candidates, filter_info = apply_metadata_filter(all_candidates, query_profile)
    t_stage = mark("summary_metadata_filter", t_stage)
    working = filtered_candidates or all_candidates
    candidates_by_id = {_summary_candidate_id(x): x for x in all_candidates if _summary_candidate_id(x)}
    t_stage = mark("summary_index_candidates", t_stage)

    rank_lists: List[List[Dict[str, Any]]] = []
    expanded_queries = _fallback_expanded_queries(question, query_profile)
    dense_by_query = _dense_rank_summaries_batch(expanded_queries, candidates_by_id, query_profile, top_k=max(top_k * 4, 30))
    t_stage = mark("summary_dense_rank", t_stage)
    for expanded_query in expanded_queries:
        dense_ranked = dense_by_query.get(_norm_space(expanded_query), [])
        bm25_ranked = _bm25_rank_summaries(expanded_query, working, top_k=max(top_k * 4, 30))
        t_stage = mark("summary_bm25_rank", t_stage)
        if dense_ranked:
            rank_lists.append(dense_ranked)
        if bm25_ranked:
            rank_lists.append(bm25_ranked)

    if not rank_lists:
        return working[:top_k]

    fused = rrf_fusion(rank_lists, rrf_k=int(getattr(settings, "rrf_k", 60) or 60), top_k=max(top_k * 2, 30))
    t_stage = mark("summary_rrf_fusion", t_stage)
    summaries: List[Dict[str, Any]] = []
    for item in fused:
        node_id = _summary_candidate_id(item)
        base = dict(candidates_by_id.get(node_id) or item)
        base.update({k: v for k, v in item.items() if k.endswith("_rank") or k in {"rrf_score", "dense_score", "bm25_score", "rank"}})
        base["filter_info"] = filter_info
        summaries.append(base)

    reranked = cross_rerank(question, summaries, top_n=min(len(summaries), max(top_k, int(getattr(settings, "rerank_top_n", top_k) or top_k))))
    mark("summary_cross_rerank", t_stage)
    return reranked[:top_k]


def _llm_navigation_decision(
    question: str,
    query_profile: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    *,
    hop: int,
    enable_llm: bool,
    graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not enable_llm:
        return _fallback_navigation_decision(question, query_profile, summaries, hop=hop)

    compact = []
    graph = graph or {"nodes": [], "edges": [], "node_index": {}}
    for item in summaries[: max(1, int(getattr(settings, "tree_judge_top_k", 6) or 6))]:
        compact.append(_enriched_candidate_for_judge(graph, item))

    prompt = f"""
You are navigating a tree of Vietnamese legal documents.
The candidates below are short summaries only, not full evidence.
Each candidate includes short previews of its parent, direct children, and siblings.
Choose the next action. Do not ask for parent/children/siblings unless it is necessary.
If the selected candidate is an article/section/appendix with children and the question asks what that unit provides, prefer deepen_node so child summaries can be judged before fetching evidence.
Never choose deepen_node for a selected candidate whose children_count is 0.
If a selected leaf candidate is one item/clause/point inside a list and the question asks about the whole article/section/list, choose fetch_siblings.
If the answer is in a title, document title, path title, or ancestor heading rather than the node's own content, keep decision="answer_now" and set answer_source to title_or_heading or parent_heading.
Classify answer_scope before choosing answer_source:
- exact: one short factual answer, such as which document, which place, effective date, article/clause content, approval title.
- definition: a legal term definition or "X là gì/là ai".
- summary: a compact summary of the selected legal unit.
- list: the user explicitly asks for all items, contents, groups, indicators, objectives, tasks, or a complete list.
- comparison: compare two or more legal units or entities.
- procedure: steps, procedure, dossier, deadline, conditions, responsibilities in order.
For exact or definition, do not use siblings/children as the main answer unless the exact answer is inside that relation. Prefer one primary node or title/heading.
For list, if the selected node is a parent heading with direct children, choose deepen_node; if selected nodes are items under the same parent, choose fetch_siblings when the full list is needed.
Mark primary_node_ids as the evidence that directly answers the question. Mark supporting_node_ids only for context checks.
If confidence is below 0.6, prefer inspecting the graph relation (children, siblings, or parent) over answer_now or refuse_or_clarify.
Use answerability="insufficient" only when the candidates are unrelated or no useful legal evidence exists.
If candidates contain relevant evidence but may be incomplete, use answerability="partial" and choose the next relation to inspect.

Allowed decisions:
- answer_now: selected summaries are enough to fetch minimal evidence and answer.
- deepen_node: inspect child summaries of selected nodes.
- fetch_parent_summary: inspect parent summaries.
- fetch_siblings: inspect sibling summaries.
- follow_cross_ref: inspect referenced legal nodes.
- rewrite_query: search again with next_query.
- refuse_or_clarify: context is not enough or the question is ambiguous.

Allowed answer_source values:
- own_text: answer from selected node's own text.
- title_or_heading: answer from official_title, path_title, or heading of the selected node.
- parent_heading: answer from the selected node's parent/ancestor heading.
- children: fetch direct children as evidence.
- siblings: fetch direct siblings as evidence.
- parent: fetch parent node content as evidence.

Allowed answer_scope values: exact, definition, summary, list, comparison, procedure.

Return JSON only:
{{
  "decision": "...",
  "selected_node_ids": ["..."],
  "primary_node_ids": ["..."],
  "supporting_node_ids": ["..."],
  "answer_scope": "exact|definition|summary|list|comparison|procedure",
  "answer_source": "own_text|title_or_heading|parent_heading|children|siblings|parent",
  "reason": "...",
  "missing_evidence": ["..."],
  "next_query": "...",
  "answerability": "sufficient|partial|insufficient",
  "needs_reasoning": true,
  "confidence": 0.0
}}

Question: {question}
Query profile: {json.dumps(query_profile, ensure_ascii=False)}
Hop: {hop}
Candidates: {json.dumps(compact, ensure_ascii=False)}
""".strip()
    try:
        resp = get_llm().invoke(
            [
                SystemMessage(content="You are a conservative legal RAG context judge. Output valid JSON only."),
                HumanMessage(content=prompt),
            ]
        )
        parsed = _json_loads_from_llm(getattr(resp, "content", "") or "")
    except Exception as exc:
        logger.info("LLM tree navigation skipped: %s", exc)
        parsed = None

    if not parsed:
        return _fallback_navigation_decision(question, query_profile, summaries, hop=hop)

    decision = str(parsed.get("decision") or "answer_now").strip()
    allowed = {"answer_now", "deepen_node", "fetch_parent_summary", "fetch_siblings", "follow_cross_ref", "rewrite_query", "refuse_or_clarify"}
    if decision not in allowed:
        decision = "answer_now"
    answer_source = str(parsed.get("answer_source") or "own_text").strip()
    allowed_sources = {"own_text", "title_or_heading", "parent_heading", "children", "siblings", "parent"}
    if answer_source not in allowed_sources:
        answer_source = "own_text"
    answer_scope = _normalize_answer_scope(parsed.get("answer_scope"), question, query_profile)
    selected = [str(x) for x in (parsed.get("selected_node_ids") or []) if str(x).strip()]
    if not selected and summaries:
        selected = [str(summaries[0].get("node_id") or "")]
    primary = [str(x) for x in (parsed.get("primary_node_ids") or []) if str(x).strip()]
    supporting = [str(x) for x in (parsed.get("supporting_node_ids") or []) if str(x).strip()]
    if not primary:
        primary = selected
    parsed["decision"] = decision
    parsed["answer_scope"] = answer_scope
    parsed["answer_source"] = answer_source
    parsed["selected_node_ids"] = selected[: max(1, int(getattr(settings, "tree_evidence_top_k", 4) or 4))]
    parsed["primary_node_ids"] = primary[: max(1, int(getattr(settings, "tree_evidence_top_k", 4) or 4))]
    parsed["supporting_node_ids"] = supporting[: max(0, int(getattr(settings, "tree_evidence_top_k", 4) or 4))]
    grounded_doc = _norm_space((query_profile.get("filters") or {}).get("doc_number") or query_profile.get("doc_number") or "")
    doc_match = dict(query_profile.get("document_match") or {})
    doc_match_score = float(doc_match.get("score") or 0.0)
    if hop == 0 and decision == "refuse_or_clarify" and summaries and (grounded_doc or doc_match_score >= 0.45):
        guarded = _fallback_navigation_decision(question, query_profile, summaries, hop=hop)
        guarded["reason"] = _norm_space(
            "guarded_strong_document_match; llm_refusal_reason: "
            + str(parsed.get("reason") or "refuse_or_clarify")
        )
        guarded["llm_rejected_decision"] = parsed
        return guarded
    return parsed


def _own_text_relevance(question: str, item: Dict[str, Any]) -> int:
    md = dict(item.get("metadata") or {})
    path_leaf = str(md.get("path_title") or md.get("title") or md.get("heading_title") or "").split(">")[-1]
    own = _ascii_key(" ".join(str(x or "") for x in (md.get("text"), md.get("heading_title"), path_leaf)))
    stop = {
        "trong",
        "cua",
        "cho",
        "voi",
        "theo",
        "nay",
        "cac",
        "nhung",
        "den",
        "nam",
        "tam",
        "nhin",
        "quyet",
        "dinh",
        "phe",
        "duyet",
        "noi",
        "dung",
        "dieu",
        "khoan",
        "diem",
        "gi",
    }
    tokens = [tok for tok in regex_tokenize(_ascii_key(question)) if len(tok) >= 3 and tok not in stop]
    return sum(1 for tok in set(tokens) if tok in own)


def _fallback_navigation_decision(question: str, query_profile: Dict[str, Any], summaries: List[Dict[str, Any]], *, hop: int) -> Dict[str, Any]:
    selected = [str(item.get("node_id") or "") for item in summaries[: max(1, int(getattr(settings, "tree_evidence_top_k", 4) or 4))]]
    selected = [x for x in selected if x]
    filters = dict(query_profile.get("filters") or {})
    answer_scope = _infer_answer_scope(question, query_profile)
    if not summaries:
        decision = "refuse_or_clarify"
        answerability = "insufficient"
    else:
        first = summaries[0]
        has_children = bool((first.get("metadata") or {}).get("children_ids"))
        if str(first.get("reason") or "") == "heading_exact_match" and not has_children:
            return {
                "decision": "answer_now",
                "selected_node_ids": [str(first.get("node_id") or "")],
                "primary_node_ids": [str(first.get("node_id") or "")],
                "supporting_node_ids": [],
                "answer_scope": answer_scope,
                "answer_source": "own_text",
                "reason": "fallback_heading_exact_match",
                "missing_evidence": [],
                "next_query": "",
                "answerability": "sufficient",
                "needs_reasoning": bool(query_profile.get("needs_reasoning") or query_profile.get("route") == "condition_circumstance"),
                "confidence": 0.75,
            }
        asks_article_scope = bool(filters.get("article") and not (filters.get("clause") or filters.get("point")))
        broad_scope = not (filters.get("article") or filters.get("clause") or filters.get("point"))
        decision = "deepen_node" if has_children and (asks_article_scope or broad_scope) else "answer_now"
        answerability = "partial" if decision == "deepen_node" else "sufficient"
        if decision == "deepen_node":
            first_type = str(first.get("node_type") or (first.get("metadata") or {}).get("node_type") or "").lower()
            if first_type in {"point", "decimal_item", "list_item", "bullet"}:
                relevant = [
                    item
                    for item in summaries[: max(1, int(getattr(settings, "tree_judge_top_k", 6) or 6))]
                    if (item.get("metadata") or {}).get("children_ids")
                ]
                relevant.sort(key=lambda item: int((item.get("metadata") or {}).get("order_index") or 0))
                selected = [str(item.get("node_id") or "") for item in relevant[: max(1, int(getattr(settings, "tree_evidence_top_k", 4) or 4))]]
            else:
                selected = []
            if not selected:
                selected = [str(first.get("node_id") or "")] if str(first.get("node_id") or "").strip() else selected[:1]
    return {
        "decision": decision,
        "selected_node_ids": selected,
        "primary_node_ids": selected,
        "supporting_node_ids": [],
        "answer_scope": answer_scope,
        "answer_source": "own_text",
        "reason": "fallback_navigation",
        "missing_evidence": [],
        "next_query": "",
        "answerability": answerability,
        "needs_reasoning": bool(query_profile.get("needs_reasoning") or query_profile.get("route") == "condition_circumstance"),
        "confidence": 0.5 if selected else 0.0,
    }


def _selected_nodes(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    out = []
    seen = set()
    for node_id in selected_node_ids:
        node_id = str(node_id or "").strip()
        if not node_id or node_id in seen or node_id not in idx:
            continue
        seen.add(node_id)
        out.append(idx[node_id])
    return out


def _find_child_bearing_counterpart(graph: Dict[str, Any], node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    md = _node_md(node)
    doc_number = _ascii_key(md.get("doc_number") or "")
    doc_id = _ascii_key(md.get("doc_id") or "")
    path = _ascii_key(md.get("path_title") or md.get("title") or md.get("heading_title") or "")
    if not path:
        return None

    selected_id = str(node.get("node_id") or "")
    for candidate in _node_index(graph).values():
        candidate_id = str(candidate.get("node_id") or "")
        if candidate_id == selected_id:
            continue
        cmd = _node_md(candidate)
        children = [str(x) for x in (cmd.get("children_ids") or []) if str(x).strip()]
        if not children:
            continue
        if doc_number and _ascii_key(cmd.get("doc_number") or "") != doc_number:
            continue
        if not doc_number and doc_id and _ascii_key(cmd.get("doc_id") or "") != doc_id:
            continue
        candidate_path = _ascii_key(cmd.get("path_title") or cmd.get("title") or cmd.get("heading_title") or "")
        if candidate_path == path:
            return candidate
    return None


def _expand_children(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    for node in _selected_nodes(graph, selected_node_ids):
        md = _node_md(node)
        if not [x for x in (md.get("children_ids") or []) if str(x).strip()]:
            counterpart = _find_child_bearing_counterpart(graph, node)
            if counterpart:
                node = counterpart
                md = _node_md(node)
        for child_id in [str(x) for x in (md.get("children_ids") or []) if str(x).strip()]:
            child = idx.get(child_id)
            if child:
                out.append(_summary_from_node(child, reason="child_summary", score=0.7))
    return out


def _expand_parent(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    for node in _selected_nodes(graph, selected_node_ids):
        parent_id = str(_node_md(node).get("parent_id") or "").strip()
        parent = idx.get(parent_id)
        if parent:
            out.append(_summary_from_node(parent, reason="parent_summary", score=0.6))
    return out


def _expand_siblings(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    seen = set()
    for node in _selected_nodes(graph, selected_node_ids):
        md = _node_md(node)
        selected_id = str(node.get("node_id") or "")
        parent = idx.get(str(md.get("parent_id") or ""))
        if parent:
            sibling_ids = [str(x) for x in (_node_md(parent).get("children_ids") or []) if str(x).strip()]
        else:
            sibling_ids = [selected_id] if selected_id else []
            sibling_ids.extend([str(x) for x in (md.get("sibling_ids") or []) if str(x).strip()])
        for sibling_id in sibling_ids:
            if sibling_id in seen:
                continue
            sibling = idx.get(sibling_id)
            if sibling:
                seen.add(sibling_id)
                out.append(_summary_from_node(sibling, reason="sibling_summary", score=0.5))
    return out


def _resolve_cross_ref(graph: Dict[str, Any], ref: Tuple[str, str, str], source_md: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    point, clause, article = ref
    doc_id = _norm_key(source_md.get("doc_id"))
    doc_number = _norm_key(source_md.get("doc_number"))
    for node in _node_index(graph).values():
        md = _node_md(node)
        if doc_number and _norm_key(md.get("doc_number")) not in {"", doc_number}:
            continue
        if not doc_number and doc_id and _norm_key(md.get("doc_id")) != doc_id:
            continue
        if article and _norm_key(md.get("article")) != _norm_key(f"Điều {article}"):
            continue
        if clause and _norm_key(md.get("clause")) != _norm_key(f"Khoản {clause}"):
            continue
        if point and _norm_key(md.get("point")) != _norm_key(f"Điểm {point.lower()}"):
            continue
        target = _target_node_type({"article": article, "clause": clause, "point": point})
        if target and _node_type(node) != target:
            continue
        return node
    return None


def _expand_cross_refs(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in _selected_nodes(graph, selected_node_ids):
        md = _node_md(node)
        text = _node_text(node)
        for match in _CROSS_REF_RE.finditer(text):
            resolved = _resolve_cross_ref(graph, (match.group(1) or "", match.group(2) or "", match.group(3) or ""), md)
            if resolved:
                out.append(_summary_from_node(resolved, reason="cross_ref_summary", score=0.6))
    return out


def _dedupe_summaries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        node_id = _summary_candidate_id(item)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        out.append(item)
    return out


def _sort_nodes_by_order(nodes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        nodes,
        key=lambda node: (
            str(_node_md(node).get("doc_id") or ""),
            int(_node_md(node).get("order_index") or 0),
            str(node.get("node_id") or ""),
        ),
    )


def _full_relation_node_ids(
    graph: Dict[str, Any],
    selected_node_ids: Iterable[str],
    action: str,
) -> List[str]:
    idx = _node_index(graph)
    selected = _selected_nodes(graph, selected_node_ids)
    out_nodes: List[Dict[str, Any]] = []

    if action == "deepen_node":
        for node in selected:
            md = _node_md(node)
            if not [x for x in (md.get("children_ids") or []) if str(x).strip()]:
                counterpart = _find_child_bearing_counterpart(graph, node)
                if counterpart:
                    node = counterpart
                    md = _node_md(node)
            for child_id in [str(x) for x in (md.get("children_ids") or []) if str(x).strip()]:
                child = idx.get(child_id)
                if child and not _is_tree_excluded_artifact(child):
                    out_nodes.append(child)
    elif action == "fetch_parent_summary":
        for node in selected:
            parent = idx.get(str(_node_md(node).get("parent_id") or ""))
            if parent and not _is_tree_excluded_artifact(parent):
                out_nodes.append(parent)
    elif action == "fetch_siblings":
        parent_ids = {
            str(_node_md(node).get("parent_id") or "").strip()
            for node in selected
            if str(_node_md(node).get("parent_id") or "").strip()
        }
        for parent_id in parent_ids:
            parent = idx.get(parent_id)
            if not parent:
                continue
            for child_id in [str(x) for x in (_node_md(parent).get("children_ids") or []) if str(x).strip()]:
                child = idx.get(child_id)
                if child and not _is_tree_excluded_artifact(child):
                    out_nodes.append(child)
        if not parent_ids:
            for node in selected:
                md = _node_md(node)
                candidates = [str(node.get("node_id") or "")]
                candidates.extend([str(x) for x in (md.get("sibling_ids") or []) if str(x).strip()])
                for candidate_id in candidates:
                    candidate = idx.get(candidate_id)
                    if candidate and not _is_tree_excluded_artifact(candidate):
                        out_nodes.append(candidate)

    out: List[str] = []
    seen = set()
    for node in _sort_nodes_by_order(out_nodes):
        node_id = str(node.get("node_id") or "")
        if node_id and node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def _low_confidence_relation_action(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> str:
    selected = _selected_nodes(graph, selected_node_ids)
    if not selected:
        return ""

    if _full_relation_node_ids(graph, selected_node_ids, "fetch_siblings"):
        has_leaf = any(
            not [x for x in (_node_md(node).get("children_ids") or []) if str(x).strip()]
            for node in selected
        )
        if has_leaf and len(_full_relation_node_ids(graph, selected_node_ids, "fetch_siblings")) > 1:
            return "fetch_siblings"

    if _full_relation_node_ids(graph, selected_node_ids, "deepen_node"):
        return "deepen_node"
    if _full_relation_node_ids(graph, selected_node_ids, "fetch_parent_summary"):
        return "fetch_parent_summary"
    return ""


def _compact_summary_debug(item: Dict[str, Any], *, limit: int = 280) -> Dict[str, Any]:
    md = dict(item.get("metadata") or {})
    snippet = md.get("text") or item.get("snippet") or item.get("summary_text") or item.get("text") or ""
    return {
        "node_id": str(item.get("node_id") or item.get("id") or ""),
        "node_type": str(item.get("node_type") or md.get("node_type") or ""),
        "artifact_type": str(item.get("artifact_type") or md.get("artifact_type") or ""),
        "doc_number": str(md.get("doc_number") or ""),
        "law_name": str(md.get("official_title") or md.get("law_name") or ""),
        "path_title": str(md.get("path_title") or md.get("title") or md.get("heading_title") or ""),
        "children_count": len([x for x in (md.get("children_ids") or []) if str(x).strip()]),
        "reason": str(item.get("reason") or ""),
        "score": float(item.get("cross_score", item.get("rrf_score", item.get("bm25_score", item.get("summary_score", 0.0)))) or 0.0),
        "snippet": _truncate(snippet, limit),
    }


def _compact_evidence_debug(item: Dict[str, Any], *, limit: int = 360) -> Dict[str, Any]:
    md = dict(item.get("metadata") or {})
    return {
        "node_id": str(item.get("node_id") or item.get("id") or ""),
        "node_type": str(md.get("node_type") or ""),
        "evidence_role": str(item.get("evidence_role") or ""),
        "answer_source": str(item.get("answer_source") or ""),
        "answer_scope": str(item.get("answer_scope") or ""),
        "doc_number": str(md.get("doc_number") or ""),
        "path_title": str(md.get("path_title") or md.get("title") or md.get("heading_title") or ""),
        "text_length": len(str(item.get("text") or "")),
        "snippet": _truncate(item.get("text") or item.get("snippet") or "", limit),
    }


def _evidence_from_node(
    node: Dict[str, Any],
    *,
    text_override: str = "",
    answer_source: str = "own_text",
    answer_scope: str = "",
    evidence_role: str = "primary",
) -> Dict[str, Any]:
    md = _node_md(node)
    node_id = str(node.get("node_id") or md.get("node_id") or md.get("chunk_id") or "")
    text = _norm_space(text_override) or _node_text(node) or _norm_space(node.get("retrieval_text") or "")
    path = _norm_space(md.get("path_title") or md.get("title") or md.get("heading_title") or "")
    shared = f"Vị trí pháp lý: {path}" if path else ""
    return {
        "id": node_id,
        "node_id": node_id,
        "chunk_id": str(md.get("chunk_id") or node_id),
        "doc_id": str(md.get("doc_id") or ""),
        "doc_key": str(md.get("doc_id") or ""),
        "metadata": md,
        "answer_source": answer_source,
        "answer_scope": answer_scope,
        "evidence_role": evidence_role,
        "text": text,
        "local_text": text,
        "shared_text": shared,
        "snippet": _truncate(text, 700),
        "retrieval_text": _norm_space(node.get("retrieval_text") or text),
        "rerank_text_short": _norm_space(node.get("rerank_text") or node.get("retrieval_text") or text)[:1500],
        "final_score": 1.0,
    }


def _is_header_only_node(node: Dict[str, Any]) -> bool:
    text = _ascii_key(_node_text(node))
    md = _node_md(node)
    heading = _ascii_key(md.get("heading_title") or md.get("title") or md.get("path_title") or "")
    child_count = len([x for x in (md.get("children_ids") or []) if str(x).strip()])
    if child_count <= 0:
        return False
    if not text:
        return True
    compact_text = re.sub(r"[^a-z0-9]+", "", text)
    compact_heading = re.sub(r"[^a-z0-9]+", "", heading)
    return bool(compact_text and compact_heading and compact_text in compact_heading)


def _parent_nodes(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    seen = set()
    for node in _selected_nodes(graph, selected_node_ids):
        parent_id = str(_node_md(node).get("parent_id") or "").strip()
        parent = idx.get(parent_id)
        if parent and parent_id not in seen and not _is_tree_excluded_artifact(parent):
            seen.add(parent_id)
            out.append(parent)
    return _sort_nodes_by_order(out)


def _ancestor_heading_nodes(graph: Dict[str, Any], selected_node_ids: Iterable[str]) -> List[Dict[str, Any]]:
    idx = _node_index(graph)
    out: List[Dict[str, Any]] = []
    seen = set()
    for node in _selected_nodes(graph, selected_node_ids):
        current = node
        chosen: Optional[Dict[str, Any]] = None
        while current:
            md = _node_md(current)
            parent_id = str(md.get("parent_id") or "").strip()
            parent = idx.get(parent_id)
            if not parent:
                break
            parent_text = _node_heading_text(parent)
            if parent_text:
                chosen = parent
            current = parent
        if not chosen:
            parents = _parent_nodes(graph, [str(node.get("node_id") or "")])
            chosen = parents[0] if parents else None
        if chosen:
            chosen_id = str(chosen.get("node_id") or "")
            if chosen_id and chosen_id not in seen and not _is_tree_excluded_artifact(chosen):
                seen.add(chosen_id)
                out.append(chosen)
    return _sort_nodes_by_order(out)


def _fetch_evidence_for_answer_source(
    graph: Dict[str, Any],
    selected_node_ids: Iterable[str],
    *,
    answer_source: str,
    answer_scope: str = "",
    evidence_role: str = "primary",
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    source = str(answer_source or "own_text")
    if source == "children":
        ids = _full_relation_node_ids(graph, selected_node_ids, "deepen_node")
        return _fetch_minimal_evidence(
            graph,
            ids,
            limit=len(ids) or limit,
            answer_source="children",
            answer_scope=answer_scope,
            evidence_role=evidence_role,
        ), "deepen_node", ids
    if source == "siblings":
        ids = _full_relation_node_ids(graph, selected_node_ids, "fetch_siblings")
        return _fetch_minimal_evidence(
            graph,
            ids,
            limit=len(ids) or limit,
            answer_source="siblings",
            answer_scope=answer_scope,
            evidence_role=evidence_role,
        ), "fetch_siblings", ids
    if source == "parent":
        ids = _full_relation_node_ids(graph, selected_node_ids, "fetch_parent_summary")
        return _fetch_minimal_evidence(
            graph,
            ids,
            limit=len(ids) or limit,
            answer_source="parent",
            answer_scope=answer_scope,
            evidence_role=evidence_role,
        ), "fetch_parent_summary", ids
    if source == "parent_heading":
        nodes = _ancestor_heading_nodes(graph, selected_node_ids)
        passages = [
            _evidence_from_node(
                node,
                text_override=_node_heading_text(node),
                answer_source="parent_heading",
                answer_scope=answer_scope,
                evidence_role=evidence_role,
            )
            for node in nodes
        ]
        passages = [item for item in passages if _norm_space(item.get("text"))]
        ids = [str(item.get("node_id") or "") for item in passages]
        return passages[: int(limit or len(passages) or 1)], "fetch_parent_summary", ids
    if source == "title_or_heading":
        passages = []
        for node in _selected_nodes(graph, selected_node_ids):
            passages.append(
                _evidence_from_node(
                    node,
                    text_override=_node_heading_text(node),
                    answer_source="title_or_heading",
                    answer_scope=answer_scope,
                    evidence_role=evidence_role,
                )
            )
        passages = [item for item in passages if _norm_space(item.get("text"))]
        ids = [str(item.get("node_id") or "") for item in passages]
        return passages[: int(limit or len(passages) or 1)], "title_or_heading", ids
    passages = _fetch_minimal_evidence(
        graph,
        selected_node_ids,
        limit=limit,
        answer_source="own_text",
        answer_scope=answer_scope,
        evidence_role=evidence_role,
    )
    ids = [str(item.get("node_id") or "") for item in passages]
    return passages, "", ids


def _fetch_minimal_evidence(
    graph: Dict[str, Any],
    selected_node_ids: Iterable[str],
    *,
    limit: Optional[int] = None,
    answer_source: str = "own_text",
    answer_scope: str = "",
    evidence_role: str = "primary",
) -> List[Dict[str, Any]]:
    limit = int(limit or getattr(settings, "tree_evidence_top_k", 4) or 4)
    out: List[Dict[str, Any]] = []
    for node in _selected_nodes(graph, selected_node_ids):
        if _is_tree_excluded_artifact(node):
            continue
        out.append(
            _evidence_from_node(
                node,
                answer_source=answer_source,
                answer_scope=answer_scope,
                evidence_role=evidence_role,
            )
        )
    return [item for item in out if _norm_space(item.get("text"))][:limit]


def _dedupe_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for passage in passages:
        node_id = str(passage.get("node_id") or passage.get("id") or "").strip()
        key = node_id or _norm_space(passage.get("text") or passage.get("snippet") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(passage)
    return out


def _fetch_evidence_for_answer_scope(
    graph: Dict[str, Any],
    selected_node_ids: Iterable[str],
    *,
    primary_node_ids: Iterable[str],
    supporting_node_ids: Iterable[str],
    answer_scope: str,
    answer_source: str,
    relation_evidence_ids: Iterable[str],
    relation_evidence_action: str,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    selected_ids = [str(x) for x in selected_node_ids if str(x).strip()]
    primary_ids = [str(x) for x in primary_node_ids if str(x).strip()] or selected_ids
    supporting_ids = [str(x) for x in supporting_node_ids if str(x).strip()]
    relation_ids = [str(x) for x in relation_evidence_ids if str(x).strip()]
    scope = _normalize_answer_scope(answer_scope)
    source = str(answer_source or "own_text")
    relation_action_for_source = {
        "children": "deepen_node",
        "siblings": "fetch_siblings",
        "parent": "fetch_parent_summary",
    }.get(source, "")

    if scope == "list":
        if relation_ids and relation_action_for_source and relation_evidence_action == relation_action_for_source:
            passages = _fetch_minimal_evidence(
                graph,
                relation_ids,
                limit=len(relation_ids),
                answer_source=source,
                answer_scope=scope,
                evidence_role="primary",
            )
            return passages, relation_evidence_action, relation_ids
        if relation_ids and relation_evidence_action in {"deepen_node", "fetch_siblings"}:
            passages = _fetch_minimal_evidence(
                graph,
                relation_ids,
                limit=len(relation_ids),
                answer_source="children" if relation_evidence_action == "deepen_node" else "siblings",
                answer_scope=scope,
                evidence_role="primary",
            )
            return passages, relation_evidence_action, relation_ids
        if source in {"children", "siblings", "parent"}:
            return _fetch_evidence_for_answer_source(
                graph,
                selected_ids or primary_ids,
                answer_source=source,
                answer_scope=scope,
                evidence_role="primary",
                limit=limit,
            )

    primary_source = source if source in {"own_text", "title_or_heading", "parent_heading"} else "own_text"
    primary_limit = limit
    if scope in {"exact", "definition"}:
        primary_limit = min(int(limit or 2), 2)
    primary_passages, source_action, source_ids = _fetch_evidence_for_answer_source(
        graph,
        primary_ids,
        answer_source=primary_source,
        answer_scope=scope,
        evidence_role="primary",
        limit=primary_limit,
    )

    support_passages: List[Dict[str, Any]] = []
    if scope in {"summary", "comparison", "procedure"}:
        extra_ids = supporting_ids
        if not extra_ids and relation_ids and relation_evidence_action in {"deepen_node", "fetch_siblings", "fetch_parent_summary"}:
            extra_ids = [node_id for node_id in relation_ids if node_id not in set(primary_ids)]
        if extra_ids:
            support_limit = max(0, int(limit or getattr(settings, "tree_evidence_top_k", 4) or 4) - len(primary_passages))
            support_passages = _fetch_minimal_evidence(
                graph,
                extra_ids,
                limit=support_limit,
                answer_source="supporting_relation",
                answer_scope=scope,
                evidence_role="supporting",
            )

    passages = _dedupe_passages(primary_passages + support_passages)
    return passages, source_action, source_ids


def retrieve_tree_guided(
    question: str,
    graph: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    *,
    final_top_k: Optional[int] = None,
    enable_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    question = _norm_space(question)
    timings_ms: Dict[str, float] = {}

    def mark(name: str, started: float) -> float:
        now = time.perf_counter()
        timings_ms[name] = timings_ms.get(name, 0.0) + round((now - started) * 1000, 2)
        return now

    t_stage = time.perf_counter()
    use_llm = bool(getattr(settings, "enable_llm_tree_judge", True)) if enable_llm is None else bool(enable_llm)
    query_profile = plan_query_with_llm(question, enable_llm=use_llm)
    t_stage = mark("query_plan", t_stage)
    merged_filters = dict(query_profile.get("filters") or {})
    if filters:
        merged_filters.update({key: value for key, value in filters.items() if value not in (None, "", [], {})})
    _sanitize_unsafe_year_filter(question, query_profile, merged_filters)
    doc_match: Dict[str, Any] = {}
    if not merged_filters.get("doc_number"):
        match_t = time.perf_counter()
        doc_match = _infer_document_filter_from_question(graph, question)
        mark("question_document_match", match_t)
        if doc_match.get("doc_number"):
            merged_filters["doc_number"] = doc_match["doc_number"]
            doc_year = doc_match.get("year")
            if merged_filters.get("year") not in (None, "") and doc_year not in (None, "") and str(merged_filters.get("year")) != str(doc_year):
                merged_filters.pop("year", None)
                query_profile.pop("year", None)
            query_profile["doc_number"] = doc_match["doc_number"]
            query_profile["document_match"] = doc_match
    query_profile["filters"] = merged_filters
    for key, value in merged_filters.items():
        query_profile[key] = value

    max_hops = max(0, int(getattr(settings, "tree_max_hops", 3) or 3))
    summary_top_k = int(final_top_k or getattr(settings, "tree_summary_top_k", 12) or 12)

    heading_first = bool(query_profile.get("heading_terms")) and not bool(query_profile.get("explicit_reference"))
    heading: List[Dict[str, Any]] = []
    if heading_first:
        heading_t = time.perf_counter()
        heading = _find_heading_match_summaries(graph, question, top_k=summary_top_k * 4, filters=merged_filters)
        heading = _filter_summaries_by_metadata(heading, merged_filters)[:summary_top_k]
        heading = _promote_heading_matches_to_ancestors(graph, heading, query_profile)[:summary_top_k]
        mark("heading_match_lookup", heading_t)

    if heading:
        summaries = heading
        route = "heading_match"
    else:
        direct = _find_direct_target_summaries(graph, query_profile)
        t_stage = mark("direct_target_lookup", t_stage)
        if direct:
            summaries = direct[:summary_top_k]
            route = "direct_tree_jump"
        else:
            if not heading_first:
                heading_t = time.perf_counter()
                heading = _find_heading_match_summaries(graph, question, top_k=summary_top_k, filters=merged_filters)
                heading = _filter_summaries_by_metadata(heading, merged_filters)[:summary_top_k]
                heading = _promote_heading_matches_to_ancestors(graph, heading, query_profile)[:summary_top_k]
                mark("heading_match_lookup", heading_t)
            if heading:
                summaries = heading
                route = "heading_match"
            else:
                summaries = retrieve_summary_candidates(question, graph, query_profile, top_k=summary_top_k, timings_ms=timings_ms)
                route = "summary_retrieval"
    t_stage = mark("initial_summary_retrieval", t_stage)

    if not summaries:
        debug_flow = {
            "pipeline": "tree_guided",
            "route_case": route,
            "query_profile": query_profile,
            "timings_ms": timings_ms,
            "steps": [
                {
                    "type": "no_candidates",
                    "title": "No summary candidates",
                    "status": "insufficient",
                    "detail": "No matching summary nodes were found after planning and filters.",
                }
            ],
            "evidence": [],
        }
        return {
            "mode": "tree_guided",
            "route_case": route,
            "query_profile": query_profile,
            "summary_candidates": [],
            "navigation_trace": [],
            "debug_flow": debug_flow,
            "passages": [],
            "insufficient_context": True,
            "context_grade": {"status": "insufficient", "reason": "no_summary_candidates"},
            "retrieval_timings_ms": timings_ms,
        }

    trace: List[Dict[str, Any]] = []
    current = _dedupe_summaries(summaries)
    selected_ids = [str(current[0].get("node_id") or "")]
    relation_evidence_ids: List[str] = []
    relation_evidence_action = ""
    decision: Dict[str, Any] = {}

    for hop in range(max_hops + 1):
        hop_t = time.perf_counter()
        try:
            decision = _llm_navigation_decision(question, query_profile, current, hop=hop, enable_llm=use_llm, graph=graph)
        except TypeError as exc:
            if "graph" not in str(exc):
                raise
            decision = _llm_navigation_decision(question, query_profile, current, hop=hop, enable_llm=use_llm)
        mark("tree_navigation_decision", hop_t)
        selected_ids = [str(x) for x in (decision.get("selected_node_ids") or []) if str(x).strip()]
        answer_scope = _normalize_answer_scope(decision.get("answer_scope"), question, query_profile)
        decision["answer_scope"] = answer_scope
        trace.append(
            {
                "hop": hop,
                "candidates": [_compact_summary_debug(item) for item in current[: max(1, int(getattr(settings, "tree_judge_top_k", 6) or 6))]],
                "decision": decision.get("decision"),
                "answer_scope": answer_scope,
                "answer_source": decision.get("answer_source") or "own_text",
                "selected_node_ids": selected_ids,
                "primary_node_ids": [str(x) for x in (decision.get("primary_node_ids") or []) if str(x).strip()],
                "supporting_node_ids": [str(x) for x in (decision.get("supporting_node_ids") or []) if str(x).strip()],
                "reason": decision.get("reason"),
                "answerability": decision.get("answerability"),
                "missing_evidence": decision.get("missing_evidence", []),
                "next_query": decision.get("next_query") or "",
                "confidence": decision.get("confidence"),
            }
        )

        action = str(decision.get("decision") or "answer_now")
        try:
            decision_confidence = float(decision.get("confidence") or 0.0)
        except Exception:
            decision_confidence = 0.0
        confidence_threshold = float(getattr(settings, "tree_judge_confidence_threshold", 0.6) or 0.6)
        if use_llm and decision_confidence < confidence_threshold:
            adjusted_action = _low_confidence_relation_action(graph, selected_ids)
            if adjusted_action and adjusted_action != action:
                trace[-1]["decision_adjusted_to"] = adjusted_action
                trace[-1]["adjustment_reason"] = "low_confidence_graph_relation"
                trace[-1]["confidence_threshold"] = confidence_threshold
                action = adjusted_action
        answer_source = str(decision.get("answer_source") or "own_text")
        if action == "answer_now" and answer_source in {"children", "siblings", "parent"}:
            relation_action = {
                "children": "deepen_node",
                "siblings": "fetch_siblings",
                "parent": "fetch_parent_summary",
            }.get(answer_source, "")
            if relation_action:
                action = relation_action
                trace[-1]["decision_adjusted_to"] = action
                trace[-1]["adjustment_reason"] = "answer_source_requires_relation"
        if action == "answer_now" and answer_source == "own_text":
            selected_nodes = _selected_nodes(graph, selected_ids)
            if any(_is_header_only_node(node) for node in selected_nodes):
                action = "deepen_node"
                trace[-1]["decision_adjusted_to"] = action
                trace[-1]["adjustment_reason"] = "header_only_node_needs_children"
        if action in {"answer_now", "refuse_or_clarify"} or hop >= max_hops:
            break

        expanded: List[Dict[str, Any]] = []
        expand_t = time.perf_counter()
        if action == "deepen_node":
            expanded = _expand_children(graph, selected_ids)
        elif action == "fetch_parent_summary":
            expanded = _expand_parent(graph, selected_ids)
        elif action == "fetch_siblings":
            expanded = _expand_siblings(graph, selected_ids)
        elif action == "follow_cross_ref":
            expanded = _expand_cross_refs(graph, selected_ids)
        elif action == "rewrite_query":
            next_query = _norm_space(decision.get("next_query") or question)
            expanded = retrieve_summary_candidates(next_query, graph, query_profile, top_k=summary_top_k, timings_ms=timings_ms)
        if not expanded and action == "deepen_node":
            sibling_ids = _full_relation_node_ids(graph, selected_ids, "fetch_siblings")
            if len(sibling_ids) > 1:
                expanded = _expand_siblings(graph, selected_ids)
                action = "fetch_siblings"
                trace[-1]["decision_adjusted_to"] = "fetch_siblings"
                trace[-1]["adjustment_reason"] = "selected_leaf_has_siblings"
        mark("tree_expand", expand_t)
        trace[-1]["expanded_candidates"] = [_compact_summary_debug(item) for item in expanded[: max(1, summary_top_k)]]

        if not expanded:
            break

        if action in {"deepen_node", "fetch_parent_summary", "fetch_siblings"}:
            relation_ids = _full_relation_node_ids(
                graph,
                selected_ids,
                action,
            )
            if relation_ids:
                relation_evidence_ids = relation_ids
                relation_evidence_action = action
                trace[-1]["relation_evidence_ids"] = relation_evidence_ids
                trace[-1]["relation_evidence_action"] = relation_evidence_action

        rerank_t = time.perf_counter()
        if action == "deepen_node":
            current = _dedupe_summaries(expanded)[: max(summary_top_k * 2, 12)]
        else:
            current = _dedupe_summaries(expanded + current)[: max(summary_top_k * 2, 12)]
        current = _bm25_rank_summaries(question, current, top_k=max(summary_top_k * 2, 12))
        current = cross_rerank(question, current, top_n=min(len(current), summary_top_k))
        mark("tree_expand_rerank", rerank_t)
        trace[-1]["after_rerank"] = [_compact_summary_debug(item) for item in current[: max(1, int(getattr(settings, "tree_judge_top_k", 6) or 6))]]

    if str(decision.get("decision") or "") == "refuse_or_clarify":
        debug_flow = {
            "pipeline": "tree_guided",
            "route_case": route,
            "query_profile": query_profile,
            "timings_ms": timings_ms,
            "steps": trace,
            "evidence": [],
        }
        return {
            "mode": "tree_guided",
            "route_case": route,
            "query_profile": query_profile,
            "summary_candidates": current,
            "navigation_trace": trace,
            "debug_flow": debug_flow,
            "passages": [],
            "insufficient_context": True,
            "context_grade": {
                "status": "insufficient",
                "reason": decision.get("reason") or "tree_judge_refused",
                "navigation_decision": decision,
            },
            "retrieval_timings_ms": timings_ms,
        }

    evidence_t = time.perf_counter()
    evidence_ids = selected_ids or [str(current[0].get("node_id") or "")]
    evidence_limit = final_top_k
    answer_scope = _normalize_answer_scope(decision.get("answer_scope"), question, query_profile)
    answer_source = str(decision.get("answer_source") or "own_text")
    primary_ids = [str(x) for x in (decision.get("primary_node_ids") or []) if str(x).strip()] or selected_ids
    supporting_ids = [str(x) for x in (decision.get("supporting_node_ids") or []) if str(x).strip()]
    passages, source_action, source_ids = _fetch_evidence_for_answer_scope(
        graph,
        evidence_ids,
        primary_node_ids=primary_ids,
        supporting_node_ids=supporting_ids,
        answer_scope=answer_scope,
        answer_source=answer_source,
        relation_evidence_ids=relation_evidence_ids,
        relation_evidence_action=relation_evidence_action,
        limit=evidence_limit,
    )
    if source_action and not relation_evidence_action:
        relation_evidence_action = source_action
    if source_ids and not relation_evidence_ids:
        relation_evidence_ids = source_ids
    if not passages and current:
        passages = _fetch_minimal_evidence(
            graph,
            [str(current[0].get("node_id") or "")],
            limit=final_top_k,
            answer_scope=answer_scope,
            evidence_role="primary",
        )
    mark("fetch_minimal_evidence", evidence_t)
    debug_flow = {
        "pipeline": "tree_guided",
        "route_case": route,
        "query_profile": query_profile,
        "timings_ms": timings_ms,
        "steps": trace,
        "relation_evidence_action": relation_evidence_action,
        "relation_evidence_ids": relation_evidence_ids,
        "answer_scope": answer_scope,
        "answer_source": answer_source,
        "primary_node_ids": primary_ids,
        "supporting_node_ids": supporting_ids,
        "evidence": [_compact_evidence_debug(item) for item in passages],
    }

    answerability = str(decision.get("answerability") or "sufficient")
    status = "needs_reasoning" if bool(decision.get("needs_reasoning")) else "sufficient"
    if not passages:
        status = "insufficient"
    elif answerability == "insufficient":
        status = "needs_reasoning"

    return {
        "mode": "tree_guided",
        "route_case": route,
        "query_profile": query_profile,
        "summary_candidates": current,
        "navigation_trace": trace,
        "debug_flow": debug_flow,
        "passages": passages,
        "insufficient_context": status == "insufficient",
        "context_grade": {
            "status": status,
            "reason": decision.get("reason") or "tree_guided_selection",
            "navigation_decision": decision,
            "answer_scope": answer_scope,
            "answer_source": answer_source,
        },
        "retrieval_timings_ms": timings_ms,
    }

