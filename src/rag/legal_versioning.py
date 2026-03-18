from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


VERSION_RELATIONS = {
    "AMENDS",
    "REPEALS",
    "REPLACES",
    "PARTIALLY_AMENDS",
    "PARTIALLY_REPEALS",
    "EFFECTIVE_FROM",
}

DATE_RE = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
    re.IGNORECASE,
)


# =========================
# Basic helpers
# =========================

def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _norm_ref_piece(text: Optional[str]) -> str:
    raw = _norm_space(str(text or "")).lower()
    raw = raw.replace("điều", "").replace("khoản", "").replace("điểm", "")
    raw = re.sub(r"[,:;]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def make_ref_key(
    article: Optional[str],
    clause: Optional[str] = None,
    point: Optional[str] = None,
) -> Tuple[str, str, str]:
    return (
        _norm_ref_piece(article),
        _norm_ref_piece(clause),
        _norm_ref_piece(point),
    )


def _is_evidence(node: Dict[str, Any]) -> bool:
    md = node.get("metadata", {}) or {}
    return md.get("artifact_type") == "evidence"


def _node_md(node: Dict[str, Any]) -> Dict[str, Any]:
    return node.get("metadata", {}) or {}


def _extract_date_candidates(text: str) -> List[str]:
    if not text:
        return []
    return sorted(set(m.group(1) for m in DATE_RE.finditer(text)))


# =========================
# Evidence index by legal ref
# =========================

def build_evidence_ref_index(graph: Dict[str, Any]) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Index:
      (article, clause, point) -> [node_id, ...]
    """
    out: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for node in graph.get("nodes", []):
        if not _is_evidence(node):
            continue

        md = _node_md(node)
        article = md.get("article")
        clause = md.get("clause")
        point = md.get("point")

        if not article:
            continue

        key = make_ref_key(article, clause, point)
        out[key].append(node["node_id"])

    return dict(out)


def resolve_evidence_node_ids(
    evidence_ref_index: Dict[Tuple[str, str, str], List[str]],
    *,
    target_article: Optional[str],
    target_clause: Optional[str] = None,
    target_point: Optional[str] = None,
) -> List[str]:
    """
    Resolve ref theo mức ưu tiên:
    1) article + clause + point
    2) article + clause
    3) article
    """
    article = _norm_ref_piece(target_article)
    clause = _norm_ref_piece(target_clause)
    point = _norm_ref_piece(target_point)

    if not article:
        return []

    exact_key = (article, clause, point)
    if exact_key in evidence_ref_index:
        return evidence_ref_index[exact_key]

    clause_key = (article, clause, "")
    if clause and clause_key in evidence_ref_index:
        return evidence_ref_index[clause_key]

    article_key = (article, "", "")
    if article_key in evidence_ref_index:
        return evidence_ref_index[article_key]

    return []


# =========================
# Collect version facts from graph
# =========================

def _edge_relation(edge: Dict[str, Any]) -> str:
    return str(edge.get("relation_type") or "").strip().upper()


def _source_node(graph: Dict[str, Any], edge: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    node_index = graph.get("node_index", {}) or {}
    source_id = edge.get("source_id")
    if not source_id:
        return None
    return node_index.get(source_id)


def _build_version_event(graph: Dict[str, Any], edge: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rel = _edge_relation(edge)
    if rel not in VERSION_RELATIONS:
        return None

    src_node = _source_node(graph, edge)
    src_text = str((src_node or {}).get("text") or "")
    src_md = _node_md(src_node or {})

    target_article = edge.get("target_article") or src_md.get("target_article")
    target_clause = edge.get("target_clause") or src_md.get("target_clause")
    target_point = edge.get("target_point") or src_md.get("target_point")

    if not target_article and not edge.get("target_id"):
        return None

    date_candidates = _extract_date_candidates(
        _norm_space(" ".join([str(edge.get("target_text") or ""), src_text]))
    )

    return {
        "relation_type": rel,
        "source_id": edge.get("source_id"),
        "target_id": edge.get("target_id"),
        "target_article": target_article,
        "target_clause": target_clause,
        "target_point": target_point,
        "source_text": edge.get("source_text"),
        "target_text": edge.get("target_text"),
        "date_candidates": date_candidates,
    }


def collect_version_events(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for edge in graph.get("edges", []):
        evt = _build_version_event(graph, edge)
        if evt:
            events.append(evt)

    return events


# =========================
# Build per-ref version map
# =========================

def build_version_map(graph: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Output:
      {
        (article, clause, point): {
            "ref": {...},
            "evidence_node_ids": [...],
            "events": [...],
            "status": "...",
            "effective_dates": [...],
            "amended_by": [...],
            "repealed_by": [...],
            "replaced_by": [...],
        }
      }
    """
    evidence_ref_index = build_evidence_ref_index(graph)
    events = collect_version_events(graph)

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def ensure_bucket(key: Tuple[str, str, str]) -> Dict[str, Any]:
        if key not in out:
            out[key] = {
                "ref": {
                    "article": key[0],
                    "clause": key[1],
                    "point": key[2],
                },
                "evidence_node_ids": list(evidence_ref_index.get(key, [])),
                "events": [],
                "status": "base",
                "effective_dates": [],
                "amended_by": [],
                "repealed_by": [],
                "replaced_by": [],
            }
        return out[key]

    for key, node_ids in evidence_ref_index.items():
        bucket = ensure_bucket(key)
        bucket["evidence_node_ids"] = list(node_ids)

    for evt in events:
        key = make_ref_key(
            evt.get("target_article"),
            evt.get("target_clause"),
            evt.get("target_point"),
        )
        if not key[0]:
            continue

        bucket = ensure_bucket(key)
        bucket["events"].append(evt)

        rel = evt["relation_type"]
        if rel in {"AMENDS", "PARTIALLY_AMENDS"}:
            bucket["amended_by"].append(evt["source_id"])
        elif rel in {"REPEALS", "PARTIALLY_REPEALS"}:
            bucket["repealed_by"].append(evt["source_id"])
        elif rel == "REPLACES":
            bucket["replaced_by"].append(evt["source_id"])

        for d in evt.get("date_candidates", []):
            if d not in bucket["effective_dates"]:
                bucket["effective_dates"].append(d)

    for bucket in out.values():
        bucket["status"] = _derive_status(bucket)

    return out


def _derive_status(bucket: Dict[str, Any]) -> str:
    """
    Heuristic status:
    - repealed > replaced > amended > effective_info_found > base
    """
    if bucket.get("repealed_by"):
        return "repealed"
    if bucket.get("replaced_by"):
        return "replaced"
    if bucket.get("amended_by"):
        return "amended"
    if bucket.get("effective_dates"):
        return "effective_info_found"
    return "base"


# =========================
# Public query helpers
# =========================

def resolve_current_state(
    graph: Dict[str, Any],
    *,
    article: str,
    clause: Optional[str] = None,
    point: Optional[str] = None,
) -> Dict[str, Any]:
    version_map = build_version_map(graph)
    key = make_ref_key(article, clause, point)

    if key in version_map:
        return version_map[key]

    # fallback logic: clause -> article, point -> clause/article
    fallback_keys = [
        make_ref_key(article, clause, None),
        make_ref_key(article, None, None),
    ]
    for fk in fallback_keys:
        if fk in version_map:
            result = dict(version_map[fk])
            result["fallback_from"] = {
                "article": article,
                "clause": clause,
                "point": point,
            }
            return result

    return {
        "ref": {
            "article": _norm_ref_piece(article),
            "clause": _norm_ref_piece(clause),
            "point": _norm_ref_piece(point),
        },
        "evidence_node_ids": [],
        "events": [],
        "status": "unknown",
        "effective_dates": [],
        "amended_by": [],
        "repealed_by": [],
        "replaced_by": [],
    }


def annotate_graph_with_versioning(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gắn metadata versioning vào các node evidence tương ứng.
    Không tạo node/edge mới; chỉ enrich metadata.
    """
    version_map = build_version_map(graph)
    node_index = graph.get("node_index", {}) or {}

    for bucket in version_map.values():
        status = bucket.get("status", "base")
        events = bucket.get("events", [])
        effective_dates = bucket.get("effective_dates", [])

        for node_id in bucket.get("evidence_node_ids", []):
            node = node_index.get(node_id)
            if not node:
                continue

            md = _node_md(node)
            md["version_status"] = status
            md["version_event_count"] = len(events)
            md["effective_dates"] = effective_dates
            md["amended_by"] = list(bucket.get("amended_by", []))
            md["repealed_by"] = list(bucket.get("repealed_by", []))
            md["replaced_by"] = list(bucket.get("replaced_by", []))
            node["metadata"] = md

    return graph


def summarize_versioning(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Dùng để debug / inspect nhanh.
    """
    version_map = build_version_map(graph)

    rows: List[Dict[str, Any]] = []
    for bucket in version_map.values():
        ref = bucket["ref"]
        rows.append(
            {
                "article": ref.get("article", ""),
                "clause": ref.get("clause", ""),
                "point": ref.get("point", ""),
                "status": bucket.get("status", "base"),
                "evidence_node_count": len(bucket.get("evidence_node_ids", [])),
                "event_count": len(bucket.get("events", [])),
                "effective_dates": bucket.get("effective_dates", []),
                "amended_by": bucket.get("amended_by", []),
                "repealed_by": bucket.get("repealed_by", []),
                "replaced_by": bucket.get("replaced_by", []),
            }
        )

    rows.sort(key=lambda x: (x["article"], x["clause"], x["point"]))
    return rows