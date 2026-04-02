from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.chunking_legal import legal_chunk
from src.rag.document_header import extract_document_header
from src.utils.loader import load_document

DOC_NUMBER_IN_TEXT_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+\b", re.IGNORECASE)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "doc"


def _extract_doc_number(text: str) -> str:
    raw = _norm_space(str(text or ""))
    if not raw:
        return ""
    m = DOC_NUMBER_IN_TEXT_RE.search(raw.upper())
    return _norm_space(m.group(0)).upper() if m else ""


def _infer_year_from_doc_number(text: str) -> int:
    m = re.search(r"/(\d{4})/", str(text or ""))
    return int(m.group(1)) if m else 0


def _normalize_doc_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    md = dict(meta or {})
    doc_number = _extract_doc_number(md.get("doc_number") or md.get("title_block") or md.get("lead_block") or "")
    if doc_number:
        md["doc_number"] = doc_number
    year = int(md.get("year") or 0)
    if not year and md.get("doc_number"):
        md["year"] = _infer_year_from_doc_number(md.get("doc_number"))
    if md.get("doc_type") and not md.get("law_type"):
        md["law_type"] = md.get("doc_type")
    if md.get("law_type") and not md.get("doc_type"):
        md["doc_type"] = md.get("law_type")
    if md.get("official_title") and not md.get("law_name"):
        md["law_name"] = md.get("official_title")
    if md.get("law_name") and not md.get("official_title"):
        md["official_title"] = md.get("law_name")
    return md


def _prefix_chunks(chunks: List[Dict[str, Any]], doc_prefix: str, *, doc_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        item = dict(ch)
        md = dict(item.get("metadata", {}) or {})

        for key in ["node_id", "chunk_id", "parent_id"]:
            val = md.get(key)
            if val:
                md[key] = f"{doc_prefix}::{val}"

        for list_key in ["children_ids", "sibling_ids", "source_node_ids"]:
            vals = md.get(list_key)
            if isinstance(vals, list):
                md[list_key] = [f"{doc_prefix}::{x}" for x in vals if x]

        md["doc_id"] = doc_prefix
        if doc_meta:
            for k, v in doc_meta.items():
                if md.get(k) in (None, "") and v not in (None, ""):
                    md[k] = v

        item["metadata"] = md
        out.append(item)
    return out


def _dedup_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        node_id = n.get("node_id")
        if not node_id:
            continue
        out[str(node_id)] = n
    return list(out.values())


def _dedup_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for e in edges:
        sig = (
            str(e.get("source_id", "")).strip().lower(),
            str(e.get("target_id", "")).strip().lower(),
            str(e.get("relation_type", "")).strip().lower(),
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(e)
    return out


def _build_sibling_map(parent_to_children: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sibling_map: Dict[str, List[str]] = {}
    for _, children in parent_to_children.items():
        for child in children:
            sibling_map[child] = [x for x in children if x != child]
    return sibling_map


def build_graph_from_normalized(input_dir: str, glob_pattern: str) -> Dict[str, Any]:
    base = Path(input_dir)
    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    doc_index: Dict[str, Dict[str, Any]] = {}
    parent_to_children: Dict[str, List[str]] = defaultdict(list)

    for fp in sorted(base.glob(glob_pattern)):
        if fp.suffix.lower() not in {".pdf", ".txt"}:
            continue

        text = load_document(fp)
        header = extract_document_header(text, fallback_name=fp.stem)
        doc_prefix = _slugify(fp.stem)
        doc_meta = _normalize_doc_meta(header.to_metadata())
        doc_meta["doc_id"] = doc_prefix
        doc_meta["file_name"] = fp.name
        doc_meta["source_path"] = str(fp)
        doc_index[doc_prefix] = doc_meta

        chunks = legal_chunk(text, fallback_doc_name=fp.stem)
        chunks = _prefix_chunks(chunks, doc_prefix, doc_meta=doc_meta)

        for ch in chunks:
            md = _normalize_doc_meta(dict(ch.get("metadata") or {}))
            node_id = str(md.get("node_id") or md.get("chunk_id") or "")
            if not node_id:
                continue
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": md.get("node_type") or "text",
                    "text": str(ch.get("text") or "").strip(),
                    "retrieval_text": str(ch.get("retrieval_text") or ch.get("text") or "").strip(),
                    "rerank_text": str(ch.get("rerank_text") or ch.get("retrieval_text") or ch.get("text") or "").strip(),
                    "metadata": md,
                }
            )
            parent_id = md.get("parent_id")
            if parent_id:
                edges.append({"source_id": str(parent_id), "target_id": node_id, "relation_type": "HAS_CHILD"})
                parent_to_children[str(parent_id)].append(node_id)

    nodes = _dedup_nodes(nodes)
    edges = _dedup_edges(edges)
    sibling_map = _build_sibling_map(parent_to_children)
    node_index = {str(n["node_id"]): n for n in nodes if n.get("node_id")}

    for node_id, node in node_index.items():
        md = node.setdefault("metadata", {})
        md["children_ids"] = parent_to_children.get(node_id, md.get("children_ids", []))
        md["sibling_ids"] = sibling_map.get(node_id, md.get("sibling_ids", []))
        doc_id = str(md.get("doc_id") or "")
        if doc_id in doc_index:
            for key, val in doc_index[doc_id].items():
                if md.get(key) in (None, "") and val not in (None, ""):
                    md[key] = val

    return {"nodes": nodes, "edges": edges, "doc_index": doc_index}
