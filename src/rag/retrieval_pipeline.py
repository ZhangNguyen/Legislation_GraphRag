from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.rag.hybrid import bm25_score, tokenize
from src.rag.openai_clients import get_embedings
from src.rag.rerank_cross import cross_rerank

logger = logging.getLogger(__name__)

SUMMARY_EDGE = "HAS_CHILD_SUMMARY"
SUMMARIZES_EDGE = "SUMMARIZES"
_DOCUMENT_TOP_K = 2
_PASSAGES_PER_DOC = 5
_FINAL_TOP_K = 5

_DOC_EMBED_CACHE: Dict[str, List[float]] = {}
_PASSAGE_EMBED_CACHE: Dict[str, List[float]] = {}
_QUESTION_EMBED_CACHE: Dict[str, List[float]] = {}


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(_norm_space(text).encode("utf-8")).hexdigest()


def _graph_runtime_cache(graph: Dict[str, Any]) -> Dict[str, Any]:
    cache = graph.get("__retrieval_runtime_cache__")
    if isinstance(cache, dict):
        return cache
    cache = {}
    graph["__retrieval_runtime_cache__"] = cache
    return cache


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


def _node_index(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx = graph.get("node_index", {}) or {}
    if idx:
        return idx
    runtime_cache = _graph_runtime_cache(graph)
    cached = runtime_cache.get("node_index") or {}
    if cached:
        return cached
    out: Dict[str, Dict[str, Any]] = {}
    for node in graph.get("nodes", []) or []:
        node_id = str(node.get("node_id") or "").strip()
        if node_id:
            out[node_id] = node
    runtime_cache["node_index"] = out
    return out


def _adjacency(graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    adj = graph.get("adjacency", {}) or {}
    if adj:
        return adj
    runtime_cache = _graph_runtime_cache(graph)
    cached = runtime_cache.get("adjacency") or {}
    if cached:
        return cached
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges", []) or []:
        source_id = str(edge.get("source_id") or "").strip()
        if source_id:
            out[source_id].append(edge)
    runtime_cache["adjacency"] = dict(out)
    return runtime_cache["adjacency"]


def _reverse_adjacency(graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    rev = graph.get("reverse_adjacency", {}) or {}
    if rev:
        return rev
    runtime_cache = _graph_runtime_cache(graph)
    cached = runtime_cache.get("reverse_adjacency") or {}
    if cached:
        return cached
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges", []) or []:
        target_id = str(edge.get("target_id") or "").strip()
        if target_id:
            out[target_id].append(edge)
    runtime_cache["reverse_adjacency"] = dict(out)
    return runtime_cache["reverse_adjacency"]


def _candidate_doc_key(metadata: Dict[str, Any]) -> str:
    law_name = str(metadata.get("law_name") or metadata.get("official_title") or "").strip().lower()
    if law_name:
        return law_name

    document_id = str(metadata.get("document_id") or "").strip().lower()
    if document_id:
        return document_id

    file_name = str(metadata.get("file_name") or metadata.get("filename") or metadata.get("file_stem") or "").strip().lower()
    if file_name:
        return file_name

    chunk_id = str(metadata.get("chunk_id") or "").strip().lower()
    if "::" in chunk_id:
        return chunk_id.split("::")[0]

    source_path = str(metadata.get("source_path") or metadata.get("file_path") or "").strip().lower()
    if source_path:
        return source_path

    source = str(metadata.get("source") or metadata.get("issuing_agency") or "").strip().lower()
    if source:
        return source

    return "unknown"


def _node_type(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or "").strip().lower()


def _summary_level(node: Dict[str, Any]) -> str:
    return str((node.get("metadata", {}) or {}).get("summary_level") or "").strip().lower()


def _is_summary_node(node: Dict[str, Any]) -> bool:
    md = node.get("metadata", {}) or {}
    return md.get("artifact_type") == "summary"


def _is_evidence_node(node: Dict[str, Any]) -> bool:
    md = node.get("metadata", {}) or {}
    return md.get("artifact_type") == "evidence"


def _path_label(md: Dict[str, Any]) -> str:
    for key in ["path_title", "title", "article", "clause", "point", "section", "subsection"]:
        value = md.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "-"


def _rrf_fuse(rank_lists: List[List[str]], *, k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, item_id in enumerate(ranked, start=1):
            if not item_id:
                continue
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


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
        embeddings = get_embedings()
        vectors = embeddings.embed_documents([text for _, text in misses])
        for (key, _), vec in zip(misses, vectors):
            cache[key] = list(vec)
            out[key] = cache[key]

    return out


def _question_embedding(question: str) -> List[float]:
    q = _norm_space(question)
    key = _text_cache_key(q)
    cached = _QUESTION_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    embeddings = get_embedings()
    vec = list(embeddings.embed_query(q))
    _QUESTION_EMBED_CACHE[key] = vec
    return vec


def _bm25_value(question: str, text: str) -> float:
    return bm25_score(tokenize(question), tokenize(text))


def _sorted_ids_by_score(items: List[Dict[str, Any]], score_key: str) -> List[str]:
    ranked = sorted(items, key=lambda x: float(x.get(score_key, 0.0) or 0.0), reverse=True)
    return [str(item.get("id") or "") for item in ranked if str(item.get("id") or "")]


def _infer_node_doc_keys(
    node_id: str,
    node_idx: Dict[str, Dict[str, Any]],
    adjacency: Dict[str, List[Dict[str, Any]]],
    cache: Dict[str, Set[str]],
) -> Set[str]:
    if node_id in cache:
        return cache[node_id]

    node = node_idx.get(node_id) or {}
    md = node.get("metadata", {}) or {}
    direct_doc_key = _candidate_doc_key(md)
    if direct_doc_key != "unknown":
        cache[node_id] = {direct_doc_key}
        return cache[node_id]

    doc_keys: Set[str] = set()

    for source_id in md.get("source_node_ids", []) or []:
        source_id = str(source_id or "").strip()
        if source_id and source_id in node_idx:
            doc_keys.update(_infer_node_doc_keys(source_id, node_idx, adjacency, cache))

    if not doc_keys:
        for edge in adjacency.get(node_id, []) or []:
            rel = str(edge.get("relation_type") or "").strip().upper()
            if rel not in {SUMMARY_EDGE, SUMMARIZES_EDGE}:
                continue
            child_id = str(edge.get("target_id") or "").strip()
            if child_id and child_id in node_idx:
                doc_keys.update(_infer_node_doc_keys(child_id, node_idx, adjacency, cache))

    cache[node_id] = doc_keys
    return doc_keys


def _build_doc_catalog(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    runtime_cache = _graph_runtime_cache(graph)
    cached = runtime_cache.get("doc_catalog")
    if isinstance(cached, list) and cached:
        return cached

    node_idx = _node_index(graph)
    adjacency = _adjacency(graph)
    infer_cache: Dict[str, Set[str]] = {}
    buckets: Dict[str, Dict[str, Any]] = {}

    for node_id, node in node_idx.items():
        md = node.get("metadata", {}) or {}
        doc_keys = _infer_node_doc_keys(node_id, node_idx, adjacency, infer_cache)
        if not doc_keys:
            direct = _candidate_doc_key(md)
            if direct != "unknown":
                doc_keys = {direct}
        if len(doc_keys) != 1:
            continue
        doc_key = next(iter(doc_keys))
        bucket = buckets.setdefault(
            doc_key,
            {
                "id": doc_key,
                "doc_key": doc_key,
                "law_name": md.get("official_title") or md.get("law_name") or md.get("document_title") or doc_key,
                "law_type": md.get("doc_type") or md.get("law_type") or "Unknown",
                "source": md.get("source") or md.get("issuing_agency") or "LocalFile",
                "doc_summary": "",
                "doc_summary_node_id": None,
                "member_node_ids": set(),
                "summary_node_ids": set(),
                "base_summary_node_ids": set(),
            },
        )
        bucket["member_node_ids"].add(node_id)

        node_text = _norm_space(str(node.get("text") or ""))
        if _is_summary_node(node):
            bucket["summary_node_ids"].add(node_id)
            if _summary_level(node) == "document" and node_text:
                bucket["doc_summary"] = node_text
                bucket["doc_summary_node_id"] = node_id
            elif _summary_level(node) != "document":
                bucket["base_summary_node_ids"].add(node_id)

        if not bucket["doc_summary"]:
            md_summary = _norm_space(str(md.get("doc_summary") or ""))
            if md_summary:
                bucket["doc_summary"] = md_summary

    out: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        bucket["member_node_ids"] = sorted(bucket["member_node_ids"])
        bucket["summary_node_ids"] = sorted(bucket["summary_node_ids"])
        bucket["base_summary_node_ids"] = sorted(bucket["base_summary_node_ids"])
        if bucket["doc_summary"]:
            out.append(bucket)

    runtime_cache["doc_catalog"] = sorted(out, key=lambda x: str(x.get("law_name") or x.get("doc_key") or ""))
    return runtime_cache["doc_catalog"]


def rank_documents_by_summary_rrf(
    question: str,
    graph: Dict[str, Any],
    *,
    top_k: int = _DOCUMENT_TOP_K,
    question_vec: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    docs = _build_doc_catalog(graph)
    if not docs:
        return []

    question_vec = question_vec or _question_embedding(question)
    summary_texts = [str(doc.get("doc_summary") or "") for doc in docs]
    summary_embeds = _batch_embed_texts(summary_texts, cache=_DOC_EMBED_CACHE)

    dense_items: List[Dict[str, Any]] = []
    bm25_items: List[Dict[str, Any]] = []
    id_to_doc: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        doc_id = str(doc.get("id") or "")
        summary_text = str(doc.get("doc_summary") or "")
        embed = summary_embeds.get(_text_cache_key(summary_text), [])
        dense_score = _cosine(question_vec, embed)
        bm25_val = _bm25_value(question, summary_text)
        item = {"id": doc_id, "dense_score": dense_score, "bm25_score": bm25_val}
        dense_items.append(item)
        bm25_items.append(item)
        id_to_doc[doc_id] = doc

    fused = _rrf_fuse([
        _sorted_ids_by_score(dense_items, "dense_score"),
        _sorted_ids_by_score(bm25_items, "bm25_score"),
    ])

    ranked: List[Dict[str, Any]] = []
    for doc_id, doc in id_to_doc.items():
        summary_text = str(doc.get("doc_summary") or "")
        embed = summary_embeds.get(_text_cache_key(summary_text), [])
        dense_score = _cosine(question_vec, embed)
        bm25_val = _bm25_value(question, summary_text)
        item = dict(doc)
        item["text"] = summary_text
        item["dense_score"] = float(dense_score)
        item["bm25_score"] = float(bm25_val)
        item["rrf_score"] = float(fused.get(doc_id, 0.0))
        item["hybrid_score"] = item["rrf_score"]
        ranked.append(item)

    ranked.sort(key=lambda x: (float(x.get("rrf_score", 0.0)), float(x.get("dense_score", 0.0))), reverse=True)
    return ranked[: max(1, int(top_k or _DOCUMENT_TOP_K))]


def _same_doc(summary_node_id: str, allowed_summary_ids: Set[str]) -> bool:
    return summary_node_id in allowed_summary_ids


def _one_hop_summary_relations(
    node_id: str,
    graph: Dict[str, Any],
    *,
    allowed_summary_ids: Set[str],
) -> Tuple[List[str], List[str], List[str]]:
    adjacency = _adjacency(graph)
    reverse = _reverse_adjacency(graph)

    parents: List[str] = []
    children: List[str] = []
    siblings: List[str] = []
    sibling_seen: Set[str] = set()

    for edge in reverse.get(node_id, []) or []:
        if str(edge.get("relation_type") or "").strip().upper() != SUMMARY_EDGE:
            continue
        parent_id = str(edge.get("source_id") or "").strip()
        if parent_id and _same_doc(parent_id, allowed_summary_ids):
            parents.append(parent_id)

    for edge in adjacency.get(node_id, []) or []:
        if str(edge.get("relation_type") or "").strip().upper() != SUMMARY_EDGE:
            continue
        child_id = str(edge.get("target_id") or "").strip()
        if child_id and _same_doc(child_id, allowed_summary_ids):
            children.append(child_id)

    for parent_id in parents:
        for edge in adjacency.get(parent_id, []) or []:
            if str(edge.get("relation_type") or "").strip().upper() != SUMMARY_EDGE:
                continue
            sib_id = str(edge.get("target_id") or "").strip()
            if not sib_id or sib_id == node_id or not _same_doc(sib_id, allowed_summary_ids):
                continue
            if sib_id not in sibling_seen:
                sibling_seen.add(sib_id)
                siblings.append(sib_id)

    return sorted(set(parents)), sorted(set(children)), siblings


def _format_context_blocks(label: str, texts: Iterable[str]) -> str:
    clean = [_norm_space(text) for text in texts if _norm_space(text)]
    if not clean:
        return ""
    return f"[{label}]\n" + "\n".join(clean)


def _path_line(md: Dict[str, Any], *, suffix: str) -> str:
    return f"[Vị trí] {_path_label(md)} | {suffix}"


def _short_anchor(text: str, *, limit: int = 240) -> str:
    clean = _norm_space(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _build_shared_local_fields(passages: List[Dict[str, Any]], doc_info: Dict[str, Any]) -> None:
    if not passages:
        return

    doc_title = _norm_space(str(doc_info.get("law_name") or doc_info.get("doc_key") or ""))
    doc_summary = _norm_space(str(doc_info.get("doc_summary") or ""))

    parent_counter: Counter[str] = Counter()
    sibling_counter: Counter[str] = Counter()
    child_counter: Counter[str] = Counter()

    for passage in passages:
        raw = passage.get("_raw_context") or {}
        parent_counter.update({_norm_space(text): 1 for text in raw.get("parent_texts", []) or [] if _norm_space(text)})
        sibling_counter.update({_norm_space(text): 1 for text in raw.get("sibling_texts", []) or [] if _norm_space(text)})
        child_counter.update({_norm_space(text): 1 for text in raw.get("child_texts", []) or [] if _norm_space(text)})

    shared_parent_texts = [text for text, count in parent_counter.items() if count >= 2]
    shared_sibling_texts = [text for text, count in sibling_counter.items() if count >= 2]
    shared_child_texts = [text for text, count in child_counter.items() if count >= 2]

    shared_parts: List[str] = []
    if doc_title:
        shared_parts.append(f"[Văn bản] {doc_title}")
    if doc_summary:
        shared_parts.append(f"[Tóm tắt văn bản]\n{doc_summary}")
    parent_block = _format_context_blocks("Cha chung", shared_parent_texts)
    sibling_block = _format_context_blocks("Sibling chung", shared_sibling_texts)
    child_block = _format_context_blocks("Con chung", shared_child_texts)
    for block in [parent_block, sibling_block, child_block]:
        if block:
            shared_parts.append(block)
    shared_text = "\n\n".join(part for part in shared_parts if _norm_space(part))

    shared_parent_set = {text.lower() for text in shared_parent_texts}
    shared_sibling_set = {text.lower() for text in shared_sibling_texts}
    shared_child_set = {text.lower() for text in shared_child_texts}

    for passage in passages:
        raw = passage.get("_raw_context") or {}
        md = dict(passage.get("metadata") or {})
        unique_parent_texts = [
            text for text in raw.get("parent_texts", []) or []
            if _norm_space(text) and _norm_space(text).lower() not in shared_parent_set
        ]
        unique_child_texts = [
            text for text in raw.get("child_texts", []) or []
            if _norm_space(text) and _norm_space(text).lower() not in shared_child_set
        ]
        unique_sibling_texts = [
            text for text in raw.get("sibling_texts", []) or []
            if _norm_space(text) and _norm_space(text).lower() not in shared_sibling_set
        ]

        local_parts: List[str] = []

        self_text = _norm_space(str(raw.get("self_text") or passage.get("text") or ""))
        if self_text:
            local_parts.append(f"[Chính node]\n{self_text}")

        if unique_parent_texts:
            local_parts.append("[Neo cha]\n" + "\n".join(_short_anchor(text) for text in unique_parent_texts[:2]))
        if unique_child_texts:
            local_parts.append("[Neo con]\n" + "\n".join(_short_anchor(text) for text in unique_child_texts[:2]))
        elif unique_sibling_texts:
            local_parts.append("[Neo sibling]\n" + "\n".join(_short_anchor(text) for text in unique_sibling_texts[:2]))

        passage["shared_text"] = shared_text
        passage["local_text"] = "\n\n".join(part for part in local_parts if _norm_space(part))
        passage.setdefault("bundle_text", passage.get("retrieval_text") or passage.get("text") or "")


def _join_summary_texts(
    *,
    graph: Dict[str, Any],
    node_id: str,
    doc_info: Dict[str, Any],
) -> Dict[str, Any]:
    node_idx = _node_index(graph)
    node = node_idx.get(node_id) or {}
    md = dict(node.get("metadata", {}) or {})
    allowed_summary_ids = set(doc_info.get("summary_node_ids") or [])

    parents, children, siblings = _one_hop_summary_relations(
        node_id,
        graph,
        allowed_summary_ids=allowed_summary_ids,
    )

    def _summary_text(nid: str) -> str:
        return _norm_space(str((node_idx.get(nid) or {}).get("text") or ""))

    self_text = _summary_text(node_id)
    parent_texts = [_summary_text(nid) for nid in parents if _summary_text(nid)]
    child_texts = [_summary_text(nid) for nid in children if _summary_text(nid)]
    sibling_texts = [_summary_text(nid) for nid in siblings if _summary_text(nid)]
    doc_summary = _norm_space(str(doc_info.get("doc_summary") or ""))

    header = f"[Văn bản] {doc_info.get('law_name') or doc_info.get('doc_key')}"
    path_line = _path_line(md, suffix=f"summary_level={_summary_level(node) or _node_type(node) or '-'}")

    retrieval_sections: List[str] = [header, path_line]
    parent_block = _format_context_blocks("Tóm tắt cha 1 hop", parent_texts)
    child_block = _format_context_blocks("Tóm tắt con 1 hop", child_texts)
    sibling_block = _format_context_blocks("Tóm tắt sibling cùng cha", sibling_texts)
    if parent_block:
        retrieval_sections.append(parent_block)
    retrieval_sections.append(f"[Tóm tắt chính node]\n{self_text}")
    if child_block:
        retrieval_sections.append(child_block)
    if sibling_block:
        retrieval_sections.append(sibling_block)

    retrieval_text = "\n\n".join(part for part in retrieval_sections if _norm_space(part))

    rerank_sections: List[str] = [header]
    if doc_summary:
        rerank_sections.append(f"[Tóm tắt văn bản]\n{doc_summary}")
    rerank_sections.append(path_line)
    if parent_block:
        rerank_sections.append(parent_block)
    rerank_sections.append(f"[Tóm tắt chính node]\n{self_text}")
    if child_block:
        rerank_sections.append(child_block)
    rerank_text = "\n\n".join(part for part in rerank_sections if _norm_space(part))

    return {
        "node_id": node_id,
        "text": self_text,
        "snippet": retrieval_text[:700],
        "bundle_text": retrieval_text,
        "retrieval_text": retrieval_text,
        "rerank_text": rerank_text,
        "shared_text": "",
        "local_text": "",
        "metadata": {
            **md,
            "doc_key": doc_info.get("doc_key"),
            "law_name": doc_info.get("law_name") or md.get("law_name"),
            "law_type": doc_info.get("law_type") or md.get("law_type"),
            "source": doc_info.get("source") or md.get("source"),
            "parent_summary_ids": parents,
            "child_summary_ids": children,
            "sibling_summary_ids": siblings,
        },
        "doc_key": str(doc_info.get("doc_key") or ""),
        "doc_score": float(doc_info.get("rrf_score", 0.0) or 0.0),
        "_raw_context": {
            "self_text": self_text,
            "parent_texts": parent_texts,
            "child_texts": child_texts,
            "sibling_texts": sibling_texts,
            "path_suffix": f"summary_level={_summary_level(node) or _node_type(node) or '-'}",
        },
    }


def _join_evidence_texts(
    *,
    graph: Dict[str, Any],
    node_id: str,
    doc_info: Dict[str, Any],
) -> Dict[str, Any]:
    node_idx = _node_index(graph)
    node = node_idx.get(node_id) or {}
    md = dict(node.get("metadata", {}) or {})
    allowed_member_ids = set(doc_info.get("member_node_ids") or [])

    def _evidence_text(nid: str) -> str:
        if not nid or nid not in allowed_member_ids:
            return ""
        target = node_idx.get(nid) or {}
        return _norm_space(str(target.get("text") or ""))

    parent_ids: List[str] = []
    parent_id = str(md.get("parent_id") or "").strip()
    if parent_id and parent_id in allowed_member_ids:
        parent_ids.append(parent_id)

    child_ids: List[str] = []
    for nid in md.get("children_ids", []) or []:
        cid = str(nid or "").strip()
        if cid and cid in allowed_member_ids:
            child_ids.append(cid)

    sibling_ids: List[str] = []
    for nid in md.get("sibling_ids", []) or []:
        sid = str(nid or "").strip()
        if sid and sid in allowed_member_ids:
            sibling_ids.append(sid)

    self_text = _norm_space(str(node.get("text") or ""))
    parent_texts = [_evidence_text(nid) for nid in parent_ids if _evidence_text(nid)]
    child_texts = [_evidence_text(nid) for nid in child_ids if _evidence_text(nid)]
    sibling_texts = [_evidence_text(nid) for nid in sibling_ids if _evidence_text(nid)]
    doc_summary = _norm_space(str(doc_info.get("doc_summary") or ""))

    header = f"[Văn bản] {doc_info.get('law_name') or doc_info.get('doc_key')}"
    path_line = _path_line(md, suffix=f"node_type={_node_type(node) or '-'}")

    retrieval_sections: List[str] = [header, path_line]
    parent_block = _format_context_blocks("Cha 1 hop", parent_texts)
    child_block = _format_context_blocks("Con 1 hop", child_texts)
    sibling_block = _format_context_blocks("Sibling cùng cha", sibling_texts)
    if parent_block:
        retrieval_sections.append(parent_block)
    retrieval_sections.append(f"[Chính node]\n{self_text}")
    if child_block:
        retrieval_sections.append(child_block)
    if sibling_block:
        retrieval_sections.append(sibling_block)

    retrieval_text = "\n\n".join(part for part in retrieval_sections if _norm_space(part))

    rerank_sections: List[str] = [header]
    if doc_summary:
        rerank_sections.append(f"[Tóm tắt văn bản]\n{doc_summary}")
    rerank_sections.append(path_line)
    if parent_block:
        rerank_sections.append(parent_block)
    rerank_sections.append(f"[Chính node]\n{self_text}")
    if child_block:
        rerank_sections.append(child_block)
    rerank_text = "\n\n".join(part for part in rerank_sections if _norm_space(part))

    return {
        "node_id": node_id,
        "text": self_text,
        "snippet": retrieval_text[:700],
        "bundle_text": retrieval_text,
        "retrieval_text": retrieval_text,
        "rerank_text": rerank_text,
        "shared_text": "",
        "local_text": "",
        "metadata": {
            **md,
            "doc_key": doc_info.get("doc_key"),
            "law_name": doc_info.get("law_name") or md.get("law_name"),
            "law_type": doc_info.get("law_type") or md.get("law_type"),
            "source": doc_info.get("source") or md.get("source"),
            "parent_node_ids": parent_ids,
            "child_node_ids": child_ids,
            "sibling_node_ids": sibling_ids,
        },
        "doc_key": str(doc_info.get("doc_key") or ""),
        "doc_score": float(doc_info.get("rrf_score", 0.0) or 0.0),
        "_raw_context": {
            "self_text": self_text,
            "parent_texts": parent_texts,
            "child_texts": child_texts,
            "sibling_texts": sibling_texts,
            "path_suffix": f"node_type={_node_type(node) or '-'}",
        },
    }


def _build_candidate_passages_for_doc(question: str, graph: Dict[str, Any], doc_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    _ = question
    node_idx = _node_index(graph)
    summary_ids = list(doc_info.get("base_summary_node_ids") or [])
    evidence_ids: List[str] = []
    for node_id in doc_info.get("member_node_ids") or []:
        node = node_idx.get(str(node_id)) or {}
        if _is_evidence_node(node):
            evidence_ids.append(str(node_id))

    candidates: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()

    for node_id in summary_ids:
        if node_id in used_ids:
            continue
        used_ids.add(node_id)
        joined = _join_summary_texts(graph=graph, node_id=node_id, doc_info=doc_info)
        if _norm_space(str(joined.get("retrieval_text") or "")):
            candidates.append(joined)

    for node_id in evidence_ids:
        if node_id in used_ids:
            continue
        used_ids.add(node_id)
        joined = _join_evidence_texts(graph=graph, node_id=node_id, doc_info=doc_info)
        if _norm_space(str(joined.get("retrieval_text") or "")):
            candidates.append(joined)

    if not candidates and doc_info.get("doc_summary"):
        fallback_text = _norm_space(str(doc_info.get("doc_summary") or ""))
        candidates.append({
            "node_id": str(doc_info.get("doc_summary_node_id") or doc_info.get("doc_key") or doc_info.get("id") or ""),
            "text": fallback_text,
            "snippet": fallback_text[:700],
            "bundle_text": fallback_text,
            "retrieval_text": fallback_text,
            "rerank_text": fallback_text,
            "shared_text": f"[Văn bản] {doc_info.get('law_name') or doc_info.get('doc_key')}\n\n[Tóm tắt văn bản]\n{fallback_text}",
            "local_text": "[Chính node]\n" + fallback_text,
            "metadata": {
                "doc_key": doc_info.get("doc_key"),
                "law_name": doc_info.get("law_name"),
                "law_type": doc_info.get("law_type"),
                "source": doc_info.get("source"),
                "path_title": "Tóm tắt văn bản",
                "node_type": "document_summary",
            },
            "doc_key": str(doc_info.get("doc_key") or ""),
            "doc_score": float(doc_info.get("rrf_score", 0.0) or 0.0),
            "_raw_context": {
                "self_text": fallback_text,
                "parent_texts": [],
                "child_texts": [],
                "sibling_texts": [],
                "path_suffix": "node_type=document_summary",
            },
        })

    return candidates


def _rank_passages_rrf(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    question_vec: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    if not passages:
        return []

    question_vec = question_vec or _question_embedding(question)
    joined_texts = [str(p.get("retrieval_text") or "") for p in passages]
    embeds = _batch_embed_texts(joined_texts, cache=_PASSAGE_EMBED_CACHE)

    dense_items: List[Dict[str, Any]] = []
    bm25_items: List[Dict[str, Any]] = []

    for passage in passages:
        pid = str(passage.get("node_id") or "")
        joined_text = str(passage.get("retrieval_text") or "")
        embed = embeds.get(_text_cache_key(joined_text), [])
        dense_score = _cosine(question_vec, embed)
        bm25_val = _bm25_value(question, joined_text)
        dense_items.append({"id": pid, "dense_score": dense_score})
        bm25_items.append({"id": pid, "bm25_score": bm25_val})

    fused = _rrf_fuse([
        _sorted_ids_by_score(dense_items, "dense_score"),
        _sorted_ids_by_score(bm25_items, "bm25_score"),
    ])

    ranked: List[Dict[str, Any]] = []
    for item in passages:
        pid = str(item.get("node_id") or "")
        joined_text = str(item.get("retrieval_text") or "")
        embed = embeds.get(_text_cache_key(joined_text), [])
        dense_score = _cosine(question_vec, embed)
        bm25_val = _bm25_value(question, joined_text)
        enriched = dict(item)
        enriched["dense_score"] = float(dense_score)
        enriched["bm25_score"] = float(bm25_val)
        enriched["rrf_score"] = float(fused.get(pid, 0.0))
        enriched["hybrid_score"] = enriched["rrf_score"]
        ranked.append(enriched)

    ranked.sort(key=lambda x: (float(x.get("rrf_score", 0.0)), float(x.get("dense_score", 0.0))), reverse=True)
    return ranked


def _build_doc_passages(
    question: str,
    graph: Dict[str, Any],
    doc_info: Dict[str, Any],
    *,
    question_vec: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    candidates = _build_candidate_passages_for_doc(question, graph, doc_info)
    ranked = _rank_passages_rrf(question, candidates, question_vec=question_vec)
    out = ranked[:_PASSAGES_PER_DOC]
    _build_shared_local_fields(out, doc_info)
    for idx, item in enumerate(out, start=1):
        item["doc_local_rank"] = idx
        item.pop("_raw_context", None)
    return out


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
    _ = filters
    _ = qdrant_top_k
    _ = graph_hops
    _ = max_graph_nodes

    question = _norm_space(question)
    question_vec = _question_embedding(question)
    doc_candidates = rank_documents_by_summary_rrf(question, graph, top_k=_DOCUMENT_TOP_K, question_vec=question_vec)

    doc_groups: List[Dict[str, Any]] = []
    candidate_pool: List[Dict[str, Any]] = []
    for doc in doc_candidates:
        passages = _build_doc_passages(question, graph, doc, question_vec=question_vec)
        doc_groups.append(
            {
                "doc_key": doc.get("doc_key"),
                "doc_score": float(doc.get("rrf_score", 0.0) or 0.0),
                "law_name": doc.get("law_name"),
                "source": doc.get("source"),
                "items": passages,
                "doc_summary": doc.get("doc_summary"),
            }
        )
        candidate_pool.extend(passages)

    cross_limit = len(candidate_pool) if cross_top_k is None else max(0, min(int(cross_top_k), len(candidate_pool)))
    reranked_pool = cross_rerank(question=question, passages=candidate_pool, top_n=cross_limit) if cross_limit > 0 else []

    if cross_limit < len(candidate_pool):
        used = {str(item.get("node_id") or "") for item in reranked_pool}
        remainder = [item for item in candidate_pool if str(item.get("node_id") or "") not in used]
        reranked_pool.extend(remainder)

    reranked_pool.sort(
        key=lambda x: (
            float(x.get("cross_score", 0.0) or 0.0),
            float(x.get("hybrid_score", 0.0) or 0.0),
            float(x.get("doc_score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    final_limit = max(1, int(final_top_k or _FINAL_TOP_K))
    final_passages = reranked_pool[:final_limit]
    for idx, passage in enumerate(final_passages, start=1):
        passage["rank"] = idx
        passage["final_score"] = float(passage.get("cross_score", 0.0) or 0.0)
        passage.pop("_raw_context", None)

    for item in reranked_pool[final_limit:]:
        item.pop("_raw_context", None)

    return {
        "question": question,
        "mode": "doc_summary_rrf_passage_rrf",
        "filters": {},
        "qdrant_top_k": None,
        "final_top_k": final_limit,
        "cross_top_k": cross_limit,
        "seed_candidates": doc_candidates,
        "doc_groups": doc_groups,
        "candidate_pool": reranked_pool,
        "passages": final_passages,
        "query_profile": {
            "pipeline": "doc_summary_rrf -> top5_docs -> joined_summary_passages -> top5_per_doc_rrf -> cross_rerank -> top10",
            "document_top_k": _DOCUMENT_TOP_K,
            "passages_per_doc": _PASSAGES_PER_DOC,
            "uses_qdrant": False,
            "uses_graph_summary_nodes": True,
        },
    }
