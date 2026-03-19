from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.graph_builder import refresh_graph_indexes
from src.rag.openai_clients import get_llm


SUMMARY_SYSTEM_PROMPT = """
Bạn là bộ tóm tắt pháp lý cho GraphRAG từ văn bản pháp luật Việt Nam.

Mục tiêu:
- Tóm tắt ngắn gọn, đúng ngữ nghĩa pháp lý.
- Không bịa thêm thông tin không có trong input.
- Không suy diễn vượt quá nội dung đầu vào.
- Nếu input là nội dung sửa đổi/bổ sung/bãi bỏ/thay thế/hiệu lực thì phải giữ rõ bản chất đó.
- Ưu tiên giữ các ý quan trọng: chủ thể, hành vi, chế tài, đối tượng áp dụng, hiệu lực, viện dẫn, sửa đổi/bãi bỏ.

Trả về JSON hợp lệ, KHÔNG markdown.

Schema bắt buộc:
{
  "summary": "..."
}

Yêu cầu:
- Viết bằng tiếng Việt.
- Ngắn gọn, rõ nghĩa.
- 1 đến 4 câu tùy lượng thông tin đầu vào.
""".strip()

REL_SUMMARIZES = "SUMMARIZES"
REL_HAS_CHILD_SUMMARY = "HAS_CHILD_SUMMARY"

CHANGE_LIKE_ROLES = {
    "amendment",
    "repeal",
    "effective",
    "applicability",
    "responsibility",
    "transition",
    "correction",
    "replacement",
}

SUMMARY_CACHE: Dict[str, str] = {}


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _safe_json_loads(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    raw = (text or "").strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return fallback


def _slugify(text: str) -> str:
    text = _norm_space(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _hash_payload(*parts: str) -> str:
    payload = "||".join(_norm_space(p) for p in parts if p is not None)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _natural_sort_key(text: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", text or "")
    out: List[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return tuple(out)


def _pack_texts(texts: List[str], max_chars: int) -> List[str]:
    cleaned = [_norm_space(t) for t in texts if _norm_space(t)]
    packed: List[str] = []
    total = 0

    for t in cleaned:
        extra = len(t) + (2 if packed else 0)
        if total + extra > max_chars:
            break
        packed.append(t)
        total += extra

    return packed


def _node_md(node: Dict[str, Any]) -> Dict[str, Any]:
    return node.get("metadata", {}) or {}


def _node_text(node: Dict[str, Any]) -> str:
    return _norm_space(str(node.get("text") or ""))


def _is_evidence_node(node: Dict[str, Any]) -> bool:
    return _node_md(node).get("artifact_type") == "evidence"


def _article_of(node: Dict[str, Any]) -> str:
    return str(_node_md(node).get("article") or "").strip()


def _clause_of(node: Dict[str, Any]) -> str:
    return str(_node_md(node).get("clause") or "").strip()


def _point_of(node: Dict[str, Any]) -> str:
    return str(_node_md(node).get("point") or "").strip()


def _legal_role_of(node: Dict[str, Any]) -> str:
    return str(_node_md(node).get("legal_role") or "").strip().lower()


def _node_type_of(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or "").strip().lower()


def _is_change_like_node(node: Dict[str, Any]) -> bool:
    node_type = _node_type_of(node)
    legal_role = _legal_role_of(node)
    text = _node_text(node).lower()

    if node_type in {"amendment", "effective"}:
        return True
    if legal_role in CHANGE_LIKE_ROLES:
        return True

    keywords = [
        "sửa đổi",
        "bổ sung",
        "bãi bỏ",
        "thay thế",
        "có hiệu lực",
        "hết hiệu lực",
        "trách nhiệm thi hành",
        "điều khoản chuyển tiếp",
        "đính chính",
    ]
    return any(k in text for k in keywords)


def _build_summary_prompt(
    *,
    level: str,
    title: str,
    texts: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    max_input_chars: int = 4500,
) -> str:
    metadata = metadata or {}
    packed_texts = _pack_texts(texts, max_chars=max_input_chars)
    joined_text = "\n\n".join(packed_texts)

    lines = [
        f"summary_level: {level}",
        f"title: {title}",
    ]

    for k, v in metadata.items():
        if v is None or v == "":
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        lines.append(f"{k}: {v}")

    lines.extend(["", "input_text:", joined_text])
    return "\n".join(lines).strip()


def summarize_with_llm(
    *,
    level: str,
    title: str,
    texts: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    max_input_chars: int = 4500,
) -> str:
    metadata = metadata or {}

    cache_key = _hash_payload(
        level,
        title,
        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "\n".join(_pack_texts(texts, max_chars=max_input_chars)),
    )
    if cache_key in SUMMARY_CACHE:
        return SUMMARY_CACHE[cache_key]

    llm = get_llm()
    prompt = _build_summary_prompt(
        level=level,
        title=title,
        texts=texts,
        metadata=metadata,
        max_input_chars=max_input_chars,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        parsed = _safe_json_loads(getattr(response, "content", ""), {"summary": ""})
        summary = _norm_space(str(parsed.get("summary", "")))
    except Exception:
        summary = ""

    if not summary:
        fallback = " ".join(_norm_space(t) for t in texts[:3] if _norm_space(t))
        if len(fallback) > 500:
            fallback = fallback[:500].rstrip() + "..."
        summary = fallback

    SUMMARY_CACHE[cache_key] = summary
    return summary


def _make_summary_node(
    *,
    node_id: str,
    summary_level: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    md = dict(metadata or {})
    md["artifact_type"] = "summary"
    md["summary_level"] = summary_level

    return {
        "node_id": node_id,
        "node_type": "summary",
        "text": text,
        "metadata": md,
    }


def build_clause_summaries(
    graph: Dict[str, Any],
    *,
    max_input_chars: int = 3000,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for node in graph.get("nodes", []):
        if not _is_evidence_node(node):
            continue
        if _node_type_of(node) != "clause":
            continue

        article = _article_of(node)
        clause = _clause_of(node)
        title = " - ".join(x for x in [article, clause] if x) or node["node_id"]

        summary = summarize_with_llm(
            level="clause",
            title=title,
            texts=[_node_text(node)],
            metadata={
                "article": article,
                "clause": clause,
                "legal_role": _node_md(node).get("legal_role"),
            },
            max_input_chars=max_input_chars,
        )

        out.append(
            _make_summary_node(
                node_id=f"summary::clause::{_slugify(node['node_id'])}",
                summary_level="clause",
                text=summary,
                metadata={
                    "article": article,
                    "clause": clause,
                    "legal_role": _node_md(node).get("legal_role"),
                    "source_node_ids": [node["node_id"]],
                    "title": title,
                },
            )
        )

    return out


def build_article_summaries(
    graph: Dict[str, Any],
    *,
    max_input_chars: int = 4500,
    clause_summaries: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    clause_summaries = clause_summaries or []

    for node in graph.get("nodes", []):
        if not _is_evidence_node(node):
            continue
        article = _article_of(node)
        if article:
            by_article[article].append(node)

    clause_summary_by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in clause_summaries:
        article = str(s.get("metadata", {}).get("article") or "").strip()
        if article:
            clause_summary_by_article[article].append(s)

    out: List[Dict[str, Any]] = []

    for article, nodes in sorted(by_article.items(), key=lambda x: _natural_sort_key(x[0])):
        normal_nodes = [n for n in nodes if not _is_change_like_node(n)]
        selected_nodes = normal_nodes if normal_nodes else nodes

        selected_nodes = sorted(
            selected_nodes,
            key=lambda n: (
                0 if _node_type_of(n) == "clause" else 1 if _node_type_of(n) == "point" else 2,
                _natural_sort_key(_clause_of(n)),
                _natural_sort_key(_point_of(n)),
                _natural_sort_key(str(n.get("node_id", ""))),
            ),
        )

        texts = [_node_text(n) for n in selected_nodes if _node_text(n)]
        if not texts:
            continue

        legal_roles = sorted(
            {
                str(_node_md(n).get("legal_role"))
                for n in selected_nodes
                if _node_md(n).get("legal_role")
            }
        )

        child_summary_ids = [s["node_id"] for s in clause_summary_by_article.get(article, [])]

        summary = summarize_with_llm(
            level="article",
            title=article,
            texts=texts,
            metadata={
                "article": article,
                "legal_roles": legal_roles,
            },
            max_input_chars=max_input_chars,
        )

        out.append(
            _make_summary_node(
                node_id=f"summary::article::{_slugify(article)}",
                summary_level="article",
                text=summary,
                metadata={
                    "article": article,
                    "legal_roles": legal_roles,
                    "source_node_ids": [n["node_id"] for n in selected_nodes],
                    "child_summary_ids": child_summary_ids,
                    "title": article,
                },
            )
        )

    return out


def build_change_summaries(
    graph: Dict[str, Any],
    *,
    max_input_chars: int = 4500,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for node in graph.get("nodes", []):
        if not _is_evidence_node(node):
            continue
        if not _is_change_like_node(node):
            continue

        role = _legal_role_of(node) or _node_type_of(node) or "change"
        buckets[role].append(node)

    out: List[Dict[str, Any]] = []

    for role, nodes in sorted(buckets.items(), key=lambda x: _natural_sort_key(x[0])):
        texts = [_node_text(n) for n in nodes if _node_text(n)]
        if not texts:
            continue

        articles = sorted({_article_of(n) for n in nodes if _article_of(n)})

        summary = summarize_with_llm(
            level="change",
            title=role,
            texts=texts,
            metadata={
                "change_role": role,
                "articles": articles,
            },
            max_input_chars=max_input_chars,
        )

        out.append(
            _make_summary_node(
                node_id=f"summary::change::{_slugify(role)}",
                summary_level="change",
                text=summary,
                metadata={
                    "change_role": role,
                    "articles": articles,
                    "source_node_ids": [n["node_id"] for n in nodes],
                    "title": role,
                },
            )
        )

    return out


def build_community_summaries(
    graph: Dict[str, Any],
    *,
    max_input_chars: int = 4500,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for node in graph.get("nodes", []):
        if not _is_evidence_node(node):
            continue
        role = _legal_role_of(node) or "general"
        buckets[role].append(node)

    out: List[Dict[str, Any]] = []

    for role, nodes in sorted(buckets.items(), key=lambda x: _natural_sort_key(x[0])):
        texts = [_node_text(n) for n in nodes if _node_text(n)]
        if not texts:
            continue

        articles = sorted({_article_of(n) for n in nodes if _article_of(n)})

        summary = summarize_with_llm(
            level="community",
            title=role,
            texts=texts,
            metadata={
                "community_role": role,
                "articles": articles,
            },
            max_input_chars=max_input_chars,
        )

        out.append(
            _make_summary_node(
                node_id=f"summary::community::{_slugify(role)}",
                summary_level="community",
                text=summary,
                metadata={
                    "community_id": role,
                    "articles": articles,
                    "source_node_ids": [n["node_id"] for n in nodes],
                    "title": role,
                },
            )
        )

    return out


def build_document_summary(
    *,
    document_title: Optional[str],
    article_summaries: List[Dict[str, Any]],
    change_summaries: List[Dict[str, Any]],
    community_summaries: Optional[List[Dict[str, Any]]] = None,
    max_input_chars: int = 5000,
) -> Optional[Dict[str, Any]]:
    title = _norm_space(document_title or "Văn bản pháp luật")
    community_summaries = community_summaries or []

    source_texts: List[str] = []

    for s in article_summaries:
        if _node_text(s):
            source_texts.append(_node_text(s))

    for s in change_summaries:
        if _node_text(s):
            source_texts.append(_node_text(s))

    for s in community_summaries[:6]:
        if _node_text(s):
            source_texts.append(_node_text(s))

    if not source_texts:
        return None

    summary = summarize_with_llm(
        level="document",
        title=title,
        texts=source_texts,
        metadata={
            "document_title": title,
            "article_summary_count": len(article_summaries),
            "change_summary_count": len(change_summaries),
        },
        max_input_chars=max_input_chars,
    )

    return _make_summary_node(
        node_id=f"summary::document::{_slugify(title)}",
        summary_level="document",
        text=summary,
        metadata={
            "document_title": title,
            "child_summary_ids": (
                [s["node_id"] for s in article_summaries]
                + [s["node_id"] for s in change_summaries]
                + [s["node_id"] for s in community_summaries]
            ),
            "title": title,
        },
    )


def attach_summary_edges(
    graph: Dict[str, Any],
    summary_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    edges = list(graph.get("edges", []))

    for s in summary_nodes:
        md = s.get("metadata", {}) or {}

        for target_id in md.get("source_node_ids", []) or []:
            if target_id:
                edges.append(
                    {
                        "source_id": s["node_id"],
                        "target_id": target_id,
                        "relation_type": REL_SUMMARIZES,
                    }
                )

        for child_id in md.get("child_summary_ids", []) or []:
            if child_id:
                edges.append(
                    {
                        "source_id": s["node_id"],
                        "target_id": child_id,
                        "relation_type": REL_HAS_CHILD_SUMMARY,
                    }
                )

    seen = set()
    uniq_edges = []
    for e in edges:
        sig = (
            str(e.get("source_id", "")).strip().lower(),
            str(e.get("target_id", "")).strip().lower(),
            str(e.get("relation_type", "")).strip().lower(),
        )
        if sig in seen:
            continue
        seen.add(sig)
        uniq_edges.append(e)

    graph["edges"] = uniq_edges
    return graph


def build_hierarchical_summaries(
    graph: Dict[str, Any],
    *,
    document_title: Optional[str] = None,
    include_clause: bool = False,
    include_article: bool = True,
    include_change: bool = True,
    include_community: bool = False,
    max_clause_input_chars: int = 3000,
    max_article_input_chars: int = 4500,
    max_change_input_chars: int = 4500,
    max_community_input_chars: int = 4500,
    max_document_input_chars: int = 5000,
) -> Dict[str, Any]:
    clause_summaries: List[Dict[str, Any]] = []
    if include_clause:
        clause_summaries = build_clause_summaries(
            graph,
            max_input_chars=max_clause_input_chars,
        )

    article_summaries: List[Dict[str, Any]] = []
    if include_article:
        article_summaries = build_article_summaries(
            graph,
            max_input_chars=max_article_input_chars,
            clause_summaries=clause_summaries,
        )

    change_summaries: List[Dict[str, Any]] = []
    if include_change:
        change_summaries = build_change_summaries(
            graph,
            max_input_chars=max_change_input_chars,
        )

    community_summaries: List[Dict[str, Any]] = []
    if include_community:
        community_summaries = build_community_summaries(
            graph,
            max_input_chars=max_community_input_chars,
        )

    document_summary = build_document_summary(
        document_title=document_title,
        article_summaries=article_summaries,
        change_summaries=change_summaries,
        community_summaries=community_summaries,
        max_input_chars=max_document_input_chars,
    )

    summary_nodes: List[Dict[str, Any]] = []
    summary_nodes.extend(clause_summaries)
    summary_nodes.extend(article_summaries)
    summary_nodes.extend(change_summaries)
    summary_nodes.extend(community_summaries)
    if document_summary:
        summary_nodes.append(document_summary)

    graph["nodes"] = list(graph.get("nodes", [])) + summary_nodes
    graph = attach_summary_edges(graph, summary_nodes)
    graph = refresh_graph_indexes(graph)

    return {
        "graph": graph,
        "summary_nodes": summary_nodes,
        "clause_summaries": clause_summaries,
        "article_summaries": article_summaries,
        "change_summaries": change_summaries,
        "community_summaries": community_summaries,
        "document_summary": document_summary,
    }
