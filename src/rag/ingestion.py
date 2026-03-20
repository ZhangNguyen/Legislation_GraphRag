from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client.http import models as qm

from src.rag.chunking_legal import LegalNode, build_chunks, parse_legal_text
from src.rag.openai_clients import get_llm
from src.storage.qdrant_store import get_qdrant_client, upsert_points

logger = logging.getLogger(__name__)

# ============================================================
# LLM PROMPTS
# ============================================================

ENTITY_SYSTEM_PROMPT = """
Bạn là bộ trích xuất thực thể pháp lý cho GraphRAG từ văn bản pháp luật Việt Nam.

Mục tiêu:
- Trích xuất các thực thể có ích cho truy vấn pháp lý đa bước.
- Không suy diễn vượt ra ngoài văn bản đầu vào.
- Giữ nguyên cụm tiếng Việt trong văn bản nếu có thể.
- Trả về JSON hợp lệ, KHÔNG markdown.

Schema bắt buộc:
{
  "entities": [
    {
      "entity_type": "actor | behavior | legal_term | penalty | remedy | document_ref | article_ref | clause_ref | point_ref | date_ref | legal_effect",
      "entity_value": "..."
    }
  ]
}

Giải thích:
- actor: chủ thể như "người lao động", "Ủy ban nhân dân cấp xã", "người có hành vi bạo lực gia đình"
- behavior: hành vi như "bạo lực gia đình", "cố ý gây thương tích"
- legal_term: khái niệm / thuật ngữ pháp lý
- penalty: chế tài như "phạt tiền", "phạt tù", "truy cứu trách nhiệm hình sự"
- remedy: biện pháp như "cấm tiếp xúc", "xin lỗi công khai", "giáo dục tại xã, phường"
- document_ref: văn bản được viện dẫn
- article_ref / clause_ref / point_ref: điều khoản viện dẫn
- date_ref: ngày tháng năm
- legal_effect: hiệu lực / bãi bỏ / sửa đổi / bổ sung / thay thế / trách nhiệm thi hành

Yêu cầu:
- Không trùng lặp.
- Nếu không có thì trả {"entities": []}.
""".strip()

RELATION_SYSTEM_PROMPT = """
Bạn là bộ trích xuất quan hệ pháp lý cho GraphRAG từ văn bản pháp luật Việt Nam.

Mục tiêu:
- Từ một đoạn luật, rút ra các quan hệ ngữ nghĩa pháp lý hữu ích cho việc duyệt đồ thị.
- Không suy diễn vượt ra ngoài văn bản đầu vào.
- Trả về JSON hợp lệ, KHÔNG markdown.

Schema bắt buộc:
{
  "relations": [
    {
      "relation_type": "MENTIONS | REGULATES | PENALIZED_BY | MAY_LEAD_TO | REQUIRES | GRANTS_RIGHT | PROHIBITS | REFERS_TO | AMENDS | REPEALS | EFFECTIVE_FROM | RESPONSIBLE_FOR | APPLIES_TO",
      "source_text": "cụm nguồn ngắn",
      "target_text": "cụm đích ngắn"
    }
  ]
}

Gợi ý:
- "hành vi ... bị phạt tiền" => PENALIZED_BY
- "có thể bị truy cứu trách nhiệm hình sự" => MAY_LEAD_TO
- "có hiệu lực từ ngày ..." => EFFECTIVE_FROM
- "chịu trách nhiệm thi hành ..." => RESPONSIBLE_FOR
- "đối tượng áp dụng ..." => APPLIES_TO
- "sửa đổi / bổ sung / thay thế ..." => AMENDS
- "bãi bỏ ..." => REPEALS
- "viện dẫn đến Luật / Nghị định / Điều / Khoản / Điểm ..." => REFERS_TO

Yêu cầu:
- source_text và target_text phải ngắn, lấy từ input.
- Nếu target không rõ thì có thể để chuỗi rỗng.
- Nếu không có thì trả {"relations": []}.
""".strip()


# ============================================================
# JSON HELPERS
# ============================================================
def _empty_entity_payload() -> Dict[str, Any]:
    return {"entities": []}


def _empty_relation_payload() -> Dict[str, Any]:
    return {"relations": []}


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

def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_text(x: Any) -> str:
    """
    Ensure the embedding input is a JSON-serializable plain str.
    Decodes bytes, converts Path/other types, and strips nulls.
    """
    if x is None:
        return ""
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    if not isinstance(x, str):
        x = str(x)
    return x.replace("\x00", "")


def _unique_dicts(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        sig = tuple(_norm_space(str(item.get(k,"")).lower()) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out

# ============================================================
# NORMALIZATION
# ============================================================
ENTITY_TYPE_MAP = {
    "actor": "actor",
    "behavior": "behavior",
    "legal_term": "legal_term",
    "penalty": "penalty",
    "remedy": "remedy",
    "document_ref": "document_ref",
    "article_ref": "article_ref",
    "clause_ref": "clause_ref",
    "point_ref": "point_ref",
    "date_ref": "date_ref",
    "legal_effect": "legal_effect",
}

RELATION_TYPE_MAP = {
    "MENTIONS": "MENTIONS",
    "REGULATES": "REGULATES",
    "PENALIZED_BY": "PENALIZED_BY",
    "MAY_LEAD_TO": "MAY_LEAD_TO",
    "REQUIRES": "REQUIRES",
    "GRANTS_RIGHT": "GRANTS_RIGHT",
    "PROHIBITS": "PROHIBITS",
    "REFERS_TO": "REFERS_TO",
    "AMENDS": "AMENDS",
    "REPEALS": "REPEALS",
    "EFFECTIVE_FROM": "EFFECTIVE_FROM",
    "RESPONSIBLE_FOR": "RESPONSIBLE_FOR",
    "APPLIES_TO": "APPLIES_TO",
}


def _canonicalize_entity_type(entity_type: str) -> Optional[str]:
    key = _norm_space(entity_type).lower()
    return ENTITY_TYPE_MAP.get(key)


def _canonicalize_relation_type(relation_type: str) -> Optional[str]:
    key = _norm_space(relation_type).upper()
    return RELATION_TYPE_MAP.get(key)


def _entity_node_id(entity_type: str, entity_value: str) -> str:
    return f"entity::{entity_type}::{_norm_space(entity_value).lower()}"

def _guess_legal_role(
    node: LegalNode,
    node_entities: List[Dict[str, Any]],
    node_relations: List[Dict[str, Any]],
) -> Optional[str]:
    joined = " ".join(
        [_norm_space(node.text).lower()]
        + [_norm_space(x.get("entity_value", "")).lower() for x in node_entities]
        + [_norm_space(x.get("source_text", "")).lower() for x in node_relations]
        + [_norm_space(x.get("target_text", "")).lower() for x in node_relations]
    )

    if "hiệu lực" in joined:
        return "effective_date"
    if "trách nhiệm thi hành" in joined or "chịu trách nhiệm thi hành" in joined:
        return "execution_responsibility"
    if "đối tượng áp dụng" in joined:
        return "subjects"
    if "phạm vi điều chỉnh" in joined:
        return "scope"
    if "chuyển tiếp" in joined:
        return "transitional"
    if "bãi bỏ" in joined:
        return "repeal"
    if node.action or "sửa đổi" in joined or "bổ sung" in joined or "thay thế" in joined:
        return "amendment"
    return None

# ============================================================
# NODE SELECTION
# ============================================================

def should_extract_with_llm(node: LegalNode) -> bool:
    """
    Chỉ gọi LLM cho các node pháp lý có giá trị truy xuất cao.
    """
    return node.node_type in {"article", "clause", "point", "amendment"}

def _build_node_context(node: LegalNode) -> str:
    return "\n".join(
        [
            f"node_type: {node.node_type}",
            f"article: {node.article or ''}",
            f"clause: {node.clause or ''}",
            f"point: {node.point or ''}",
            f"action: {node.action or ''}",
            f"target_article: {node.target_article or ''}",
            f"target_clause: {node.target_clause or ''}",
            f"target_point: {node.target_point or ''}",
            "",
            "text:",
            node.text or "",
        ]
    ).strip()


def _sanitize_for_llm(value: Any) -> str:
    """
    Sanitize text before sending to ChatCompletions to avoid invalid JSON payloads.
    Removes null bytes / control chars and drops invalid unicode surrogates.
    """
    text = safe_text(value)
    text = "".join(ch if (ch == "\n" or ch == "\t" or ord(ch) >= 32) else " " for ch in text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return _norm_space(text)


def _invoke_llm_safe(system_prompt: str, prompt: str, *, node_id: str) -> str:
    """
    Call LLM in fail-open mode:
    - sanitize payload
    - retry with truncated prompt if upstream rejects payload
    - return empty JSON-ish content on failure so pipeline can continue
    """
    llm = get_llm()
    clean_prompt = _sanitize_for_llm(prompt)
    candidates: List[str] = [clean_prompt]
    if clean_prompt:
        candidates.extend(
            [
                clean_prompt[:12000],
                clean_prompt[:8000],
                clean_prompt[:4000],
            ]
        )

    seen = set()
    for prompt_candidate in candidates:
        if not prompt_candidate or prompt_candidate in seen:
            continue
        seen.add(prompt_candidate)
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_sanitize_for_llm(system_prompt)),
                    HumanMessage(content=prompt_candidate),
                ]
            )
            return safe_text(getattr(response, "content", ""))
        except Exception as exc:
            logger.warning(
                "LLM extraction failed for node_id=%s (prompt_len=%s): %s",
                node_id,
                len(prompt_candidate),
                exc,
            )

    logger.error("Skipping LLM extraction for node_id=%s after retries.", node_id)
    return ""

# ============================================================
# LLM EXTRACTION
# ============================================================

def extract_entities_with_llm(node: LegalNode) -> List[Dict[str, Any]]:
    prompt = _build_node_context(node)
    raw_content = _invoke_llm_safe(ENTITY_SYSTEM_PROMPT, prompt, node_id=node.node_id)
    parsed = _safe_json_loads(raw_content, _empty_entity_payload())

    entities: List[Dict[str, Any]] = []
    for item in parsed.get("entities", []):
        entity_type = _canonicalize_entity_type(str(item.get("entity_type", "")))
        entity_value = _norm_space(str(item.get("entity_value", "")))

        if not entity_type or not entity_value:
            continue

        entities.append(
            {
                "entity_type": entity_type,
                "entity_value": entity_value,
                "node_id": node.node_id,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
            }
        )

    # Bổ sung entity từ metadata parser cho amendment/reference
    if node.target_article:
        entities.append(
            {
                "entity_type": "article_ref",
                "entity_value": node.target_article,
                "node_id": node.node_id,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
            }
        )
    if node.target_clause:
        entities.append(
            {
                "entity_type": "clause_ref",
                "entity_value": node.target_clause,
                "node_id": node.node_id,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
            }
        )
    if node.target_point:
        entities.append(
            {
                "entity_type": "point_ref",
                "entity_value": node.target_point,
                "node_id": node.node_id,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
            }
        )
    if node.action:
        entities.append(
            {
                "entity_type": "legal_effect",
                "entity_value": node.action,
                "node_id": node.node_id,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
            }
        )

    return _unique_dicts(entities, ["entity_type", "entity_value", "node_id"])

def extract_relations_with_llm(node: LegalNode) -> List[Dict[str, Any]]:
    prompt = _build_node_context(node)
    raw_content = _invoke_llm_safe(RELATION_SYSTEM_PROMPT, prompt, node_id=node.node_id)
    parsed = _safe_json_loads(raw_content, _empty_relation_payload())

    relations: List[Dict[str, Any]] = []
    for item in parsed.get("relations", []):
        relation_type = _canonicalize_relation_type(str(item.get("relation_type", "")))
        source_text = _norm_space(str(item.get("source_text", "")))
        target_text = _norm_space(str(item.get("target_text", "")))

        if not relation_type:
            continue

        relations.append(
            {
                "source_id": node.node_id,
                "target_id": None,
                "relation_type": relation_type,
                "source_text": source_text,
                "target_text": target_text,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
                "target_article": node.target_article,
                "target_clause": node.target_clause,
                "target_point": node.target_point,
            }
        )

    # Quan hệ amendment từ parser luôn được giữ
    if node.node_type == "amendment" or node.action:
        relations.append(
            {
                "source_id": node.node_id,
                "target_id": None,
                "relation_type": "AMENDS",
                "source_text": _norm_space(node.action or ""),
                "target_text": _norm_space(
                    " ".join(
                        part
                        for part in [node.target_point, node.target_clause, node.target_article]
                        if part
                    )
                ),
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
                "target_article": node.target_article,
                "target_clause": node.target_clause,
                "target_point": node.target_point,
            }
        )

    return _unique_dicts(
        relations,
        [
            "source_id",
            "relation_type",
            "source_text",
            "target_text",
            "target_article",
            "target_clause",
            "target_point",
        ],
    )


# ============================================================
# GRAPH BUILDERS
# ============================================================

def build_graph_nodes(
    parsed_nodes: List[LegalNode],
    entities_by_node: Dict[str, List[Dict[str, Any]]],
    relations_by_node: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for node in parsed_nodes:
        node_entities = entities_by_node.get(node.node_id, [])
        node_relations = relations_by_node.get(node.node_id, [])
        legal_role = _guess_legal_role(node, node_entities, node_relations)

        out.append(
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "text": node.text,
                "metadata": {
                    "artifact_type": "evidence",
                    "node_type": node.node_type,
                    "article": node.article,
                    "clause": node.clause,
                    "point": node.point,
                    "parent_id": node.parent_id,
                    "action": node.action,
                    "target_article": node.target_article,
                    "target_clause": node.target_clause,
                    "target_point": node.target_point,
                    "legal_role": legal_role,
                },
            }
        )

    return out

def build_entity_nodes(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup = _unique_dicts(entities, ["entity_type", "entity_value"])
    out: List[Dict[str, Any]] = []

    for ent in dedup:
        out.append(
            {
                "node_id": _entity_node_id(ent["entity_type"], ent["entity_value"]),
                "node_type": "entity",
                "text": ent["entity_value"],
                "metadata": {
                    "artifact_type": "entity",
                    "entity_type": ent["entity_type"],
                    "entity_value": ent["entity_value"],
                },
            }
        )

    return out

def _entity_index(
    entities: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], str]:
    idx: Dict[Tuple[str, str], str] = {}
    for ent in entities:
        key = (
            ent["entity_type"],
            _norm_space(ent["entity_value"]).lower(),
        )
        idx[key] = _entity_node_id(ent["entity_type"], ent["entity_value"])
    return idx

def _resolve_entity_target_id(
    target_text: str,
    entities_for_node: List[Dict[str, Any]],
    global_entity_idx: Dict[Tuple[str, str], str],
) -> Optional[str]:
    target_norm = _norm_space(target_text).lower()
    if not target_norm:
        return None

    # Ưu tiên match trong cùng node
    for ent in entities_for_node:
        if _norm_space(ent["entity_value"]).lower() == target_norm:
            return _entity_node_id(ent["entity_type"], ent["entity_value"])

    # Match toàn cục
    for (etype, evalue), eid in global_entity_idx.items():
        if evalue == target_norm:
            return eid

    return None


def build_graph_edges(
    parsed_nodes: List[LegalNode],
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    entities_by_node: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []

    # 1) Structural edges
    for node in parsed_nodes:
        if node.parent_id:
            edges.append(
                {
                    "source_id": node.parent_id,
                    "target_id": node.node_id,
                    "relation_type": "HAS_CHILD",
                }
            )

    # 2) Node -> Entity mention edges
    for ent in entities:
        edges.append(
            {
                "source_id": ent["node_id"],
                "target_id": _entity_node_id(ent["entity_type"], ent["entity_value"]),
                "relation_type": "MENTIONS_ENTITY",
            }
        )

    # 3) Semantic/legal edges from LLM
    entity_idx = _entity_index(entities)
    for rel in relations:
        source_id = rel["source_id"]
        target_id = _resolve_entity_target_id(
            rel.get("target_text", ""),
            entities_by_node.get(source_id, []),
            entity_idx,
        )

        edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": rel["relation_type"],
                "source_text": rel.get("source_text", ""),
                "target_text": rel.get("target_text", ""),
                "target_article": rel.get("target_article"),
                "target_clause": rel.get("target_clause"),
                "target_point": rel.get("target_point"),
            }
        )

    return _unique_dicts(
        edges,
        [
            "source_id",
            "target_id",
            "relation_type",
            "source_text",
            "target_text",
            "target_article",
            "target_clause",
            "target_point",
        ],
    )

def build_article_summaries(parsed_nodes: List[LegalNode]) -> List[Dict[str, Any]]:
    """
    Placeholder summary tầng article để chuẩn bị cho bước hierarchical_summary.py.
    """
    by_article: Dict[str, List[LegalNode]] = {}

    for node in parsed_nodes:
        if not node.article:
            continue
        by_article.setdefault(node.article, []).append(node)

    summaries: List[Dict[str, Any]] = []
    for article, nodes in by_article.items():
        parts: List[str] = []
        for n in nodes:
            if n.node_type in {"article", "clause", "point", "amendment"}:
                parts.append(_norm_space(n.text))

        summary_text = " ".join(parts[:6]).strip()
        if len(summary_text) > 1500:
            summary_text = summary_text[:1500].rstrip() + "..."

        summaries.append(
            {
                "node_id": f"summary::{article}",
                "node_type": "summary",
                "text": summary_text,
                "metadata": {
                    "artifact_type": "summary",
                    "summary_level": "article",
                    "article": article,
                },
            }
        )

    return summaries

# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def ingest_document(text: str) -> Dict[str, Any]:
    """
    Pipeline:
    parse legal tree
    -> LLM entity extraction
    -> LLM relation extraction
    -> build graph artifacts
    -> build evidence chunks for vector index
    """
    parsed_nodes = parse_legal_text(text)

    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []

    entities_by_node: Dict[str, List[Dict[str, Any]]] = {}
    relations_by_node: Dict[str, List[Dict[str, Any]]] = {}

    for node in parsed_nodes:
        if not should_extract_with_llm(node):
            entities_by_node[node.node_id] = []
            relations_by_node[node.node_id] = []
            continue

        try:
            node_entities = extract_entities_with_llm(node)
            node_relations = extract_relations_with_llm(node)
        except Exception as exc:
            logger.exception(
                "Unhandled ingestion error for node_id=%s. Skip node and continue. Error: %s",
                node.node_id,
                exc,
            )
            node_entities = []
            node_relations = []

        entities_by_node[node.node_id] = node_entities
        relations_by_node[node.node_id] = node_relations

        all_entities.extend(node_entities)
        all_relations.extend(node_relations)

    all_entities = _unique_dicts(all_entities, ["entity_type", "entity_value", "node_id"])
    all_relations = _unique_dicts(
        all_relations,
        [
            "source_id",
            "relation_type",
            "source_text",
            "target_text",
            "target_article",
            "target_clause",
            "target_point",
        ],
    )

    graph_nodes = build_graph_nodes(parsed_nodes, entities_by_node, relations_by_node)
    entity_nodes = build_entity_nodes(all_entities)
    graph_edges = build_graph_edges(parsed_nodes, all_entities, all_relations, entities_by_node)
    # Summary nodes will be generated later by hierarchical_summary.py
    summary_nodes: List[Dict[str, Any]] = []
    chunks = build_chunks(parsed_nodes)

    return {
        "parsed_nodes": parsed_nodes,
        "chunks": chunks,
        "graph_nodes": graph_nodes,
        "entity_nodes": entity_nodes,
        "summary_nodes": summary_nodes,
        "entities": all_entities,
        "relations": all_relations,
        "graph_edges": graph_edges,
    }


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    *,
    embeddings: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Embed và upsert các chunk vào Qdrant.
    - chunks: [{"text": "...", "metadata": {...}}, ...]
    - meta: metadata mặc định ở mức document, merge vào metadata chunk.
    """
    if not chunks:
        return 0

    points: List[qm.PointStruct] = []
    base_meta = dict(meta or {})

    def _sanitize_embedding_text(value: str) -> str:
        # Tránh ký tự control / surrogate gây lỗi encode JSON ở HTTP client.
        cleaned = safe_text(value).replace("\x00", " ")
        cleaned = "".join(ch if (ch == "\n" or ch == "\t" or ord(ch) >= 32) else " " for ch in cleaned)
        cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")
        return _norm_space(cleaned)

    def _embed_safe(value: str, *, chunk_id: str) -> Optional[List[float]]:
        candidates = [
            _norm_space(value),
            _sanitize_embedding_text(value),
        ]
        # Retry cuối cùng: truncate để tránh payload quá lớn / ký tự lạ cuối chuỗi.
        if candidates[-1]:
            candidates.append(candidates[-1][:7000])

        tried = set()
        for text_candidate in candidates:
            key = text_candidate
            if not key or key in tried:
                continue
            tried.add(key)
            try:
                return embeddings.embed_query(text_candidate)
            except Exception as exc:
                logger.warning(
                    "Embedding failed for chunk_id=%s (len=%s): %s",
                    chunk_id,
                    len(text_candidate),
                    exc,
                )
        return None

    for idx, chunk in enumerate(chunks):
        text = _norm_space(safe_text(chunk.get("text")))
        if not text:
            continue

        chunk_meta = dict(chunk.get("metadata") or {})
        merged_meta = {**base_meta, **chunk_meta}

        chunk_id = str(merged_meta.get("chunk_id") or merged_meta.get("node_id") or f"chunk_{idx}")
        merged_meta["chunk_id"] = chunk_id

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

        vector = _embed_safe(text, chunk_id=chunk_id)
        if not vector:
            logger.error("Skip chunk due to embedding failure: chunk_id=%s", chunk_id)
            continue
        payload = {
            "text": text,
            "metadata": merged_meta,
            **merged_meta,
        }

        points.append(
            qm.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    if not points:
        return 0

    client = get_qdrant_client()
    upsert_points(client, points)
    return len(points)
