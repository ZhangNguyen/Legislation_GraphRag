from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


# =========================
# Core utilities
# =========================

def _unique_dicts(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for item in items:
        sig = tuple(str(item.get(k, "")).strip().lower() for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)

    return out


def _index_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(n["node_id"]): n
        for n in nodes
        if n.get("node_id")
    }


def _build_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for e in edges:
        source_id = e.get("source_id")
        if source_id:
            adj[str(source_id)].append(e)

    return dict(adj)


def _build_reverse_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rev: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for e in edges:
        target_id = e.get("target_id")
        if target_id:
            rev[str(target_id)].append(e)

    return dict(rev)


def refresh_graph_indexes(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild node_index / adjacency / reverse_adjacency
    after nodes or edges are updated.
    """
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))

    graph["node_index"] = _index_nodes(nodes)
    graph["adjacency"] = _build_adjacency(edges)
    graph["reverse_adjacency"] = _build_reverse_adjacency(edges)
    return graph


# =========================
# Build graph
# =========================

def build_graph(
    graph_nodes: List[Dict[str, Any]],
    entity_nodes: Optional[List[Dict[str, Any]]] = None,
    summary_nodes: Optional[List[Dict[str, Any]]] = None,
    graph_edges: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Gom toàn bộ artifacts thành graph object chuẩn cho GraphRAG.

    Input:
    - graph_nodes: legal evidence nodes (article / clause / point / amendment / effective / ...)
    - entity_nodes: semantic/entity nodes
    - summary_nodes: summary nodes (có thể rỗng ở giai đoạn graph thô)
    - graph_edges: edges giữa các node

    Output:
    {
        "nodes": [...],
        "edges": [...],
        "node_index": {...},
        "adjacency": {...},
        "reverse_adjacency": {...}
    }
    """
    entity_nodes = entity_nodes or []
    summary_nodes = summary_nodes or []
    graph_edges = graph_edges or []

    all_nodes = list(graph_nodes) + list(entity_nodes) + list(summary_nodes)
    all_nodes = _unique_dicts(all_nodes, ["node_id"])

    graph_edges = _unique_dicts(
        graph_edges,
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

    graph = {
        "nodes": all_nodes,
        "edges": graph_edges,
    }
    graph = refresh_graph_indexes(graph)
    return graph


# =========================
# Entity co-occurrence edges
# =========================

def attach_entity_cooccurrence_edges(
    graph: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Nếu trong cùng 1 legal node có nhiều entities,
    tạo edge RELATED_IN_CONTEXT giữa các entity nodes.
    """
    edges = list(graph.get("edges", []))

    entities_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ent in entities:
        node_id = ent.get("node_id")
        if node_id:
            entities_by_source[str(node_id)].append(ent)

    for source_node_id, ents in entities_by_source.items():
        entity_ids: List[str] = []

        for ent in ents:
            entity_type = ent.get("entity_type")
            entity_value = ent.get("entity_value")
            if entity_type and entity_value:
                entity_ids.append(
                    f"entity::{entity_type}::{str(entity_value).strip().lower()}"
                )

        entity_ids = sorted(set(entity_ids))

        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                e1 = entity_ids[i]
                e2 = entity_ids[j]

                edges.append(
                    {
                        "source_id": e1,
                        "target_id": e2,
                        "relation_type": "RELATED_IN_CONTEXT",
                    }
                )
                edges.append(
                    {
                        "source_id": e2,
                        "target_id": e1,
                        "relation_type": "RELATED_IN_CONTEXT",
                    }
                )

    graph["edges"] = _unique_dicts(
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
    return refresh_graph_indexes(graph)


# =========================
# Main helper: ingestion -> graph
# =========================

def materialize_graph_from_ingestion(ingestion_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper chính để nối ingestion -> graph object.

    ingestion_output kỳ vọng có:
    {
        "graph_nodes": [...],
        "entity_nodes": [...],
        "summary_nodes": [...],   # có thể rỗng
        "graph_edges": [...],
        "entities": [...]
    }

    Lưu ý:
    - graph_builder chỉ materialize graph thô
    - summary edges sẽ do hierachical_summary.py gắn sau
    """
    graph = build_graph(
        graph_nodes=ingestion_output.get("graph_nodes", []),
        entity_nodes=ingestion_output.get("entity_nodes", []),
        summary_nodes=ingestion_output.get("summary_nodes", []),
        graph_edges=ingestion_output.get("graph_edges", []),
    )

    graph = attach_entity_cooccurrence_edges(
        graph=graph,
        entities=ingestion_output.get("entities", []),
    )

    return graph


# =========================
# Neighbor access
# =========================

def get_neighbors(
    graph: Dict[str, Any],
    node_id: str,
    relation_types: Optional[List[str]] = None,
    direction: str = "out",
) -> List[Dict[str, Any]]:
    """
    Lấy neighbors của 1 node.

    direction:
    - "out": source_id -> target_id
    - "in":  target_id <- source_id
    """
    relation_types = relation_types or []

    if direction == "in":
        candidate_edges = graph.get("reverse_adjacency", {}).get(node_id, [])
    else:
        candidate_edges = graph.get("adjacency", {}).get(node_id, [])

    if not relation_types:
        return candidate_edges

    allowed = {x.upper() for x in relation_types}
    return [
        e for e in candidate_edges
        if str(e.get("relation_type", "")).upper() in allowed
    ]