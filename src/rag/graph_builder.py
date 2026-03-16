from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

def _unique_dicts(items: List[Dict[str,Any]], keys: List[str]) -> List[Dict[str,Any]]:
    seen = set()
    out: List[Dict[str,Any]] = []
    for item in items:
        sig = tuple(str(item.get(k, "")).strip().lower() for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out

def _index_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["node_id"]: n for n in nodes if n.get("node_id")}

def _build_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adj: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
    for e in edges:
        source_id = e.get("source_id")
        if source_id:
            adj[source_id].append(e)
    return dict(adj)

def _build_reverse_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rev: Dict[str,List[Dict[str,Any]]] = defaultdict(list)
    for e in edges:
        target_id = e.get("target_id")
        if target_id:
            rev[target_id].append(e)
    return dict(rev)

def build_graph(
    graph_nodes: List[Dict[str, Any]],
    entity_nodes: List[Dict[str, Any]],
    summary_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Gom toàn bộ artifacts thành 1 graph object chuẩn cho GraphRAG.

    Input:
    - graph_nodes: legal text nodes (article / clause / point / amendment)
    - entity_nodes: semantic/entity nodes
    - summary_nodes: summary nodes (article summary ...)
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
    all_nodes = graph_nodes + entity_nodes + summary_nodes
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

    node_index = _index_nodes(all_nodes)
    adjacency = _build_adjacency(graph_edges)
    reverse_adjacency = _build_reverse_adjacency(graph_edges)

    return {
        "nodes": all_nodes,
        "edges": graph_edges,
        "node_index": node_index,
        "adjacency": adjacency,
        "reverse_adjacency": reverse_adjacency,
    }

def attach_summary_links(
    graph: Dict[str, Any],
    summary_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Gắn edge từ summary node -> legal node theo article.
    Dùng để GraphRAG nhảy từ summary xuống evidence.
    """
    nodes = graph["nodes"]
    edges = list(graph["edges"])

    node_index = graph["node_index"]

    # legal evidence nodes theo article
    evidence_by_article: Dict[str, List[str]] = defaultdict(list)
    for node in nodes:
        md = node.get("metadata", {})
        if md.get("artifact_type") == "evidence":
            article = md.get("article")
            if article:
                evidence_by_article[str(article)].append(node["node_id"])

    for summary in summary_nodes:
        md = summary.get("metadata", {})
        article = md.get("article")
        if not article:
            continue

        for target_id in evidence_by_article.get(str(article), []):
            edges.append(
                {
                    "source_id": summary["node_id"],
                    "target_id": target_id,
                    "relation_type": "SUMMARIZES",
                }
            )

    edges = _unique_dicts(
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

    graph["edges"] = edges
    graph["adjacency"] = _build_adjacency(edges)
    graph["reverse_adjacency"] = _build_reverse_adjacency(edges)
    return graph


def attach_entity_cooccurrence_edges(
    graph: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Nếu trong cùng 1 legal node có nhiều entities, tạo edge RELATED_IN_CONTEXT
    giữa các entity để tăng khả năng graph traversal.
    """
    edges = list(graph["edges"])

    entities_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ent in entities:
        node_id = ent.get("node_id")
        if node_id:
            entities_by_source[node_id].append(ent)

    for source_node_id, ents in entities_by_source.items():
        entity_ids = []
        for ent in ents:
            entity_type = ent.get("entity_type")
            entity_value = ent.get("entity_value")
            if entity_type and entity_value:
                entity_ids.append(f"entity::{entity_type}::{str(entity_value).strip().lower()}")

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

    edges = _unique_dicts(
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

    graph["edges"] = edges
    graph["adjacency"] = _build_adjacency(edges)
    graph["reverse_adjacency"] = _build_reverse_adjacency(edges)
    return graph


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


def materialize_graph_from_ingestion(ingestion_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper chính để nối ingestion -> graph object.

    ingestion_output kỳ vọng có:
    {
        "graph_nodes": [...],
        "entity_nodes": [...],
        "summary_nodes": [...],
        "graph_edges": [...],
        "entities": [...]
    }
    """
    graph = build_graph(
        graph_nodes=ingestion_output.get("graph_nodes", []),
        entity_nodes=ingestion_output.get("entity_nodes", []),
        summary_nodes=ingestion_output.get("summary_nodes", []),
        graph_edges=ingestion_output.get("graph_edges", []),
    )

    graph = attach_summary_links(
        graph=graph,
        summary_nodes=ingestion_output.get("summary_nodes", []),
    )

    graph = attach_entity_cooccurrence_edges(
        graph=graph,
        entities=ingestion_output.get("entities", []),
    )

    return graph