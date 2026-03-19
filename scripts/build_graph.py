from __future__ import annotations

import argparse
import json

from src.app.runtime import build_runtime_graph, get_runtime_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime graph before starting server.")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--glob", type=str, default=None)
    args = parser.parse_args()

    graph = build_runtime_graph(input_dir=args.input_dir, glob_pattern=args.glob)
    status = get_runtime_status()

    print(
        json.dumps(
            {
                "graph_nodes": len(graph.get("nodes", [])),
                "graph_edges": len(graph.get("edges", [])),
                "status": status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
