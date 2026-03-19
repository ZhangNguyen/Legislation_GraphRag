from __future__ import annotations

import argparse
import json
import logging
import os

# Giảm log nhiễu từ grpc/absl nếu có trong dependency tree.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.app.runtime import build_runtime_graph, get_runtime_status


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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
