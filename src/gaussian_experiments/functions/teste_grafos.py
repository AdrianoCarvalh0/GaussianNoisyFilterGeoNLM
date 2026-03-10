from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from Utils import load_pickle


RESULTS_DIR = Path("/workspace/data/output/set12/dataset_experiment/results")
EXPERIMENT_FILE = "regional_experiment_dataset.pkl"
EXPORTS_DIR = RESULTS_DIR / "exports"
GRAPHS_DIR = EXPORTS_DIR / "graphs_single"

BACKGROUND_COLOR = "#ffffff"
EDGE_COLOR = "#9a9a9a"
NODE_COLOR = "#4169E1"
CENTER_NODE_COLOR = "#f10808"
TYPE_PREFIX = {
    "uniforme": "uniform",
    "nao_uniforme": "non_uniform",
    "mesclada": "mixed",
}


def load_experiment() -> dict:
    return load_pickle("experiment", RESULTS_DIR / EXPERIMENT_FILE)


def ensure_dirs() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def build_graph_positions(graph: nx.Graph, embedding: np.ndarray | None) -> dict:
    nodes = list(graph.nodes())
    if embedding is not None and len(nodes) == len(embedding) and embedding.shape[1] >= 2:
        return {
            node: (float(embedding[idx][0]), float(embedding[idx][1]))
            for idx, node in enumerate(nodes)
        }
    return nx.spring_layout(graph, seed=10)


def save_graph(entry: dict, sigma: int) -> Path:
    graph = entry["graphs"][sigma]
    embedding = entry["embeddings"].get(sigma)
    center_node = entry["patch_center_index"]
    positions = build_graph_positions(graph, embedding)
    type_prefix = TYPE_PREFIX.get(entry["tipo"], entry["tipo"])

    output_path = GRAPHS_DIR / f"{type_prefix}_{entry['index']:02d}_sigma_{sigma}.png"

    fig, ax = plt.subplots(figsize=(10,10), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color=EDGE_COLOR,
        width=0.25,
        alpha=0.35,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=NODE_COLOR,
        node_size=80,
        linewidths=0,
        alpha=0.95,
    )

    if center_node in graph.nodes:
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            nodelist=[center_node],
            node_color=CENTER_NODE_COLOR,
            node_size=80,
            linewidths=0,
            alpha=0.95,
        )

    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(output_path, dpi=200, facecolor=BACKGROUND_COLOR, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def export_all_graphs(experiment: dict) -> list[Path]:
    saved_paths = []

    for entries in experiment.values():
        for entry in entries:
            for sigma in sorted(entry["graphs"]):
                saved_paths.append(save_graph(entry, sigma))

    return saved_paths


def main() -> None:
    ensure_dirs()
    experiment = load_experiment()
    saved_paths = export_all_graphs(experiment)

    print(f"Saved {len(saved_paths)} graph images to:")
    print(GRAPHS_DIR)


if __name__ == "__main__":
    main()
