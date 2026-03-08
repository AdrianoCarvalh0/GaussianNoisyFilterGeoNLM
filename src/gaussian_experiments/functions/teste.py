from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment
from skimage.io import imread

from Utils import load_pickle, save_pickle


RESULTS_DIR = Path("/workspace/data/output/set12/dataset_experiment/results")
EXPERIMENT_FILE = "regional_experiment_dataset.pkl"
SOURCE_IMAGE = Path("/workspace/data/input/set12/05.png")
EXPORTS_DIR = RESULTS_DIR / "exports"
IMAGES_DIR = EXPORTS_DIR / "images"
GRAPHS_DIR = EXPORTS_DIR / "graphs"
TABLES_DIR = EXPORTS_DIR / "tables"
REGIONS_DIR = EXPORTS_DIR / "regions"
PADDING = 50
RECT_HALF_SIZE = 15


def load_experiment() -> dict:
    return load_pickle("experiment", RESULTS_DIR / EXPERIMENT_FILE)


def ensure_dirs() -> None:
    for path in (EXPORTS_DIR, IMAGES_DIR, GRAPHS_DIR, TABLES_DIR, REGIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def plota_grafo(G, centro, dimensions=2, layout='spring'):
    if dimensions == 3:
        if layout == 'umap':
            pos = nx.get_node_attributes(G, 'pos_3d')
        else:
            pos = nx.spring_layout(G, dim=3)

        fig = plt.figure(figsize=(18, 16))
        ax = fig.add_subplot(111, projection='3d')

        for edge in G.edges():
            x = np.array([pos[edge[0]][0], pos[edge[1]][0]])
            y = np.array([pos[edge[0]][1], pos[edge[1]][1]])
            z = np.array([pos[edge[0]][2], pos[edge[1]][2]])
            ax.plot(x, y, z, c='gray', alpha=0.7, linewidth=0.1)

        node_xyz = np.array([pos[v] for v in G.nodes()])
        ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=35, c='black', alpha=0.7)

        if centro is not None and centro in G.nodes:
            ax.scatter(pos[centro][0], pos[centro][1], pos[centro][2], s=35, c='red', alpha=0.7)

    else:  # 2D plotting
        if layout == 'kamada':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=20)

        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx(
            G,
            pos,
            ax=ax,
            node_size=30,
            with_labels=False,
            width=0.1,
            edge_color='gray',
            alpha=0.7,
        )
        if centro is not None and centro in G.nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_size=30,
                nodelist=[centro],
                node_color='red',
                alpha=0.7,
            )
        ax.axis("off")

    return fig


def save_region_overview(tipo: str, entries: list[dict]) -> Path:
    image = imread(SOURCE_IMAGE, as_gray=True)
    image = (255 * image).astype(np.uint8)
    image = np.pad(image, ((PADDING, PADDING), (PADDING, PADDING)), mode='symmetric')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    output_path = REGIONS_DIR / f"{tipo}_regions.pdf"
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_rgb)

    for entry in entries:
        cx, cy = entry["center"]

        cv2.rectangle(
            image_rgb,
            (cx - RECT_HALF_SIZE, cy - RECT_HALF_SIZE),
            (cx + RECT_HALF_SIZE, cy + RECT_HALF_SIZE),
            (0, 255, 0),
            3,
        )
        cv2.circle(image_rgb, (cx, cy), 5, (255, 0, 0), -1)

    ax.clear()
    ax.imshow(image_rgb)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(output_path, dpi=300, pad_inches=0.2)
    plt.close(fig)
    return output_path


def build_graph_positions(graph: nx.Graph, embedding: np.ndarray | None) -> dict:
    nodes = list(graph.nodes())
    if embedding is not None and len(nodes) == len(embedding):
        return {
            node: (float(embedding[idx][0]), float(embedding[idx][1]))
            for idx, node in enumerate(nodes)
        }
    return nx.spring_layout(graph, seed=42)


def save_image_panel(entry: dict) -> Path:
    image_path = IMAGES_DIR / f"{entry['tipo']}_{entry['index']:02d}_images.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    panels = [
        entry["clean"],
        entry["noisy"][5],
        entry["noisy"][50],
    ]

    for ax, image in zip(axes, panels):
        ax.imshow(image, cmap="gray")
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03)
    fig.savefig(image_path, dpi=300, pad_inches=0.2)
    plt.close(fig)
    return image_path


def save_graph_figure(entry: dict, sigma: int) -> Path:
    graph = entry["graphs"][sigma]
    graph_path = GRAPHS_DIR / f"{entry['tipo']}_{entry['index']:02d}_graph_sigma_{sigma}.pdf"

    fig = plota_grafo(graph, entry["patch_center_index"], dimensions=2, layout="spring")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(graph_path, dpi=300, pad_inches=0.3)
    plt.close(fig)
    return graph_path


def build_assets_table(experiment: dict) -> pd.DataFrame:
    rows = []

    for tipo, entries in experiment.items():
        region_overview_path = save_region_overview(tipo, entries)

        for entry in entries:
            image_panel_path = save_image_panel(entry)
            graph_5_path = save_graph_figure(entry, 5)
            graph_50_path = save_graph_figure(entry, 50)

            rows.append(
                {
                    "tipo": entry["tipo"],
                    "index": entry["index"],
                    "center": entry["center"],
                    "parameters": entry["parameters"],
                    "region_overview_path": str(region_overview_path),
                    "image_panel_path": str(image_panel_path),
                    "graph_5_path": str(graph_5_path),
                    "graph_50_path": str(graph_50_path),
                    "embeddings_5": entry["embeddings"].get(5),
                    "embeddings_50": entry["embeddings"].get(50),
                }
            )

    return pd.DataFrame(rows)


def build_metrics_table(experiment: dict) -> pd.DataFrame:
    rows = []

    for entries in experiment.values():
        for entry in entries:
            for sigma in (5, 50):
                patch_metrics = entry["patch_metrics"][sigma]
                graph_metrics = entry["graph_metrics"][sigma]
                graph = entry["graphs"][sigma]
                embedding = entry["embeddings"].get(sigma)

                rows.append(
                    {
                        "tipo": entry["tipo"],
                        "index": entry["index"],
                        "center": entry["center"],
                        "sigma": sigma,
                        "patch_center_index": entry["patch_center_index"],
                        "patch_rank": patch_metrics["rank"],
                        "patch_energy": patch_metrics["energy"],
                        "patch_spectral_ratio": patch_metrics["spectral_ratio"],
                        "graph_clustering": graph_metrics["clustering"],
                        "graph_density": graph_metrics["density"],
                        "graph_num_nodes": graph.number_of_nodes(),
                        "graph_num_edges": graph.number_of_edges(),
                        "embedding_dim": (
                            int(embedding.shape[1]) if embedding is not None else None
                        ),
                        "embedding_num_points": (
                            int(embedding.shape[0]) if embedding is not None else None
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save_dataframe_as_xlsx(dataframe: pd.DataFrame, output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="data")
        worksheet = writer.sheets["data"]

        float_columns = {
            "patch_energy",
            "patch_spectral_ratio",
            "graph_clustering",
            "graph_density",
        }

        for column_index, column_name in enumerate(dataframe.columns, start=1):
            worksheet.cell(row=1, column=column_index).alignment = Alignment(
                horizontal="center"
            )

            if column_name in float_columns:
                for row_index in range(2, len(dataframe) + 2):
                    worksheet.cell(row=row_index, column=column_index).number_format = (
                        "0.000000000000000"
                    )


def save_tables(assets_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    assets_pickle = TABLES_DIR / "experiment_assets.pkl"
    metrics_pickle = TABLES_DIR / "experiment_metrics.pkl"
    assets_xlsx = TABLES_DIR / "experiment_assets.xlsx"
    metrics_xlsx = TABLES_DIR / "experiment_metrics.xlsx"

    save_pickle(assets_df, TABLES_DIR, assets_pickle.name)
    save_pickle(metrics_df, TABLES_DIR, metrics_pickle.name)
    save_dataframe_as_xlsx(assets_df, assets_xlsx)
    save_dataframe_as_xlsx(metrics_df, metrics_xlsx)

    print(assets_pickle)
    print(metrics_pickle)
    print(assets_xlsx)
    print(metrics_xlsx)


def main() -> None:
    ensure_dirs()
    experiment = load_experiment()
    assets_df = build_assets_table(experiment)
    metrics_df = build_metrics_table(experiment)
    save_tables(assets_df, metrics_df)

    print("\nAssets preview:")
    print(assets_df[["tipo", "index", "image_panel_path", "graph_5_path"]].head())

    print("\nMetrics preview:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
