"""Transformer attention and GNNExplainer undirected edge visualization."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.explain import Explainer, GNNExplainer

from train import GNNTF, ConnectivityData, load_data, set_seed


def load_checkpoint(path: Path, device: torch.device) -> tuple[GNNTF, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GNNTF(**checkpoint["model_config"])
    model.load_compact_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


class SubjectExplainerModel(nn.Module):
    """Expose a subject's disjoint window graphs to PyG's explainer API."""

    def __init__(self, model: GNNTF, windows: int, nodes: int) -> None:
        super().__init__()
        self.model = model
        self.windows = windows
        self.nodes = nodes

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, tabular: Tensor
    ) -> Tensor:
        source, target = edge_index
        graph_id = source // self.nodes
        adjacency = x.new_zeros(self.windows, self.nodes, self.nodes)
        adjacency[graph_id, source % self.nodes, target % self.nodes] = edge_weight
        return self.model(adjacency.unsqueeze(0), x[: self.nodes].unsqueeze(0), tabular)


def explain_directed_edges(
    model: GNNTF,
    adjacency: Tensor,
    node_features: Tensor,
    tabular: Tensor,
    steps: int,
    learning_rate: float,
) -> Tensor:
    """Explain all window graphs for one subject using PyG GNNExplainer."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    windows, nodes, _ = adjacency.shape
    graph_id, source, target = adjacency.ne(0).nonzero(as_tuple=True)
    if not graph_id.numel():
        return torch.zeros_like(adjacency).cpu()
    edge_index = torch.stack([graph_id * nodes + source, graph_id * nodes + target])
    edge_weight = adjacency[graph_id, source, target]
    explainer = Explainer(
        model=SubjectExplainerModel(model, windows, nodes).eval(),
        algorithm=GNNExplainer(epochs=steps, lr=learning_rate),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification", task_level="graph", return_type="raw"
        ),
    )
    explanation = explainer(
        node_features.repeat(windows, 1),
        edge_index,
        edge_weight=edge_weight,
        tabular=tabular.unsqueeze(0),
    )
    masks = torch.zeros_like(adjacency)
    masks[graph_id, source, target] = explanation.edge_mask.detach()
    return masks.cpu()


def average_pyg_directions(
    edge_index: Tensor, edge_mask: Tensor
) -> dict[tuple[int, int], float]:
    """Convert PyG directed storage to one mean score per undirected edge.

    For example, ``(u, v)=0.2`` and ``(v, u)=0.8`` become
    ``{min(u,v), max(u,v)}=0.5``. Self-loops are excluded from connectivity
    ranking. This conversion must happen before sums and Top-K selection.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_mask.ndim != 1 or edge_mask.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_mask must have one value per directed edge")

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for edge_number in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_number])
        target = int(edge_index[1, edge_number])
        if source == target:
            continue
        undirected_key = (min(source, target), max(source, target))
        grouped[undirected_key].append(float(edge_mask[edge_number]))
    return {edge: float(np.mean(values)) for edge, values in grouped.items()}


def aggregate_undirected_edges(
    directed_masks: Tensor, adjacency: Tensor
) -> list[dict[str, float | int]]:
    """Average directions first, then aggregate subject-window instances.

    Dividing by every subject-window instance follows the study's aggregation
    definition. Missing edges therefore contribute zero to that denominator.
    """

    if directed_masks.shape != adjacency.shape:
        raise ValueError("directed_masks and adjacency must have the same shape")

    score_sum: dict[tuple[int, int], float] = defaultdict(float)
    presence_count: dict[tuple[int, int], int] = defaultdict(int)
    subjects, windows, _, _ = directed_masks.shape
    denominator = subjects * windows

    for subject in range(subjects):
        for window in range(windows):
            directed_edge_index = adjacency[subject, window].ne(0).nonzero().T
            directed_edge_mask = directed_masks[subject, window][
                directed_edge_index[0], directed_edge_index[1]
            ]
            undirected_window = average_pyg_directions(
                directed_edge_index, directed_edge_mask
            )
            for edge, value in undirected_window.items():
                score_sum[edge] += value
                presence_count[edge] += 1

    rows = [
        {
            "node_a": edge[0],
            "node_b": edge[1],
            "importance": value / denominator,
            "present_instances": presence_count[edge],
        }
        for edge, value in score_sum.items()
    ]
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)


@torch.no_grad()
def collect_readout_attention(
    model: GNNTF, data: ConnectivityData, subject_ids: list[int], device: torch.device
) -> np.ndarray:
    attention_rows = []
    for subject_id in subject_ids:
        model(
            data.adjacency[subject_id : subject_id + 1].to(device),
            data.node_features.unsqueeze(0).to(device),
            data.tabular[subject_id : subject_id + 1].to(device),
            record_attention=True,
        )
        if model.last_attention is None:
            raise RuntimeError("the selected backbone did not return attention")
        attention = model.last_attention.detach().cpu()
        query_index = -1 if model.backbone == "gpt2" else 0
        attention_rows.append(attention[0, :, query_index, :].mean(dim=0).numpy())
    return np.mean(attention_rows, axis=0)


def save_attention_plot(values: np.ndarray, backbone: str, path: Path) -> None:
    if backbone == "gpt2":
        labels = [f"window {index + 1}" for index in range(len(values) - 1)]
        labels.append("tabular")
        title = "Attention from the final tabular token"
    else:
        labels = ["CLS", "tabular"] + [
            f"window {index + 1}" for index in range(len(values) - 2)
        ]
        title = "Attention from the CLS token"

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, values, color="#4472C4")
    axis.set_ylabel("Mean last-layer attention")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_top_edges_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "rank",
                "node_a",
                "node_b",
                "importance",
                "present_instances",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})


def save_edge_plot(
    rows: list[dict[str, float | int]], data: ConnectivityData, path: Path
) -> None:
    if not rows:
        return
    graph = nx.Graph()
    for row in rows:
        graph.add_edge(
            int(row["node_a"]), int(row["node_b"]), weight=row["importance"]
        )

    coordinates = data.node_features[:, :2].numpy()
    positions = {node: coordinates[node] for node in graph.nodes}
    systems = data.system_ids.numpy()
    node_colors = [systems[node] for node in graph.nodes]
    weights = np.array(
        [graph[node_a][node_b]["weight"] for node_a, node_b in graph.edges],
        dtype=float,
    )
    if np.ptp(weights) > 0:
        widths = 0.8 + 4.2 * (weights - weights.min()) / np.ptp(weights)
    else:
        widths = np.full_like(weights, 2.0)

    figure, axis = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        cmap="tab10",
        node_size=430,
        edgecolors="black",
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        width=widths,
        edge_color="#E6A700",
        alpha=0.85,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=8, ax=axis)
    axis.set_title(f"Top {len(rows)} undirected connections")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    data_path = Path(args.data) if args.data else run_dir / "data.npz"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "model.pt"
    data = load_data(data_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    candidate_ids = checkpoint.get("explain_ids", checkpoint["test_ids"])
    subject_ids = list(candidate_ids[: args.subjects])
    output_dir.mkdir(parents=True, exist_ok=True)

    attention = collect_readout_attention(model, data, subject_ids, device)
    save_attention_plot(
        attention, model.backbone, output_dir / "token_attention.png"
    )

    masks = []
    print(f"Explaining {len(subject_ids)} subjects with directed edge masks...")
    for number, subject_id in enumerate(subject_ids, start=1):
        masks.append(
            explain_directed_edges(
                model,
                data.adjacency[subject_id].to(device),
                data.node_features.to(device),
                data.tabular[subject_id].to(device),
                steps=args.explainer_steps,
                learning_rate=args.explainer_lr,
            )
        )
        print(f"  subject {number}/{len(subject_ids)} complete")

    selected_adjacency = data.adjacency[subject_ids]
    all_edges = aggregate_undirected_edges(torch.stack(masks), selected_adjacency)
    top_edges = all_edges[: args.top_k]
    save_top_edges_csv(top_edges, output_dir / "top_undirected_edges.csv")
    save_edge_plot(top_edges, data, output_dir / "top_undirected_edges.png")
    print(f"Saved visualizations to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain and visualize a fitted GNN-TF model.")
    parser.add_argument("--run-dir", default="outputs/scratch")
    parser.add_argument("--data", help="Override the run's data NPZ file.")
    parser.add_argument("--checkpoint", help="Override the run's model checkpoint.")
    parser.add_argument("--output-dir", help="Figure directory (default: <run-dir>/visualizations).")
    parser.add_argument("--subjects", type=int, default=32)
    parser.add_argument("--explainer-steps", type=int, default=500)
    parser.add_argument("--explainer-lr", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
