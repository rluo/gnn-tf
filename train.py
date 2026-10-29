"""GNN-TF training and evaluation for dynamic connectivity and tabular data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn import GCNConv


DEFAULT_CHECKPOINTS = {
    "bert": "textattack/bert-base-uncased-yelp-polarity",
    "gpt2": "microsoft/DialogRPT-updown",
    "vit": "google/vit-base-patch16-224",
}


@dataclass(frozen=True)
class ConnectivityData:
    """Subject-level connectivity graphs, ROI attributes, and covariates."""

    adjacency: Tensor  # [subjects, windows, nodes, nodes]
    node_features: Tensor  # [nodes, node_features]
    tabular: Tensor  # [subjects, tabular_features]
    labels: Tensor  # [subjects]
    system_ids: Tensor  # [nodes]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str_to_bool(value: str | bool) -> bool:
    """Parse command-line booleans such as ``--droplast=False``."""

    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def bh_fdr_mask(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return a BH-FDR rejection mask for unique undirected tests."""

    if p_values.ndim != 2 or p_values.shape[0] != p_values.shape[1]:
        raise ValueError("p_values must be a square matrix")

    upper = np.triu_indices(p_values.shape[0], k=1)
    p = p_values[upper]
    order = np.argsort(p)
    ranked_p = p[order]
    critical = alpha * np.arange(1, len(p) + 1) / len(p)
    passed = np.flatnonzero(ranked_p <= critical)

    rejected = np.zeros_like(p, dtype=bool)
    if passed.size:
        rejected[order[: passed[-1] + 1]] = True

    mask = np.zeros_like(p_values, dtype=bool)
    mask[upper] = rejected
    return mask | mask.T


def correlation_graph(window: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Construct A = B * C from one [time, node] sliding window."""

    correlation = np.corrcoef(window, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    denominator = np.maximum(1.0 - correlation**2, np.finfo(float).eps)
    t_statistic = np.abs(correlation) * np.sqrt(
        (window.shape[0] - 2) / denominator
    )
    p_values = 2.0 * stats.t.sf(t_statistic, df=window.shape[0] - 2)
    np.fill_diagonal(p_values, 1.0)

    adjacency = correlation * bh_fdr_mask(p_values, alpha=alpha)
    np.fill_diagonal(adjacency, 0.0)  # the GCN adds self-loops
    return adjacency.astype(np.float32)


def make_synthetic_dataset(
    num_subjects: int = 80,
    num_nodes: int = 24,
    num_systems: int = 4,
    num_windows: int = 8,
    window_size: int = 130,
    step_size: int = 20,
    alpha: float = 0.05,
    seed: int = 7,
) -> ConnectivityData:
    """Generate ROI signals with time-varying coupling and binary outcomes."""

    if num_nodes < num_systems or num_nodes % num_systems:
        raise ValueError("num_nodes must be divisible by num_systems")
    if num_subjects < 20:
        raise ValueError("num_subjects must be at least 20")

    rng = np.random.default_rng(seed)
    total_time = window_size + step_size * (num_windows - 1)
    system_ids = np.repeat(np.arange(num_systems), num_nodes // num_systems)

    coordinates = rng.uniform(-1.0, 1.0, size=(num_nodes, 3))
    one_hot_system = np.eye(num_systems, dtype=np.float32)[system_ids]
    node_features = np.concatenate([coordinates, one_hot_system], axis=1).astype(
        np.float32
    )

    age = rng.uniform(12.0, 21.0, size=num_subjects)
    sex = rng.integers(0, 2, size=num_subjects)
    individual_risk = rng.normal(size=num_subjects)
    risk_score = individual_risk + 0.20 * (age - age.mean()) + 0.35 * (sex - 0.5)
    labels = (risk_score > np.median(risk_score)).astype(np.int64)
    tabular = np.column_stack([age, sex]).astype(np.float32)

    all_graphs = []
    for subject in range(num_subjects):
        system_signal = rng.normal(size=(total_time, num_systems))
        for time_index in range(1, total_time):
            system_signal[time_index] += 0.65 * system_signal[time_index - 1]

        shared_signal = rng.normal(size=total_time)
        time_ramp = np.linspace(0.0, 1.0, total_time)
        dynamic_strength = 0.10 + 0.55 * labels[subject] * time_ramp

        time_series = np.empty((total_time, num_nodes), dtype=np.float32)
        for node in range(num_nodes):
            signal = 0.75 * system_signal[:, system_ids[node]]
            if system_ids[node] in (0, 1):
                signal = signal + dynamic_strength * shared_signal
            time_series[:, node] = signal + rng.normal(scale=0.65, size=total_time)

        subject_graphs = []
        for window_index in range(num_windows):
            start = window_index * step_size
            stop = start + window_size
            subject_graphs.append(correlation_graph(time_series[start:stop], alpha))
        all_graphs.append(subject_graphs)

    return ConnectivityData(
        adjacency=torch.tensor(np.asarray(all_graphs), dtype=torch.float32),
        node_features=torch.tensor(node_features, dtype=torch.float32),
        tabular=torch.tensor(tabular, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        system_ids=torch.tensor(system_ids, dtype=torch.long),
    )


def save_data(data: ConnectivityData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        adjacency=data.adjacency.numpy(),
        node_features=data.node_features.numpy(),
        tabular=data.tabular.numpy(),
        labels=data.labels.numpy(),
        system_ids=data.system_ids.numpy(),
    )


def load_data(path: Path) -> ConnectivityData:
    with np.load(path, allow_pickle=False) as arrays:
        data = ConnectivityData(
            adjacency=torch.tensor(arrays["adjacency"], dtype=torch.float32),
            node_features=torch.tensor(arrays["node_features"], dtype=torch.float32),
            tabular=torch.tensor(arrays["tabular"], dtype=torch.float32),
            labels=torch.tensor(arrays["labels"], dtype=torch.long),
            system_ids=torch.tensor(arrays["system_ids"], dtype=torch.long),
        )
    return data


class DynamicGraphDataset(Dataset):
    def __init__(self, data: ConnectivityData) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.data.adjacency[index],
            self.data.node_features,
            self.data.tabular[index],
            self.data.labels[index],
        )


class TrainingBatchSampler(torch.utils.data.BatchSampler):
    """Merge a singleton final batch so BatchNorm can run in training mode."""

    def __iter__(self):
        batches = list(super().__iter__())
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = super().__len__()
        if not self.drop_last and len(self.sampler) % self.batch_size == 1 and count > 1:
            count -= 1
        return count


class DenseGCNLayer(nn.Module):
    """Apply a PyG GCN to a batch of adjacency matrices."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv = GCNConv(in_features, out_features)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        graphs, nodes, _ = adjacency.shape
        graph_id, source, target = adjacency.ne(0).nonzero(as_tuple=True)
        edge_index = torch.stack([graph_id * nodes + source, graph_id * nodes + target])
        edge_weight = adjacency[graph_id, source, target]
        # Standard GCN normalization requires positive signed weighted degrees.
        if torch.any(adjacency.sum(dim=-1) + 1 <= 0):
            raise ValueError("GCN requires positive weighted degrees after adding self-loops")
        output = self.conv(x.reshape(graphs * nodes, -1), edge_index, edge_weight)
        return output.reshape(graphs, nodes, -1)


class InspectableEncoderLayer(nn.Module):
    """Standard encoder layer that exposes attention without patching PyTorch."""

    def __init__(self, hidden_dim: int, heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.1, batch_first=True
        )
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.last_attention: Tensor | None = None

    def forward(self, x: Tensor, record_attention: bool = False) -> Tensor:
        attention_output, attention = self.self_attention(
            x,
            x,
            x,
            need_weights=record_attention,
            average_attn_weights=False,
        )
        self.last_attention = attention.detach() if record_attention else None
        x = self.norm1(x + self.dropout1(attention_output))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout2(feedforward))


class GNNTF(nn.Module):
    """Shared spatial GCN with scratch or pretrained Transformer fusion."""

    def __init__(
        self,
        num_node_features: int,
        num_tabular_features: int,
        num_windows: int,
        hidden_dim: int = 128,
        backbone: str = "scratch",
        checkpoint_name: str | None = None,
        transformer_layers: int = 3,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if backbone not in {"scratch", "bert", "gpt2", "vit"}:
            raise ValueError(f"unsupported backbone: {backbone}")

        self.backbone = backbone
        self.num_windows = num_windows
        self.hidden_dim = hidden_dim
        self.transformer_layers = transformer_layers
        self.attention_heads = attention_heads
        self.checkpoint_name = checkpoint_name or DEFAULT_CHECKPOINTS.get(backbone)
        self.encoder_is_frozen = backbone != "scratch"
        self.last_attention: Tensor | None = None

        if backbone == "scratch":
            self.transformer_dim = hidden_dim
            self.encoder_layers = nn.ModuleList(
                [
                    InspectableEncoderLayer(
                        hidden_dim, attention_heads, feedforward_dim=2048
                    )
                    for _ in range(transformer_layers)
                ]
            )
        else:
            self.transformer_dim, self.encoder_layers = self._load_pretrained_layers()
            for parameter in self.encoder_layers.parameters():
                parameter.requires_grad = False

        # The paper uses nhid=128 and two 256-dimensional GCN outputs.
        self.gcn_dim = hidden_dim * 2
        self.gcn1 = DenseGCNLayer(num_node_features, self.gcn_dim)
        self.gcn2 = DenseGCNLayer(self.gcn_dim, self.gcn_dim)
        self.graph_projection = nn.Linear(self.gcn_dim, self.transformer_dim)
        self.tabular_norm = nn.BatchNorm1d(num_tabular_features)
        self.tabular_projection = nn.Linear(
            num_tabular_features, self.transformer_dim
        )

        sequence_length = num_windows + (1 if backbone == "gpt2" else 2)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, self.transformer_dim)
        )
        if backbone != "gpt2":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.input_norm = nn.LayerNorm(self.transformer_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.transformer_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
        )

    def _load_pretrained_layers(self) -> tuple[int, nn.ModuleList]:
        try:
            from transformers import (
                BertForSequenceClassification,
                GPT2ForSequenceClassification,
                ViTForImageClassification,
            )
        except ImportError as error:
            raise RuntimeError(
                "Pretrained backbones require: pip install -r requirements-transfer.txt"
            ) from error

        if self.backbone == "bert":
            pretrained = BertForSequenceClassification.from_pretrained(
                self.checkpoint_name
            )
            return int(pretrained.config.hidden_size), pretrained.bert.encoder.layer
        if self.backbone == "vit":
            pretrained = ViTForImageClassification.from_pretrained(
                self.checkpoint_name
            )
            return int(pretrained.config.hidden_size), pretrained.vit.encoder.layer
        pretrained = GPT2ForSequenceClassification.from_pretrained(
            self.checkpoint_name
        )
        return int(pretrained.config.n_embd), pretrained.transformer.h

    def train(self, mode: bool = True) -> "GNNTF":
        super().train(mode)
        if self.encoder_is_frozen:
            self.encoder_layers.eval()
        return self

    def _encode_tokens(self, tokens: Tensor, record_attention: bool) -> Tensor:
        self.last_attention = None
        for layer in self.encoder_layers:
            if self.backbone == "scratch":
                tokens = layer(tokens, record_attention=record_attention)
                if record_attention:
                    self.last_attention = layer.last_attention
            elif self.backbone == "bert":
                outputs = layer(tokens, output_attentions=record_attention)
                tokens = outputs[0]
                if record_attention:
                    self.last_attention = outputs[1]
            elif self.backbone == "vit":
                outputs = layer(tokens, output_attentions=record_attention)
                tokens = outputs[0]
                if record_attention:
                    self.last_attention = outputs[1]
            else:
                outputs = layer(
                    tokens, use_cache=False, output_attentions=record_attention
                )
                tokens = outputs[0]
                if record_attention:
                    self.last_attention = outputs[1]
        return tokens

    def forward(
        self,
        adjacency: Tensor,
        node_features: Tensor,
        tabular: Tensor,
        record_attention: bool = False,
    ) -> Tensor:
        batch_size, num_windows, num_nodes, _ = adjacency.shape
        if num_windows != self.num_windows:
            raise ValueError(f"expected {self.num_windows} windows, got {num_windows}")

        # Flatten subject and time dimensions so one GCN is shared by all windows.
        graphs = adjacency.reshape(batch_size * num_windows, num_nodes, num_nodes)
        nodes = node_features[:, None].expand(-1, num_windows, -1, -1)
        nodes = nodes.reshape(batch_size * num_windows, num_nodes, -1)
        hidden = F.selu(self.gcn1(nodes, graphs))
        hidden = F.selu(self.gcn2(hidden, graphs))
        window_tokens = F.selu(self.graph_projection(hidden.mean(dim=1)))
        window_tokens = window_tokens.reshape(batch_size, num_windows, -1)

        tabular_token = F.selu(self.tabular_projection(self.tabular_norm(tabular)))
        tabular_token = tabular_token.unsqueeze(1)
        if self.backbone == "gpt2":
            # Causal GPT-2 reads the final tabular token after all graph windows.
            tokens = torch.cat([window_tokens, tabular_token], dim=1)
            readout_index = -1
        else:
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tabular_token, window_tokens], dim=1)
            readout_index = 0

        tokens = self.input_norm(tokens + self.position_embedding)
        encoded = self._encode_tokens(tokens, record_attention=record_attention)
        return self.classifier(encoded[:, readout_index])

    def config_dict(self) -> dict[str, object]:
        return {
            "num_windows": self.num_windows,
            "hidden_dim": self.hidden_dim,
            "backbone": self.backbone,
            "checkpoint_name": self.checkpoint_name,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
        }

    def compact_state_dict(self) -> dict[str, Tensor]:
        """Store fitted parameters without duplicating frozen pretrained weights."""

        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
            if self.backbone == "scratch" or not name.startswith("encoder_layers.")
        }

    def load_compact_state_dict(self, state: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed_missing = (
            [name for name in missing if name.startswith("encoder_layers.")]
            if self.encoder_is_frozen
            else []
        )
        if unexpected or sorted(missing) != sorted(allowed_missing):
            raise RuntimeError(
                f"checkpoint mismatch; missing={missing}, unexpected={unexpected}"
            )


def stratified_split(labels: Tensor, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic 70/15/15 class-stratified splits."""

    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    labels_numpy = labels.numpy()
    for label in labels.unique(sorted=True).tolist():
        indices = np.flatnonzero(labels_numpy == label)
        rng.shuffle(indices)
        n_train = int(0.70 * len(indices))
        n_validation = int(0.15 * len(indices))
        if n_validation < 1 or len(indices) - n_train - n_validation < 1:
            raise ValueError("holdout evaluation requires at least 7 subjects in each class")
        train.extend(indices[:n_train].tolist())
        validation.extend(indices[n_train : n_train + n_validation].tolist())
        test.extend(indices[n_train + n_validation :].tolist())
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def stratified_kfold(
    labels: Tensor, n_splits: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Return deterministic stratified train/test indices without scikit-learn."""

    labels_numpy = labels.numpy()
    class_counts = [int((labels_numpy == label).sum()) for label in np.unique(labels_numpy)]
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if min(class_counts) < n_splits:
        raise ValueError(
            f"each class needs at least {n_splits} samples; counts={class_counts}"
        )

    rng = np.random.default_rng(seed)
    per_class_folds: dict[int, list[np.ndarray]] = {}
    for label in np.unique(labels_numpy):
        indices = np.flatnonzero(labels_numpy == label)
        rng.shuffle(indices)
        per_class_folds[int(label)] = list(np.array_split(indices, n_splits))

    all_indices = np.arange(len(labels_numpy))
    splits = []
    for fold_index in range(n_splits):
        test_indices = np.concatenate(
            [per_class_folds[label][fold_index] for label in sorted(per_class_folds)]
        )
        rng.shuffle(test_indices)
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        splits.append((train_indices.tolist(), test_indices.tolist()))
    return splits


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels, probabilities = [], []
    for adjacency, node_features, tabular, target in loader:
        logits = model(
            adjacency.to(device), node_features.to(device), tabular.to(device)
        )
        target = target.to(device)
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        labels.append(target.cpu().numpy())
        probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    y_true = np.concatenate(labels)
    y_probability = np.concatenate(probabilities)
    accuracy = ((y_probability >= 0.5) == y_true).mean()
    return total_loss / len(loader.dataset), float(accuracy), y_true, y_probability


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUC without a scikit-learn dependency."""

    positive = y_true == 1
    n_positive = int(positive.sum())
    n_negative = len(y_true) - n_positive
    if not n_positive or not n_negative:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def build_model(data: ConnectivityData, args: argparse.Namespace, device: torch.device) -> GNNTF:
    return GNNTF(
        num_node_features=data.node_features.shape[1],
        num_tabular_features=data.tabular.shape[1],
        num_windows=args.num_windows,
        hidden_dim=args.hidden_dim,
        backbone=args.model,
        checkpoint_name=args.checkpoint,
    ).to(device)


def train_with_validation(
    dataset: DynamicGraphDataset,
    data: ConnectivityData,
    train_ids: list[int],
    validation_ids: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[GNNTF, dict[str, Tensor], float, int]:
    """Train one model and restore the epoch with minimum validation loss."""

    set_seed(args.seed)
    training_set = Subset(dataset, train_ids)
    train_loader = DataLoader(
        training_set,
        batch_sampler=TrainingBatchSampler(
            torch.utils.data.RandomSampler(training_set), args.batch_size, args.drop_last
        ),
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_ids), batch_size=args.batch_size
    )
    if len(train_loader) == 0:
        raise ValueError("the training loader is empty; disable --drop-last")

    model = build_model(data, args, device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_class = (
        torch.optim.AdamW if args.optimizer.lower() == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_class(trainable_parameters, lr=args.learning_rate)

    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for adjacency, node_features, tabular, target in train_loader:
            optimizer.zero_grad()
            logits = model(
                adjacency.to(device), node_features.to(device), tabular.to(device)
            )
            loss = F.cross_entropy(logits, target.to(device))
            loss.backward()
            optimizer.step()

        validation_loss, _, _, _ = evaluate(model, validation_loader, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = model.compact_state_dict()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_compact_state_dict(best_state)
    return model, best_state, best_validation_loss, best_epoch


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir = Path(args.output_dir or f"outputs/{args.model}")
    if any((output_dir / name).exists() for name in ("data.npz", "model.pt", "metrics.json")):
        raise ValueError(f"{output_dir} already contains a run; choose a new --output-dir")

    print(f"Using device: {device}")
    if args.data:
        print(f"Loading data from {args.data}...")
        data = load_data(Path(args.data))
    else:
        print("Generating synthetic ROI signals and connectivity graphs...")
        data = make_synthetic_dataset(
            num_subjects=args.num_subjects,
            num_nodes=args.num_nodes,
            num_windows=args.num_windows,
            window_size=args.window_size,
            step_size=args.step_size,
            alpha=args.fdr_alpha,
            seed=args.seed,
        )
    args.num_windows = data.adjacency.shape[1]
    density = (data.adjacency != 0).float().mean().item()
    print(
        f"Data: {len(data.labels)} subjects, {args.num_windows} windows, "
        f"{data.adjacency.shape[-1]} nodes; graph density={density:.3f}"
    )

    dataset = DynamicGraphDataset(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(data, output_dir / "data.npz")
    model_config = {
        "num_node_features": data.node_features.shape[1],
        "num_tabular_features": data.tabular.shape[1],
        "num_windows": args.num_windows,
        "hidden_dim": args.hidden_dim,
        "backbone": args.model,
        "checkpoint_name": args.checkpoint or DEFAULT_CHECKPOINTS.get(args.model),
    }

    if args.evaluation == "holdout":
        train_ids, validation_ids, test_ids = stratified_split(data.labels, args.seed)
        print(f"Training one {args.model} model with a stratified holdout split...")
        model, best_state, best_loss, best_epoch = train_with_validation(
            dataset, data, train_ids, validation_ids, args, device
        )
        test_loader = DataLoader(
            Subset(dataset, test_ids), batch_size=args.batch_size
        )
        test_loss, test_accuracy, test_labels, test_probabilities = evaluate(
            model, test_loader, device
        )
        test_auc = binary_auc(test_labels, test_probabilities)
        print(
            f"Best epoch={best_epoch} | test loss={test_loss:.4f} | "
            f"AUC={test_auc:.3f}"
        )
        checkpoint = {
            "model_config": model_config,
            "model_state": best_state,
            "test_ids": test_ids,
            "explain_ids": train_ids[:32],
            "seed": args.seed,
        }
        metrics = {
            "evaluation": "holdout",
            "backbone": args.model,
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
        }
        prediction_rows = [
            (subject_id, int(label), float(probability), 0)
            for subject_id, label, probability in zip(
                test_ids, test_labels, test_probabilities
            )
        ]
    else:
        print(
            f"Running nested stratified CV: {args.outer_folds} outer folds x "
            f"{args.cv_size} inner folds..."
        )
        outer_splits = stratified_kfold(
            data.labels, n_splits=args.outer_folds, seed=args.seed
        )
        fold_results = []
        prediction_rows = []
        representative_state: dict[str, Tensor] | None = None
        representative_test_ids: list[int] = []
        representative_explain_ids: list[int] = []
        representative_loss = float("inf")

        for outer_fold, (outer_train_ids, outer_test_ids) in enumerate(
            outer_splits, start=1
        ):
            outer_train_array = np.asarray(outer_train_ids)
            inner_labels = data.labels[outer_train_ids]
            inner_splits = stratified_kfold(
                inner_labels,
                n_splits=args.cv_size,
                seed=args.seed + outer_fold,
            )
            inner_test_probabilities = []
            inner_best_epochs = []
            outer_test_labels: np.ndarray | None = None

            for inner_fold, (relative_train, relative_validation) in enumerate(
                inner_splits, start=1
            ):
                inner_train_ids = outer_train_array[relative_train].tolist()
                inner_validation_ids = outer_train_array[relative_validation].tolist()
                model, state, validation_loss, best_epoch = train_with_validation(
                    dataset,
                    data,
                    inner_train_ids,
                    inner_validation_ids,
                    args,
                    device,
                )
                outer_test_loader = DataLoader(
                    Subset(dataset, outer_test_ids), batch_size=args.batch_size
                )
                _, _, labels, probabilities = evaluate(
                    model, outer_test_loader, device
                )
                outer_test_labels = labels
                inner_test_probabilities.append(probabilities)
                inner_best_epochs.append(best_epoch)
                print(
                    f"  outer {outer_fold}/{args.outer_folds}, "
                    f"inner {inner_fold}/{args.cv_size}: "
                    f"best epoch={best_epoch}, val loss={validation_loss:.4f}"
                )

                if outer_fold == 1 and validation_loss < representative_loss:
                    representative_loss = validation_loss
                    representative_state = state
                    representative_test_ids = outer_test_ids
                    representative_explain_ids = inner_train_ids[:32]
                del model

            if outer_test_labels is None:
                raise RuntimeError("inner cross-validation produced no predictions")
            # One score per outer-test subject is required for out-of-fold metrics.
            # We average the predictions from the inner-fold early-stopped models.
            fold_probabilities = np.mean(inner_test_probabilities, axis=0)
            fold_auc = binary_auc(outer_test_labels, fold_probabilities)
            fold_results.append(
                {
                    "outer_fold": outer_fold,
                    "auc": fold_auc,
                    "mean_best_epoch": float(np.mean(inner_best_epochs)),
                }
            )
            prediction_rows.extend(
                (
                    subject_id,
                    int(label),
                    float(probability),
                    outer_fold,
                )
                for subject_id, label, probability in zip(
                    outer_test_ids, outer_test_labels, fold_probabilities
                )
            )
            print(f"Outer fold {outer_fold}: AUC={fold_auc:.3f}")

        if representative_state is None:
            raise RuntimeError("nested cross-validation produced no checkpoint")
        fold_aucs = np.array([row["auc"] for row in fold_results])
        oof_labels = np.array([row[1] for row in prediction_rows])
        oof_probabilities = np.array([row[2] for row in prediction_rows])
        print(
            f"Nested CV AUC={fold_aucs.mean():.3f} +/- {fold_aucs.std(ddof=1):.3f}"
        )
        checkpoint = {
            "model_config": model_config,
            "model_state": representative_state,
            "test_ids": representative_test_ids,
            "explain_ids": representative_explain_ids,
            "seed": args.seed,
            "note": "Model from outer fold 1 used for visualization.",
        }
        metrics = {
            "evaluation": "nested-cv",
            "backbone": args.model,
            "outer_folds": args.outer_folds,
            "inner_folds": args.cv_size,
            "patience": args.patience,
            "folds": fold_results,
            "mean_auc": float(fold_aucs.mean()),
            "std_auc": float(fold_aucs.std(ddof=1)),
            "oof_auc": binary_auc(oof_labels, oof_probabilities),
        }

    metrics["data_source"] = "provided" if args.data else "synthetic"
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["subject_id", "label", "probability", "outer_fold"])
        writer.writerows(prediction_rows)
    print(f"Saved data, checkpoint, and metrics to: {output_dir.resolve()}")
    print("Run `python visualize.py` to create attention and Top-25 edge figures.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNN-TF."
    )
    parser.add_argument(
        "--model", choices=["scratch", "bert", "gpt2", "vit"], default="scratch"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional Hugging Face checkpoint override for a pretrained model.",
    )
    parser.add_argument(
        "--evaluation",
        choices=["holdout", "nested-cv"],
        default="holdout",
        help="Use 'nested-cv' for outer/inner stratified evaluation.",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--cv-size", "--cv_size", dest="cv_size", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-subjects", type=int, default=80)
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=130)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--hidden-dim", "--nhid", dest="hidden_dim", type=int, default=128)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument(
        "--drop-last",
        "--droplast",
        dest="drop_last",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--optimizer",
        "--ifadamw",
        dest="optimizer",
        choices=["Adam", "AdamW", "Adamw"],
        default="Adam",
    )
    parser.add_argument("--data", help="Input NPZ file; omit to generate synthetic data.")
    parser.add_argument("--output-dir", help="Run directory (default: outputs/<model>).")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="'auto' uses CUDA when available and otherwise uses CPU.",
    )
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for BatchNorm")
    return args


if __name__ == "__main__":
    run_training(parse_args())
