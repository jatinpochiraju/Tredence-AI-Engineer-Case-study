"""Self-pruning neural network case study on CIFAR-10.

This script implements a custom PrunableLinear layer whose weights are
multiplied by learnable sigmoid gates. Training adds an L1 penalty over the gate
values so the model learns to suppress weak connections while optimizing the
classification objective.

Example:
    python self_pruning_cifar10.py --epochs 10 --lambdas 0 1e-6 5e-6

For a fast correctness check without downloading CIFAR-10:
    python self_pruning_cifar10.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Keep matplotlib/font cache writes inside the project when home cache
# directories are not writable, as can happen in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PrunableLinear(nn.Module):
    """Linear layer with one learnable gate per weight.

    gate_scores are unconstrained trainable parameters. The forward pass maps
    them into (0, 1) with sigmoid, then multiplies them element-wise with the
    layer weights before applying the linear operation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_init = gate_init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.prune_threshold: float | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # A score of 0 starts each gate at sigmoid(0) = 0.5. That keeps all
        # connections trainable while leaving room for L1 pressure to close
        # unhelpful gates within a short case-study run.
        nn.init.constant_(self.gate_scores, self.gate_init)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gates = self.gates
        if self.prune_threshold is not None:
            gates = gates * (gates >= self.prune_threshold).to(gates.dtype)
        pruned_weights = self.weight * gates
        return F.linear(inputs, pruned_weights, self.bias)


class SelfPruningMLP(nn.Module):
    """Feed-forward CIFAR-10 classifier using PrunableLinear layers."""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (1024, 512),
        dropout: float = 0.15,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 3 * 32 * 32
        for hidden_size in hidden_sizes:
            layers.append(PrunableLinear(in_features, hidden_size, gate_init=gate_init))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        layers.append(PrunableLinear(in_features, 10, gate_init=gate_init))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened = torch.flatten(images, start_dim=1)
        return self.net(flattened)

    def prunable_layers(self) -> Iterable[PrunableLinear]:
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def set_pruning_threshold(self, threshold: float | None) -> None:
        for layer in self.prunable_layers():
            layer.prune_threshold = threshold


@dataclass
class ExperimentResult:
    lambda_value: float
    test_accuracy: float
    sparsity_percent: float
    best_epoch: int
    best_accuracy: float
    checkpoint_path: str
    gate_histogram_path: str


def sparsity_loss(model: SelfPruningMLP) -> torch.Tensor:
    losses = [layer.gates.sum() for layer in model.prunable_layers()]
    if not losses:
        raise ValueError("No PrunableLinear layers found.")
    return torch.stack(losses).sum()


@torch.no_grad()
def gate_values(model: SelfPruningMLP) -> torch.Tensor:
    return torch.cat([layer.gates.detach().flatten().cpu() for layer in model.prunable_layers()])


@torch.no_grad()
def sparsity_level(model: SelfPruningMLP, threshold: float) -> float:
    values = gate_values(model)
    return 100.0 * (values < threshold).float().mean().item()


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_one_epoch(
    model: SelfPruningMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_value: float,
    log_every: int,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_sparsity_loss = 0.0
    seen_batches = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        classification_loss = F.cross_entropy(logits, labels)
        gate_penalty = sparsity_loss(model)
        total_loss = classification_loss + lambda_value * gate_penalty
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_accuracy += accuracy_from_logits(logits, labels)
        running_sparsity_loss += gate_penalty.item()
        seen_batches += 1

        if log_every > 0 and batch_idx % log_every == 0:
            avg_loss = running_loss / seen_batches
            avg_acc = 100.0 * running_accuracy / seen_batches
            avg_gate_penalty = running_sparsity_loss / seen_batches
            print(
                f"  batch {batch_idx:04d}/{len(loader)} "
                f"loss={avg_loss:.4f} acc={avg_acc:.2f}% gate_l1={avg_gate_penalty:.0f}"
            )

    return (
        running_loss / max(seen_batches, 1),
        100.0 * running_accuracy / max(seen_batches, 1),
        running_sparsity_loss / max(seen_batches, 1),
    )


@torch.no_grad()
def evaluate(model: SelfPruningMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def build_optimizer(
    model: SelfPruningMLP,
    lr: float,
    weight_decay: float,
    gate_lr_multiplier: float,
) -> torch.optim.Optimizer:
    gate_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("gate_scores"):
            gate_params.append(parameter)
        else:
            other_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": gate_params, "lr": lr * gate_lr_multiplier, "weight_decay": 0.0},
        ]
    )


def subset_dataset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=train_transform,
        download=not args.no_download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=test_transform,
        download=not args.no_download,
    )

    train_dataset = subset_dataset(train_dataset, args.train_subset, args.seed)
    test_dataset = subset_dataset(test_dataset, args.test_subset, args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    return train_loader, test_loader


def plot_gate_histogram(values: torch.Tensor, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values.numpy(), bins=80, color="#2a9d8f", edgecolor="#1f2933")
    plt.xlabel("Gate value: sigmoid(gate_score)")
    plt.ylabel("Number of weights")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_for_lambda(
    lambda_value: float,
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> ExperimentResult:
    model = SelfPruningMLP(
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
        gate_init=args.gate_init,
    ).to(device)
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gate_lr_multiplier=args.gate_lr_multiplier,
    )

    best_accuracy = -1.0
    best_epoch = 0
    checkpoint_path = args.output_dir / f"model_lambda_{lambda_value:g}.pt"

    if checkpoint_path.exists() and not args.force_train:
        print(f"\nLambda={lambda_value:g} | reusing checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_accuracy = float(checkpoint["test_accuracy"])
        best_epoch = int(checkpoint["epoch"])
        final_sparsity = sparsity_level(model, args.sparsity_threshold)
        model.set_pruning_threshold(args.sparsity_threshold)
        final_accuracy = evaluate(model, test_loader, device)
        model.set_pruning_threshold(None)
        histogram_path = args.output_dir / f"gate_histogram_lambda_{lambda_value:g}.png"
        plot_gate_histogram(
            gate_values(model),
            histogram_path,
            title=f"Final gate distribution, lambda={lambda_value:g}",
        )
        return ExperimentResult(
            lambda_value=lambda_value,
            test_accuracy=final_accuracy,
            sparsity_percent=final_sparsity,
            best_epoch=best_epoch,
            best_accuracy=best_accuracy,
            checkpoint_path=str(checkpoint_path),
            gate_histogram_path=str(histogram_path),
        )

    for epoch in range(1, args.epochs + 1):
        print(f"\nLambda={lambda_value:g} | epoch {epoch}/{args.epochs}")
        train_loss, train_acc, gate_l1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_value=lambda_value,
            log_every=args.log_every,
        )
        test_accuracy = evaluate(model, test_loader, device)
        current_sparsity = sparsity_level(model, args.sparsity_threshold)
        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_acc={test_accuracy:.2f}% sparsity={current_sparsity:.2f}% "
            f"gate_l1={gate_l1:.0f}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "lambda_value": lambda_value,
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_sparsity = sparsity_level(model, args.sparsity_threshold)
    model.set_pruning_threshold(args.sparsity_threshold)
    final_accuracy = evaluate(model, test_loader, device)
    model.set_pruning_threshold(None)

    histogram_path = args.output_dir / f"gate_histogram_lambda_{lambda_value:g}.png"
    plot_gate_histogram(
        gate_values(model),
        histogram_path,
        title=f"Final gate distribution, lambda={lambda_value:g}",
    )

    return ExperimentResult(
        lambda_value=lambda_value,
        test_accuracy=final_accuracy,
        sparsity_percent=final_sparsity,
        best_epoch=best_epoch,
        best_accuracy=best_accuracy,
        checkpoint_path=str(checkpoint_path),
        gate_histogram_path=str(histogram_path),
    )


def write_results(results: list[ExperimentResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "results_table.md"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(result) for result in results], file, indent=2)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write("| Lambda | Test Accuracy | Sparsity Level (%) |\n")
        file.write("|---:|---:|---:|\n")
        for result in results:
            file.write(
                f"| {result.lambda_value:g} | {result.test_accuracy:.2f}% "
                f"| {result.sparsity_percent:.2f}% |\n"
            )

    print(f"\nSaved results to {csv_path}, {json_path}, and {markdown_path}")


def run_smoke_test() -> None:
    """Verify shapes and gradient flow through weights and gate scores."""
    seed_everything(7)
    model = SelfPruningMLP(hidden_sizes=(32,), dropout=0.0)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    logits = model(images)
    loss = F.cross_entropy(logits, labels) + 1e-6 * sparsity_loss(model)
    loss.backward()

    first_layer = next(model.prunable_layers())
    assert logits.shape == (8, 10)
    assert first_layer.weight.grad is not None
    assert first_layer.gate_scores.grad is not None
    assert torch.isfinite(first_layer.weight.grad).all()
    assert torch.isfinite(first_layer.gate_scores.grad).all()
    print("Smoke test passed: PrunableLinear forward/backward gradients are valid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning CIFAR-10 MLP.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--gate-init", type=float, default=0.0)
    parser.add_argument("--gate-lr-multiplier", type=float, default=4.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1e-6, 5e-6])
    parser.add_argument("--sparsity-threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested_device)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
        return

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    args.device = device.type
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(args)
    results = [
        train_for_lambda(lambda_value, args, train_loader, test_loader, device)
        for lambda_value in args.lambdas
    ]
    write_results(results, args.output_dir)


if __name__ == "__main__":
    main()
