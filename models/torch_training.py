from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from features.sequences import SequenceDatasetArrays


class NumpySequenceDataset(Dataset):
    def __init__(self, arrays: SequenceDatasetArrays) -> None:
        self.x = torch.from_numpy(arrays.x)
        self.y = torch.from_numpy(arrays.y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass(frozen=True)
class TorchTrainingResult:
    model_name: str
    latest_checkpoint: Path
    best_checkpoint: Path
    final_model: Path
    history_path: Path
    start_epoch: int
    completed_epochs: int
    best_val_loss: float


def train_torch_classifier(
    *,
    model: nn.Module,
    model_name: str,
    train_arrays: SequenceDatasetArrays,
    validation_arrays: SequenceDatasetArrays,
    model_dir: Path,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_workers: int = 2,
    device: torch.device | None = None,
    log_every_batches: int = 0,
) -> TorchTrainingResult:
    model_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(
        NumpySequenceDataset(train_arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        NumpySequenceDataset(validation_arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"{model_name}: device={device}, train_sequences={len(train_arrays.y):,}, "
        f"validation_sequences={len(validation_arrays.y):,}, batch_size={batch_size}, "
        f"train_batches={len(train_loader):,}, validation_batches={len(validation_loader):,}"
        f"{_device_memory_summary(device)}",
        flush=True,
    )

    weights = _class_weights(train_arrays.y).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    latest_path = model_dir / f"{model_name}_latest.pt"
    best_path = model_dir / f"{model_name}_best.pt"
    final_path = model_dir / f"{model_name}_final.pt"
    history_path = model_dir / f"{model_name}_history.json"
    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", []))
        print(f"Resumed {model_name} from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, epochs):
        print(
            f"{model_name} epoch={epoch} starting training..."
            f"{_device_memory_summary(device)}",
            flush=True,
        )
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            model_name=model_name,
            epoch=epoch,
            log_every_batches=log_every_batches,
        )
        print(
            f"{model_name} epoch={epoch} starting validation..."
            f"{_device_memory_summary(device)}",
            flush=True,
        )
        val_loss, val_accuracy = _evaluate(model, validation_loader, criterion, device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"{model_name} epoch={epoch} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_accuracy={val_accuracy:.4f}"
            f"{_device_memory_summary(device)}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_path)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
        }
        torch.save(checkpoint, latest_path)
        if epoch % 5 == 0:
            shutil.copy2(latest_path, model_dir / f"{model_name}_epoch_{epoch}.pt")

    torch.save({"model_state": model.state_dict()}, final_path)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return TorchTrainingResult(
        model_name=model_name,
        latest_checkpoint=latest_path,
        best_checkpoint=best_path,
        final_model=final_path,
        history_path=history_path,
        start_epoch=start_epoch,
        completed_epochs=max(0, epochs - start_epoch),
        best_val_loss=best_val_loss,
    )


def predict_torch_probabilities(
    *,
    model: nn.Module,
    arrays: SequenceDatasetArrays,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(
        NumpySequenceDataset(arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    values = np.concatenate(probabilities, axis=0)
    return pd.DataFrame(values, index=arrays.index, columns=["p_sell", "p_hold", "p_buy"])


def write_torch_training_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def result_to_dict(result: TorchTrainingResult) -> dict[str, Any]:
    data = asdict(result)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    model_name: str,
    epoch: int,
    log_every_batches: int,
) -> float:
    model.train()
    losses: list[float] = []
    autocast_context = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    total_batches = len(loader)
    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            loss = criterion(model(x_batch), y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if log_every_batches and (
            batch_index == 1 or batch_index % log_every_batches == 0 or batch_index == total_batches
        ):
            print(
                f"{model_name} epoch={epoch} train batch "
                f"{batch_index:,}/{total_batches:,} loss={losses[-1]:.5f}"
                f"{_device_memory_summary(device)}",
                flush=True,
            )
    return float(np.mean(losses)) if losses else 0.0


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    autocast_context = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with autocast_context():
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == y_batch).sum().cpu())
            total += int(y_batch.numel())
    return float(np.mean(losses)) if losses else 0.0, (correct / total if total else 0.0)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=3).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _device_memory_summary(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    try:
        free, total = torch.cuda.mem_get_info(device)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        free_text = f" cuda_free={free_gb:.2f}/{total_gb:.2f}GB"
    except Exception:
        free_text = ""
    return (
        f" cuda_alloc={allocated:.2f}GB"
        f" cuda_reserved={reserved:.2f}GB"
        f" cuda_max_alloc={max_allocated:.2f}GB"
        f"{free_text}"
    )
