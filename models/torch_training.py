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
    def __init__(self, arrays: SequenceDatasetArrays, *, feature_clip: float | None = 20.0) -> None:
        self.x = torch.from_numpy(_clean_feature_array(arrays.x, feature_clip=feature_clip))
        self.y = torch.from_numpy(arrays.y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class FrameSequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        lookback: int,
        group_column: str = "symbol",
        label_column: str = "label",
        feature_clip: float | None = 20.0,
    ) -> None:
        if lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        missing = [column for column in feature_columns + [label_column] if column not in frame.columns]
        if missing:
            raise ValueError(f"Sequence dataframe missing columns: {missing}")

        self.lookback = lookback
        self.feature_blocks: list[np.ndarray] = []
        self.label_blocks: list[np.ndarray] = []
        self.sequence_counts: list[int] = []
        self.nonfinite_values_replaced = 0
        self.clipped_values = 0
        self.feature_clip = feature_clip

        if group_column in frame.columns:
            groups = frame.groupby(group_column, sort=False, observed=True)
        else:
            groups = [(None, frame)]

        for _, group in groups:
            ordered = group.sort_index()
            if len(ordered) < lookback:
                continue
            raw_features = ordered[feature_columns].to_numpy(dtype=np.float32)
            self.nonfinite_values_replaced += int(np.count_nonzero(~np.isfinite(raw_features)))
            if feature_clip is not None:
                finite_features = raw_features[np.isfinite(raw_features)]
                self.clipped_values += int(np.count_nonzero(np.abs(finite_features) > feature_clip))
            features = _clean_feature_array(raw_features, feature_clip=feature_clip)
            labels = ordered[label_column].to_numpy(dtype=np.int64)
            sequence_count = len(ordered) - lookback + 1
            self.feature_blocks.append(np.ascontiguousarray(features))
            self.label_blocks.append(labels[lookback - 1 :])
            self.sequence_counts.append(sequence_count)

        if not self.sequence_counts:
            raise RuntimeError("No sequence arrays could be built for this split.")
        self.cumulative_counts = np.cumsum(self.sequence_counts)
        self.labels = np.concatenate(self.label_blocks)

    def __len__(self) -> int:
        return int(self.cumulative_counts[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        block_index = int(np.searchsorted(self.cumulative_counts, index, side="right"))
        previous_count = 0 if block_index == 0 else int(self.cumulative_counts[block_index - 1])
        local_index = index - previous_count
        features = self.feature_blocks[block_index][local_index : local_index + self.lookback]
        label = self.label_blocks[block_index][local_index]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


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
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
) -> TorchTrainingResult:
    train_dataset = NumpySequenceDataset(train_arrays)
    validation_dataset = NumpySequenceDataset(validation_arrays)
    return _train_torch_classifier_from_datasets(
        model=model,
        model_name=model_name,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_labels=train_arrays.y,
        shuffle_train=False,
        model_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        device=device,
        log_every_batches=log_every_batches,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
    )


def train_torch_classifier_from_frames(
    *,
    model: nn.Module,
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    model_dir: Path,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_workers: int = 2,
    device: torch.device | None = None,
    log_every_batches: int = 0,
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
) -> TorchTrainingResult:
    train_dataset = FrameSequenceDataset(train_frame, feature_columns=feature_columns, lookback=lookback)
    validation_dataset = FrameSequenceDataset(
        validation_frame,
        feature_columns=feature_columns,
        lookback=lookback,
    )
    print(
        f"{model_name}: cleaned sequence features "
        f"train_nonfinite={train_dataset.nonfinite_values_replaced:,} "
        f"validation_nonfinite={validation_dataset.nonfinite_values_replaced:,} "
        f"train_clipped={train_dataset.clipped_values:,} "
        f"validation_clipped={validation_dataset.clipped_values:,}",
        flush=True,
    )
    return _train_torch_classifier_from_datasets(
        model=model,
        model_name=model_name,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_labels=train_dataset.labels,
        shuffle_train=True,
        model_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        device=device,
        log_every_batches=log_every_batches,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
    )


def _train_torch_classifier_from_datasets(
    *,
    model: nn.Module,
    model_name: str,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    train_labels: np.ndarray,
    shuffle_train: bool,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    device: torch.device | None,
    log_every_batches: int,
    max_grad_norm: float | None,
    use_amp: bool,
) -> TorchTrainingResult:
    model_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"{model_name}: device={device}, train_sequences={len(train_dataset):,}, "
        f"validation_sequences={len(validation_dataset):,}, batch_size={batch_size}, "
        f"train_batches={len(train_loader):,}, validation_batches={len(validation_loader):,}",
        flush=True,
    )

    weights = _class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and use_amp)

    latest_path = model_dir / f"{model_name}_latest.pt"
    best_path = model_dir / f"{model_name}_best.pt"
    final_path = model_dir / f"{model_name}_final.pt"
    history_path = model_dir / f"{model_name}_history.json"
    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        checkpoint_history = list(checkpoint.get("history", []))
        if _history_has_nonfinite(checkpoint_history) or not _state_dict_is_finite(checkpoint["model_state"]):
            print(
                f"Ignoring {model_name} checkpoint with non-finite history or weights: {latest_path}",
                flush=True,
            )
        else:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            history = checkpoint_history
            print(f"Resumed {model_name} from epoch {start_epoch}", flush=True)

    if start_epoch >= epochs:
        print(
            f"{model_name}: checkpoint already completed requested epochs "
            f"(start_epoch={start_epoch}, epochs={epochs}). Increase EPOCHS or remove the checkpoint to train more.",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        print(f"{model_name} epoch={epoch} starting training...", flush=True)
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            model_name=model_name,
            epoch=epoch,
            log_every_batches=log_every_batches,
        )
        print(f"{model_name} epoch={epoch} starting validation...", flush=True)
        val_loss, val_accuracy = _evaluate(
            model,
            validation_loader,
            criterion,
            device=device,
            use_amp=use_amp,
        )
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
            f"val_accuracy={val_accuracy:.4f}",
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
    scaler: torch.amp.GradScaler,
    max_grad_norm: float | None,
    use_amp: bool,
    model_name: str,
    epoch: int,
    log_every_batches: int,
) -> float:
    model.train()
    losses: list[float] = []
    total_batches = len(loader)
    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        if not torch.isfinite(x_batch).all():
            raise FloatingPointError(f"{model_name} epoch={epoch} batch={batch_index} contains non-finite inputs.")
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, use_amp=use_amp):
            loss = criterion(model(x_batch), y_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{model_name} epoch={epoch} batch={batch_index} produced non-finite loss. "
                "Check feature cleaning and reduce learning_rate or disable mixed precision if this persists."
            )
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"{model_name} epoch={epoch} batch={batch_index} produced non-finite gradients."
                )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if log_every_batches and (
            batch_index == 1 or batch_index % log_every_batches == 0 or batch_index == total_batches
        ):
            print(
                f"{model_name} epoch={epoch} train batch "
                f"{batch_index:,}/{total_batches:,} loss={losses[-1]:.5f}",
                flush=True,
            )
    return float(np.mean(losses)) if losses else 0.0


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if not torch.isfinite(x_batch).all():
                raise FloatingPointError("Validation batch contains non-finite inputs.")
            with _autocast_context(device=device, use_amp=use_amp):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation produced non-finite loss.")
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == y_batch).sum().cpu())
            total += int(y_batch.numel())
    return float(np.mean(losses)) if losses else 0.0, (correct / total if total else 0.0)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=3).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _autocast_context(*, device: torch.device, use_amp: bool) -> Any:
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


def _clean_feature_array(values: np.ndarray, *, feature_clip: float | None) -> np.ndarray:
    cleaned = np.asarray(values, dtype=np.float32)
    if not np.isfinite(cleaned).all():
        cleaned = cleaned.copy()
        np.nan_to_num(cleaned, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_clip is not None:
        cleaned = np.clip(cleaned, -feature_clip, feature_clip)
    return np.ascontiguousarray(cleaned)


def _history_has_nonfinite(history: list[dict[str, Any]]) -> bool:
    for row in history:
        for key in ("train_loss", "val_loss", "val_accuracy"):
            value = row.get(key)
            if value is not None and not np.isfinite(float(value)):
                return True
    return False


def _state_dict_is_finite(state_dict: dict[str, Any]) -> bool:
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            return False
    return True
