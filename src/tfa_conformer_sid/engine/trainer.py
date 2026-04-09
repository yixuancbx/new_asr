from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from ..config import TrainingConfig
from ..utils import classification_metrics


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    lr: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "lr": self.lr,
        }


def _iter_with_progress(loader: Iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(loader, desc=desc, leave=False)
    except ImportError:
        return loader


def run_one_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    train_cfg: TrainingConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    train_mode: bool = True,
    epoch_idx: int = 0,
) -> EpochResult:
    if train_mode and optimizer is None:
        raise ValueError("train_mode=True 时必须提供 optimizer")

    use_amp = bool(train_cfg.amp and device.type == "cuda")
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0
    all_true = []
    all_pred = []

    context = torch.enable_grad if train_mode else torch.no_grad
    iter_loader = _iter_with_progress(
        loader, desc=f"{'Train' if train_mode else 'Eval'} Epoch {epoch_idx}"
    )

    with context():
        for step, (feats, labels) in enumerate(iter_loader, start=1):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(feats)
                loss = F.cross_entropy(
                    outputs["logits"],
                    labels,
                    label_smoothing=train_cfg.label_smoothing,
                )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if train_cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if train_cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if train_cfg.log_interval > 0 and step % train_cfg.log_interval == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(
                        f"[Train] epoch={epoch_idx} step={step} "
                        f"loss={loss.item():.4f} lr={lr_now:.6g}"
                    )

            logits = outputs["logits"]
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_true.append(labels.detach().cpu())
            all_pred.append(preds.detach().cpu())

    if train_mode and optimizer is not None:
        lr_val = float(optimizer.param_groups[0]["lr"])
    else:
        lr_val = 0.0

    if total_count == 0:
        return EpochResult(
            loss=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            lr=lr_val,
        )

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    m = classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    return EpochResult(
        loss=total_loss / max(total_count, 1),
        accuracy=m["accuracy"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        lr=lr_val,
    )

