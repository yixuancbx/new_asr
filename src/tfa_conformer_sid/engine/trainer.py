from __future__ import annotations

import math
import random
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


def _resolve_loss_type(train_cfg: TrainingConfig) -> str:
    loss_type = str(train_cfg.loss_type).strip().lower()
    if loss_type not in {"ce", "arcface", "cosface"}:
        raise ValueError("train.loss_type 仅支持: ce / arcface / cosface")
    return loss_type


def _build_margin_logits(
    cosine_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    margin: float,
    scale: float,
    easy_margin: bool,
) -> torch.Tensor:
    cosine_logits = cosine_logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    if loss_type == "arcface":
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        sine = torch.sqrt(torch.clamp(1.0 - cosine_logits.pow(2), min=1e-7))
        phi = cosine_logits * cos_m - sine * sin_m

        if easy_margin:
            target_logits = torch.where(cosine_logits > 0, phi, cosine_logits)
        else:
            threshold = math.cos(math.pi - margin)
            mm = math.sin(math.pi - margin) * margin
            target_logits = torch.where(cosine_logits > threshold, phi, cosine_logits - mm)
    elif loss_type == "cosface":
        target_logits = cosine_logits - margin
    else:
        raise ValueError(f"不支持的 margin loss 类型: {loss_type}")

    one_hot = F.one_hot(labels, num_classes=cosine_logits.size(1)).type_as(cosine_logits)
    logits = (1.0 - one_hot) * cosine_logits + one_hot * target_logits
    return logits * scale


def _compute_loss_and_metric_logits(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_cfg: TrainingConfig,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_type == "ce":
        logits = outputs["logits"]
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=train_cfg.label_smoothing,
        )
        return loss, logits

    cosine_logits = outputs.get("cosine_logits")
    if cosine_logits is None:
        raise KeyError("模型输出缺少 cosine_logits，无法计算 ArcFace/CosFace 损失")

    margin = float(train_cfg.loss_margin)
    scale = float(train_cfg.loss_scale)
    if margin < 0:
        raise ValueError("train.loss_margin 不能为负数")
    if scale <= 0:
        raise ValueError("train.loss_scale 必须大于 0")

    logits_for_loss = _build_margin_logits(
        cosine_logits=cosine_logits,
        labels=labels,
        loss_type=loss_type,
        margin=margin,
        scale=scale,
        easy_margin=bool(train_cfg.loss_easy_margin),
    )
    loss = F.cross_entropy(
        logits_for_loss,
        labels,
        label_smoothing=train_cfg.label_smoothing,
    )

    # 评估分类效果时使用不带 margin 的推理 logits，更接近实际推理场景
    logits_for_metric = cosine_logits * scale
    return loss, logits_for_metric


def _to_device_batch(feats, device: torch.device):
    if isinstance(feats, dict):
        moved = {}
        for key, value in feats.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        return moved
    return feats.to(device, non_blocking=True)


def _apply_modality_dropout(feats, train_cfg: TrainingConfig, train_mode: bool):
    if not train_mode or not isinstance(feats, dict):
        return feats
    p_video = min(max(float(train_cfg.modality_drop_video_prob), 0.0), 1.0)
    p_audio = min(max(float(train_cfg.modality_drop_audio_prob), 0.0), 1.0)
    if "video" in feats and p_video > 0 and random.random() < p_video:
        feats["video"] = torch.zeros_like(feats["video"])
        if "video_mask" in feats:
            feats["video_mask"] = torch.zeros_like(feats["video_mask"])
    if "audio" in feats and p_audio > 0 and random.random() < p_audio:
        feats["audio"] = torch.zeros_like(feats["audio"])
    return feats


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
    loss_type = _resolve_loss_type(train_cfg)

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
            feats = _to_device_batch(feats, device=device)
            feats = _apply_modality_dropout(feats, train_cfg=train_cfg, train_mode=train_mode)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(feats)
                loss, metric_logits = _compute_loss_and_metric_logits(
                    outputs=outputs,
                    labels=labels,
                    train_cfg=train_cfg,
                    loss_type=loss_type,
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

            preds = metric_logits.argmax(dim=1)

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

