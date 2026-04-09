from __future__ import annotations

from typing import Dict

import torch


def _confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    cm = _confusion_matrix(y_true.cpu(), y_pred.cpu(), num_classes=num_classes).float()

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    support = cm.sum(dim=1)
    valid = support > 0

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    if valid.any():
        macro_precision = precision[valid].mean().item()
        macro_recall = recall[valid].mean().item()
        macro_f1 = f1[valid].mean().item()
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    accuracy = (tp.sum() / (cm.sum() + eps)).item()
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }

