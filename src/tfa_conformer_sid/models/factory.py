from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import torch.nn as nn

from .baselines import (
    BaselineECAPASpeakerNet,
    BaselineMFAConformerSpeakerNet,
    BaselineResNetXvectorSpeakerNet,
)
from .tfa_multiscale_conformer import ModelConfig, TFAMultiScaleConformerSpeakerNet


_BACKBONE_ALIASES = {
    "tfa": "tfa_multiscale_conformer",
    "tfa_multiscale_conformer": "tfa_multiscale_conformer",
    "default": "tfa_multiscale_conformer",
    "ecapa": "ecapa_tdnn",
    "ecapa_tdnn": "ecapa_tdnn",
    "baseline_ecapa": "ecapa_tdnn",
    "resnet": "resnet_xvector",
    "resnet_xvector": "resnet_xvector",
    "baseline_resnet": "resnet_xvector",
    "mfa": "mfa_conformer",
    "mfa_conformer": "mfa_conformer",
    "conformer_cat": "mfa_conformer",
}


def normalize_backbone_type(backbone_type: str) -> str:
    key = str(backbone_type).strip().lower().replace("-", "_")
    return _BACKBONE_ALIASES.get(key, key)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(value))
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _build_baseline_model(cfg: ModelConfig, backbone_type: str, width: int) -> nn.Module:
    if backbone_type == "ecapa_tdnn":
        return BaselineECAPASpeakerNet(cfg=cfg, channels=width)
    if backbone_type == "resnet_xvector":
        return BaselineResNetXvectorSpeakerNet(cfg=cfg, base_channels=width)
    if backbone_type == "mfa_conformer":
        num_blocks = max(1, int(cfg.baseline_mfa_num_blocks))
        return BaselineMFAConformerSpeakerNet(
            cfg=cfg,
            model_dim=width,
            num_blocks=num_blocks,
        )
    raise ValueError(f"不支持的 baseline 类型: {backbone_type}")


def _baseline_seed_width(cfg: ModelConfig, backbone_type: str) -> int:
    if backbone_type == "ecapa_tdnn":
        return max(64, int(cfg.baseline_ecapa_channels))
    if backbone_type == "resnet_xvector":
        return max(16, int(cfg.baseline_resnet_base_channels))
    if backbone_type == "mfa_conformer":
        return max(64, int(cfg.baseline_mfa_conformer_dim))
    raise ValueError(f"不支持的 baseline 类型: {backbone_type}")


def _candidate_widths(seed: int, backbone_type: str) -> Tuple[int, ...]:
    if backbone_type == "ecapa_tdnn":
        multiples = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.5)
        widths = {_round_to_multiple(int(seed * ratio), 16) for ratio in multiples}
        widths.add(_round_to_multiple(seed, 16))
        return tuple(sorted(w for w in widths if w >= 64))

    if backbone_type == "resnet_xvector":
        multiples = (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
        widths = {_round_to_multiple(int(seed * ratio), 8) for ratio in multiples}
        widths.add(_round_to_multiple(seed, 8))
        return tuple(sorted(w for w in widths if w >= 16))

    if backbone_type == "mfa_conformer":
        multiples = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0, 2.5)
        widths = {_round_to_multiple(int(seed * ratio), 16) for ratio in multiples}
        widths.add(_round_to_multiple(seed, 16))
        return tuple(sorted(w for w in widths if w >= 64))

    raise ValueError(f"不支持的 baseline 类型: {backbone_type}")


def _resolve_target_param_budget(cfg: ModelConfig) -> int:
    explicit_budget = int(cfg.baseline_target_params)
    if explicit_budget > 0:
        return explicit_budget

    ref_cfg = replace(cfg, backbone_type="tfa_multiscale_conformer")
    try:
        reference_model = TFAMultiScaleConformerSpeakerNet(ref_cfg)
        return count_trainable_parameters(reference_model)
    except Exception:
        # baseline 是纯音频模型；当视频依赖不可用时，回退到音频分支预算
        audio_ref_cfg = replace(ref_cfg, use_audio_branch=True, use_video_branch=False)
        reference_model = TFAMultiScaleConformerSpeakerNet(audio_ref_cfg)
        return count_trainable_parameters(reference_model)


def _build_baseline_with_budget_match(cfg: ModelConfig, backbone_type: str) -> nn.Module:
    target_params = _resolve_target_param_budget(cfg)
    seed_width = _baseline_seed_width(cfg, backbone_type=backbone_type)
    candidates = _candidate_widths(seed=seed_width, backbone_type=backbone_type)
    if not candidates:
        raise RuntimeError("未生成有效的 baseline 宽度候选，无法进行参数匹配")

    best_model = None
    best_meta = None
    for width in candidates:
        model = _build_baseline_model(cfg=cfg, backbone_type=backbone_type, width=width)
        params = count_trainable_parameters(model)
        diff = abs(params - target_params)
        if best_meta is None or diff < best_meta["diff"]:
            best_model = model
            best_meta = {"width": width, "params": params, "diff": diff}

    if best_model is None or best_meta is None:
        raise RuntimeError("baseline 参数匹配失败，未找到可用模型")

    best_model.param_match_info = {
        "enabled": True,
        "target_params": target_params,
        "selected_width": best_meta["width"],
        "selected_params": best_meta["params"],
        "abs_diff": best_meta["diff"],
    }
    return best_model


def build_speaker_model(cfg: ModelConfig) -> nn.Module:
    backbone_type = normalize_backbone_type(cfg.backbone_type)
    if backbone_type == "tfa_multiscale_conformer":
        model = TFAMultiScaleConformerSpeakerNet(cfg)
        model.backbone_type = "tfa_multiscale_conformer"
        model.param_match_info = {
            "enabled": False,
            "target_params": None,
            "selected_width": None,
            "selected_params": count_trainable_parameters(model),
            "abs_diff": None,
        }
        return model

    if backbone_type not in {"ecapa_tdnn", "resnet_xvector", "mfa_conformer"}:
        raise ValueError(
            "model.backbone_type 仅支持: "
            "tfa_multiscale_conformer / ecapa_tdnn / resnet_xvector / mfa_conformer"
        )

    if bool(cfg.auto_match_baseline_params):
        return _build_baseline_with_budget_match(cfg=cfg, backbone_type=backbone_type)

    width = _baseline_seed_width(cfg, backbone_type=backbone_type)
    model = _build_baseline_model(cfg=cfg, backbone_type=backbone_type, width=width)
    model.param_match_info = {
        "enabled": False,
        "target_params": int(cfg.baseline_target_params) if int(cfg.baseline_target_params) > 0 else None,
        "selected_width": width,
        "selected_params": count_trainable_parameters(model),
        "abs_diff": None,
    }
    return model


__all__ = [
    "build_speaker_model",
    "count_trainable_parameters",
    "normalize_backbone_type",
]
