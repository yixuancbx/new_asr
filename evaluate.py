from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfa_conformer_sid.config import load_yaml_config
from tfa_conformer_sid.dataio import (
    AudioFeatureExtractor,
    SpeakerFeatureDataset,
    VideoROIExtractor,
    build_label_map,
    scan_speaker_samples,
    speaker_batch_collate,
    split_samples_per_speaker,
)
from tfa_conformer_sid.engine import run_one_epoch
from tfa_conformer_sid.models import build_speaker_model, count_trainable_parameters


def load_model_state_dict_flexible(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    source: str,
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    print(f"[Runtime] 从 {source} 加载模型参数（strict=False）")
    if missing_keys:
        print(
            f"[Warn] 缺少参数 {len(missing_keys)} 个，示例: {missing_keys[:10]}"
        )
    if unexpected_keys:
        print(
            f"[Warn] 多余参数 {len(unexpected_keys)} 个，示例: {unexpected_keys[:10]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 TFA-Conformer 说话人识别模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_experiment.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    device = torch.device(
        "cuda"
        if cfg.runtime.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[Runtime] device={device}")

    print("正在扫描数据集目录，请稍候...", flush=True)
    samples = scan_speaker_samples(cfg.data)
    print("扫描完成！正在构建 DataLoader...", flush=True)

    print("正在构建说话人标签映射...", flush=True)
    label_map = build_label_map(samples)
    print(f"标签映射完成，共 {len(label_map)} 位说话人。", flush=True)

    print("正在按说话人划分测试集...", flush=True)
    _, _, test_samples = split_samples_per_speaker(
        samples=samples,
        ratios=cfg.data.split_ratio,
        seed=cfg.data.split_seed,
    )
    print("数据划分完成。", flush=True)

    print("正在初始化特征提取器并构建测试集 DataLoader...", flush=True)
    audio_extractor = AudioFeatureExtractor(data_cfg=cfg.data, feature_cfg=cfg.feature)
    video_extractor = VideoROIExtractor(data_cfg=cfg.data)
    test_ds = SpeakerFeatureDataset(
        samples=test_samples,
        label_map=label_map,
        audio_extractor=audio_extractor,
        video_extractor=video_extractor,
        training=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=bool(cfg.data.pin_memory and torch.cuda.is_available()),
        collate_fn=speaker_batch_collate,
    )
    print("测试集 DataLoader 构建完成。", flush=True)

    cfg.model.num_speakers = len(label_map)
    model = build_speaker_model(cfg.model).to(device)
    model_backbone = str(getattr(model, "backbone_type", cfg.model.backbone_type))
    model_params = count_trainable_parameters(model)
    param_match_info = getattr(model, "param_match_info", None)
    print(
        f"[Model] backbone_type={model_backbone} "
        f"params={model_params} ({model_params / 1_000_000:.2f}M)"
    )
    if isinstance(param_match_info, dict) and bool(param_match_info.get("enabled", False)):
        print(
            "[Model] baseline_param_match "
            f"target={param_match_info.get('target_params')} "
            f"selected={param_match_info.get('selected_params')} "
            f"diff={param_match_info.get('abs_diff')} "
            f"width={param_match_info.get('selected_width')}"
        )
    print(
        f"[Model] use_audio_branch={bool(getattr(model, 'use_audio_branch', True))} "
        f"use_video_branch={bool(getattr(model, 'use_video_branch', False))} "
        f"use_conformer_conv={bool(getattr(model, 'use_conformer_conv', False))} "
        f"use_balance_se={bool(getattr(model, 'use_balance_se', False))} "
        f"use_tfa_pooling={bool(getattr(model, 'use_tfa_pooling', False))}"
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    load_model_state_dict_flexible(
        model=model,
        state_dict=ckpt["model_state_dict"],
        source=args.checkpoint,
    )
    result = run_one_epoch(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=len(label_map),
        train_cfg=cfg.train,
        train_mode=False,
        epoch_idx=0,
    )
    print(
        "[Eval] "
        f"loss={result.loss:.4f} "
        f"acc={result.accuracy:.4f} "
        f"precision={result.precision:.4f} "
        f"recall={result.recall:.4f} "
        f"f1={result.f1:.4f}"
    )


if __name__ == "__main__":
    main()

