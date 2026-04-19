from __future__ import annotations

import argparse
import copy
import csv
import random
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
from tfa_conformer_sid.utils import seed_everything

CSV_FIELDS = [
    "model_name",
    "checkpoint",
    "snr_label",
    "snr_db",
    "seed",
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path


@dataclass(frozen=True)
class NoiseSpec:
    label: str
    snr_db: float | None
    enable_musan: bool


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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="单模型 checkpoint 路径（兼容旧用法）",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help='多模型 checkpoint 列表，支持 "模型名=路径" 或直接给路径',
    )
    parser.add_argument(
        "--snr-list",
        type=str,
        default="clean,20,15,10,5,0",
        help="逗号分隔的 SNR 列表，可包含 clean",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=1.0,
        help="评估阶段加噪概率（默认 1.0）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="评估随机种子，默认使用 runtime.seed",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=-1,
        help="评估 DataLoader worker 数；<0 表示沿用配置文件",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="runs/eval_snr_results.csv",
        help="总评估结果 CSV 输出路径",
    )
    parser.add_argument(
        "--per-model-csv-dir",
        type=str,
        default="",
        help="可选：为每个模型单独导出 CSV 的目录",
    )
    return parser.parse_args()


def _safe_model_filename(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")
    return safe or "model"


def _parse_checkpoint_specs(args: argparse.Namespace) -> List[ModelSpec]:
    tokens: List[str] = list(args.checkpoints or [])
    if args.checkpoint.strip():
        tokens.append(args.checkpoint.strip())
    if not tokens:
        raise ValueError("请至少提供一个 checkpoint（--checkpoint 或 --checkpoints）")

    specs: List[ModelSpec] = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        if "=" in text:
            model_name, path_text = text.split("=", 1)
            model_name = model_name.strip()
            path_text = path_text.strip()
        else:
            path_text = text
            path_obj = Path(path_text)
            model_name = path_obj.parent.name or path_obj.stem
        if not model_name:
            raise ValueError(f"模型名解析失败，请检查参数: {token}")

        ckpt_path = Path(path_text)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
        specs.append(ModelSpec(name=model_name, checkpoint=ckpt_path))

    if not specs:
        raise ValueError("未解析到有效的 checkpoint")
    return specs


def _parse_snr_specs(raw: str) -> List[NoiseSpec]:
    tokens = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not tokens:
        raise ValueError("--snr-list 不能为空")

    specs: List[NoiseSpec] = []
    for token in tokens:
        lower = token.lower()
        if lower in {"clean", "none", "no_noise", "inf"}:
            specs.append(NoiseSpec(label="clean", snr_db=None, enable_musan=False))
            continue
        try:
            snr = float(token)
        except ValueError as exc:
            raise ValueError(f"SNR 解析失败: {token}") from exc
        specs.append(NoiseSpec(label=f"{snr:g}dB", snr_db=snr, enable_musan=True))
    return specs


def _merge_model_config_from_checkpoint(
    cfg_model,
    checkpoint_obj: Dict[str, Any],
):
    out = copy.deepcopy(cfg_model)
    raw_cfg = checkpoint_obj.get("config")
    if not isinstance(raw_cfg, dict):
        return out
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, dict):
        return out
    for key, value in raw_model_cfg.items():
        if hasattr(out, key):
            setattr(out, key, value)
    return out


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)


def _snr_eval_seed(base_seed: int, snr_label: str) -> int:
    offset = 0
    for idx, ch in enumerate(str(snr_label)):
        offset += (idx + 1) * ord(ch)
    return int(base_seed) + offset


def _build_test_loader(
    cfg,
    test_samples,
    label_map,
    noise_spec: NoiseSpec,
    eval_seed: int,
    eval_num_workers: int,
    noise_prob: float,
):
    audio_extractor = AudioFeatureExtractor(data_cfg=cfg.data, feature_cfg=cfg.feature)
    audio_extractor.configure_eval_musan(
        enable=noise_spec.enable_musan,
        snr_db=noise_spec.snr_db,
        prob=noise_prob,
    )
    if noise_spec.enable_musan:
        if not audio_extractor.musan_paths:
            print(
                f"[Warn] snr={noise_spec.label} 但未扫描到 MUSAN 文件，"
                "本轮将退化为 clean 评估。"
            )
        elif noise_prob <= 0.0:
            print(
                f"[Warn] snr={noise_spec.label} 且 noise_prob<=0，"
                "本轮不会实际加噪。"
            )
    video_extractor = VideoROIExtractor(data_cfg=cfg.data)
    test_ds = SpeakerFeatureDataset(
        samples=test_samples,
        label_map=label_map,
        audio_extractor=audio_extractor,
        video_extractor=video_extractor,
        training=False,
    )

    num_workers = int(cfg.data.num_workers) if eval_num_workers < 0 else int(eval_num_workers)
    generator = torch.Generator()
    generator.manual_seed(int(eval_seed))
    worker_init_fn = None
    if num_workers > 0:
        worker_init_fn = partial(_seed_worker, base_seed=int(eval_seed))

    return DataLoader(
        test_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory and torch.cuda.is_available()),
        collate_fn=speaker_batch_collate,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    model_specs = _parse_checkpoint_specs(args)
    noise_specs = _parse_snr_specs(args.snr_list)
    noise_prob = min(max(float(args.noise_prob), 0.0), 1.0)
    base_seed = int(cfg.runtime.seed if args.seed is None else args.seed)
    seed_everything(base_seed)

    device = torch.device(
        "cuda"
        if cfg.runtime.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[Runtime] device={device}")
    print(f"[Runtime] eval_seed={base_seed}")

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
    print(f"[Data] 测试样本数={len(test_samples)} 说话人数={len(label_map)}")

    all_rows: List[Dict[str, Any]] = []
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for model_spec in model_specs:
        print(f"\n[Model] 正在评估: {model_spec.name}")
        ckpt = torch.load(str(model_spec.checkpoint), map_location=device)
        model_cfg = _merge_model_config_from_checkpoint(cfg.model, ckpt)
        model_cfg.num_speakers = len(label_map)

        model = build_speaker_model(model_cfg).to(device)
        model_backbone = str(getattr(model, "backbone_type", model_cfg.backbone_type))
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

        load_model_state_dict_flexible(
            model=model,
            state_dict=ckpt["model_state_dict"],
            source=str(model_spec.checkpoint),
        )

        model_rows: List[Dict[str, Any]] = []
        for noise_spec in noise_specs:
            eval_seed = _snr_eval_seed(base_seed=base_seed, snr_label=noise_spec.label)
            seed_everything(eval_seed)
            test_loader = _build_test_loader(
                cfg=cfg,
                test_samples=test_samples,
                label_map=label_map,
                noise_spec=noise_spec,
                eval_seed=eval_seed,
                eval_num_workers=int(args.eval_num_workers),
                noise_prob=noise_prob,
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
            row = {
                "model_name": model_spec.name,
                "checkpoint": str(model_spec.checkpoint),
                "snr_label": noise_spec.label,
                "snr_db": "" if noise_spec.snr_db is None else f"{float(noise_spec.snr_db):g}",
                "seed": str(eval_seed),
                "loss": f"{result.loss:.6f}",
                "accuracy": f"{result.accuracy:.6f}",
                "precision": f"{result.precision:.6f}",
                "recall": f"{result.recall:.6f}",
                "f1": f"{result.f1:.6f}",
            }
            all_rows.append(row)
            model_rows.append(row)
            print(
                f"[Eval] model={model_spec.name} snr={noise_spec.label} "
                f"loss={result.loss:.4f} acc={result.accuracy:.4f} "
                f"precision={result.precision:.4f} recall={result.recall:.4f} f1={result.f1:.4f}"
            )

        rows_by_model[model_spec.name] = model_rows

    output_csv = Path(args.output_csv)
    _write_csv(output_csv, all_rows)
    print(f"\n[Done] 总结果已保存: {output_csv}")

    if args.per_model_csv_dir.strip():
        per_model_dir = Path(args.per_model_csv_dir.strip())
        per_model_dir.mkdir(parents=True, exist_ok=True)
        for model_name, rows in rows_by_model.items():
            csv_path = per_model_dir / f"{_safe_model_filename(model_name)}.csv"
            _write_csv(csv_path, rows)
            print(f"[Done] {model_name} 结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
