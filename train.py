from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfa_conformer_sid.config import ProjectConfig, dump_yaml_config, load_yaml_config
from tfa_conformer_sid.data import (
    AudioFeatureExtractor,
    SpeakerFeatureDataset,
    build_label_map,
    scan_speaker_samples,
    speaker_batch_collate,
    split_samples_per_speaker,
)
from tfa_conformer_sid.engine import run_one_epoch
from tfa_conformer_sid.models import TFAMultiScaleConformerSpeakerNet
from tfa_conformer_sid.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFA-Conformer 多尺度说话人识别训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_experiment.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument("--run-name", type=str, default="", help="自定义运行名称")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的 checkpoint 路径")
    return parser.parse_args()


def prepare_run_dir(cfg: ProjectConfig, run_name: str) -> Path:
    output_root = Path(cfg.experiment.output_dir)
    if run_name.strip():
        folder = run_name.strip()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{cfg.experiment.name}_{ts}"
    run_dir = output_root / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_dataloaders(cfg: ProjectConfig) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    print("正在扫描数据集目录，请稍候...", flush=True)
    samples = scan_speaker_samples(cfg.data)
    print("扫描完成！正在构建 DataLoader...", flush=True)

    print("正在构建说话人标签映射...", flush=True)
    label_map = build_label_map(samples)
    print(f"标签映射完成，共 {len(label_map)} 位说话人。", flush=True)

    print("正在按说话人划分训练/验证/测试集...", flush=True)
    train_samples, val_samples, test_samples = split_samples_per_speaker(
        samples=samples,
        ratios=cfg.data.split_ratio,
        seed=cfg.data.split_seed,
    )
    print("数据划分完成。", flush=True)

    print("正在初始化特征提取器...", flush=True)
    extractor = AudioFeatureExtractor(data_cfg=cfg.data, feature_cfg=cfg.feature)
    print("特征提取器初始化完成。", flush=True)

    print("正在构建 Dataset 对象...", flush=True)
    train_ds = SpeakerFeatureDataset(
        samples=train_samples, label_map=label_map, extractor=extractor, training=True
    )
    val_ds = SpeakerFeatureDataset(
        samples=val_samples, label_map=label_map, extractor=extractor, training=False
    )
    test_ds = SpeakerFeatureDataset(
        samples=test_samples, label_map=label_map, extractor=extractor, training=False
    )
    print("Dataset 构建完成。", flush=True)

    common = {
        "num_workers": cfg.data.num_workers,
        "pin_memory": bool(cfg.data.pin_memory and torch.cuda.is_available()),
        "collate_fn": speaker_batch_collate,
    }
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.data.train_batch_size,
            shuffle=True,
            drop_last=False,
            **common,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.data.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.data.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
    print("DataLoader 构建完成。", flush=True)
    print(
        f"[Data] 总样本={len(samples)} 训练={len(train_ds)} 验证={len(val_ds)} 测试={len(test_ds)} "
        f"说话人数={len(label_map)}"
    )
    return loaders, label_map


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: ProjectConfig,
    label_map: Dict[str, int],
    best_val_f1: float,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": cfg.to_dict(),
        "label_map": label_map,
        "best_val_f1": best_val_f1,
    }
    torch.save(state, str(path))


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed_everything(cfg.runtime.seed)

    device = torch.device(
        "cuda"
        if cfg.runtime.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[Runtime] device={device}")

    run_dir = prepare_run_dir(cfg, args.run_name)
    dump_yaml_config(cfg, run_dir / "resolved_config.yaml")
    print(f"[Runtime] run_dir={run_dir}")

    loaders, label_map = build_dataloaders(cfg)
    num_classes = len(label_map)
    if cfg.model.num_speakers != num_classes:
        print(
            f"[Model] num_speakers 从 {cfg.model.num_speakers} 自动调整为数据集说话人数 {num_classes}"
        )
        cfg.model.num_speakers = num_classes
        dump_yaml_config(cfg, run_dir / "resolved_config.yaml")

    model = TFAMultiScaleConformerSpeakerNet(cfg.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.train.scheduler_step_size,
        gamma=cfg.train.scheduler_gamma,
    )
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if (cfg.train.amp and device.type == "cuda")
        else None
    )

    history = []
    best_val_f1 = -1.0
    start_epoch = 1

    if args.resume.strip():
        ckpt = torch.load(args.resume.strip(), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Runtime] 从 checkpoint 恢复训练: {args.resume.strip()}, start_epoch={start_epoch}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_result = run_one_epoch(
            model=model,
            loader=loaders["train"],
            device=device,
            num_classes=num_classes,
            train_cfg=cfg.train,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_mode=True,
            epoch_idx=epoch,
        )
        val_result = run_one_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            num_classes=num_classes,
            train_cfg=cfg.train,
            train_mode=False,
            epoch_idx=epoch,
        )

        item = {
            "epoch": epoch,
            "train": train_result.to_dict(),
            "val": val_result.to_dict(),
        }
        history.append(item)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.4f} "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f} "
            f"val_f1={val_result.f1:.4f}"
        )

        if val_result.f1 > best_val_f1:
            best_val_f1 = val_result.f1
            save_checkpoint(
                path=run_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cfg=cfg,
                label_map=label_map,
                best_val_f1=best_val_f1,
            )
            print(f"[Best] 更新最优模型，val_f1={best_val_f1:.4f}")

        if cfg.train.save_every_epoch:
            save_checkpoint(
                path=run_dir / f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cfg=cfg,
                label_map=label_map,
                best_val_f1=best_val_f1,
            )

        save_checkpoint(
            path=run_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            label_map=label_map,
            best_val_f1=best_val_f1,
        )

    best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_result = run_one_epoch(
        model=model,
        loader=loaders["test"],
        device=device,
        num_classes=num_classes,
        train_cfg=cfg.train,
        train_mode=False,
        epoch_idx=cfg.train.epochs,
    )
    print(
        "[Test] "
        f"loss={test_result.loss:.4f} "
        f"acc={test_result.accuracy:.4f} "
        f"precision={test_result.precision:.4f} "
        f"recall={test_result.recall:.4f} "
        f"f1={test_result.f1:.4f}"
    )

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with (run_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"[Done] 训练结束，日志与模型已保存到: {run_dir}")


if __name__ == "__main__":
    main()

