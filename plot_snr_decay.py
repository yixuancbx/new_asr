from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt

METRIC_CHOICES = ("accuracy", "precision", "recall", "f1", "loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制不同模型的 SNR 性能衰减曲线")
    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="一个或多个 evaluate.py 导出的 CSV 文件路径",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=METRIC_CHOICES,
        help="绘图指标",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/snr_decay_curve.png",
        help="输出图像路径",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Performance Degradation under MUSAN Noise",
        help="图标题",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="输出图像 DPI",
    )
    return parser.parse_args()


def _parse_float(text: str, field_name: str, row: Dict[str, str], source: Path) -> float:
    try:
        return float(text)
    except Exception as exc:
        raise ValueError(f"{source} 中字段 {field_name} 解析失败: {row}") from exc


def _parse_snr_db(row: Dict[str, str]) -> float | None:
    raw_snr = str(row.get("snr_db", "")).strip()
    if raw_snr:
        return float(raw_snr)
    label = str(row.get("snr_label", "")).strip().lower()
    if label in {"clean", "none", "no_noise", "inf"}:
        return None
    if label.endswith("db"):
        label = label[:-2]
    if not label:
        return None
    return float(label)


def load_curve_points(csv_files: List[str], metric: str):
    points = defaultdict(lambda: defaultdict(list))
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = str(row.get("model_name", "")).strip() or csv_path.stem
                metric_text = str(row.get(metric, "")).strip()
                if not metric_text:
                    continue
                metric_val = _parse_float(metric_text, metric, row, csv_path)
                snr_db = _parse_snr_db(row)
                points[model_name][snr_db].append(metric_val)
    return points


def main() -> None:
    args = parse_args()
    metric = str(args.metric).strip().lower()
    if metric not in METRIC_CHOICES:
        raise ValueError(f"不支持的 metric: {metric}")

    points = load_curve_points(args.csv_files, metric=metric)
    if not points:
        raise RuntimeError("没有读取到可绘图的数据")

    all_snr_values = set()
    has_clean = False
    for model_map in points.values():
        for snr_db in model_map.keys():
            if snr_db is None:
                has_clean = True
            else:
                all_snr_values.add(float(snr_db))

    ordered_snr: List[float | None] = []
    if has_clean:
        ordered_snr.append(None)
    ordered_snr.extend(sorted(all_snr_values, reverse=True))
    if not ordered_snr:
        raise RuntimeError("未检测到有效的 SNR 数据点")

    x = list(range(len(ordered_snr)))
    x_labels = ["clean" if snr is None else f"{snr:g}" for snr in ordered_snr]

    plt.figure(figsize=(9, 5))
    for model_name in sorted(points.keys()):
        model_map = points[model_name]
        y = []
        for snr_db in ordered_snr:
            values = model_map.get(snr_db, [])
            y.append(mean(values) if values else float("nan"))
        plt.plot(x, y, marker="o", linewidth=2, label=model_name)

    plt.xticks(x, x_labels)
    plt.xlabel("SNR (dB, clean/high -> low)")
    plt.ylabel(metric.upper())
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.35)
    if metric != "loss":
        plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=max(80, int(args.dpi)))
    print(f"[Done] 图像已保存: {output_path}")


if __name__ == "__main__":
    main()
