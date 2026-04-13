from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ModelConfig


@dataclass
class ExperimentConfig:
    name: str = "paper_tfa_conformer_sid"
    output_dir: str = "runs"


@dataclass
class DataConfig:
    roots: List[str] = field(default_factory=lambda: ["data/TIMIT", "data/ST-CMDS"])
    extensions: List[str] = field(default_factory=lambda: [".wav", ".flac"])
    speaker_level: int = 1
    max_samples_per_speaker: int = 6
    speaker_sample_seed: int = 42
    sample_rate: int = 16000
    segment_seconds: float = 2.5
    split_ratio: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])  # 3:1:1
    split_seed: int = 42
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    musan_enable: bool = False
    musan_roots: List[str] = field(default_factory=list)
    musan_extensions: List[str] = field(default_factory=lambda: [".wav", ".flac"])
    musan_prob: float = 0.5
    musan_snr_min_db: float = 5.0
    musan_snr_max_db: float = 20.0
    video_enable: bool = False
    video_roots: List[str] = field(default_factory=list)
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"]
    )
    video_num_frames: int = 8
    video_frame_size: int = 112


@dataclass
class FeatureConfig:
    type: str = "mfcc"  # mfcc | log_mel
    n_mfcc: int = 80
    n_mels: int = 80
    n_fft: int = 256
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    f_min: float = 20.0
    f_max: Optional[float] = 7600.0
    cmvn: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-4
    scheduler_type: str = "cosine"  # cosine | step
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    scheduler_min_lr: float = 1e-6
    scheduler_step_size: int = 650
    scheduler_gamma: float = 0.97
    grad_clip: float = 5.0
    amp: bool = True
    label_smoothing: float = 0.0
    loss_type: str = "arcface"  # ce | arcface | cosface
    loss_margin: float = 0.2
    loss_scale: float = 30.0
    loss_easy_margin: bool = False
    modality_drop_video_prob: float = 0.1
    modality_drop_audio_prob: float = 0.1
    log_interval: int = 20
    save_every_epoch: bool = False


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(dataclass_obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(dataclass_obj, key):
            continue
        setattr(dataclass_obj, key, value)
    return dataclass_obj


def load_yaml_config(path: str | Path) -> ProjectConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "缺少 PyYAML 依赖，请先安装：pip install pyyaml"
        ) from exc

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ProjectConfig()
    if "experiment" in raw:
        cfg.experiment = _merge_dataclass(cfg.experiment, raw["experiment"])
    if "runtime" in raw:
        cfg.runtime = _merge_dataclass(cfg.runtime, raw["runtime"])
    if "data" in raw:
        cfg.data = _merge_dataclass(cfg.data, raw["data"])
    if "feature" in raw:
        cfg.feature = _merge_dataclass(cfg.feature, raw["feature"])
    if "model" in raw:
        cfg.model = _merge_dataclass(cfg.model, raw["model"])
    if "train" in raw:
        cfg.train = _merge_dataclass(cfg.train, raw["train"])

    return cfg


def dump_yaml_config(config: ProjectConfig, path: str | Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "缺少 PyYAML 依赖，请先安装：pip install pyyaml"
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)

