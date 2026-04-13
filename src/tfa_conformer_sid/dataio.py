from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import DataConfig, FeatureConfig


def _normalize_extensions(extensions: Sequence[str]) -> set[str]:
    normalized = set()
    for ext in extensions:
        ext = str(ext).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def _speaker_seed(base_seed: int, speaker_id: str) -> int:
    return int(base_seed) + sum(ord(ch) for ch in speaker_id)


@dataclass(frozen=True)
class SpeakerSample:
    wav_path: str
    speaker_id: str


class AudioFeatureExtractor:
    def __init__(self, data_cfg: DataConfig, feature_cfg: FeatureConfig) -> None:
        self.data_cfg = data_cfg
        self.feature_cfg = feature_cfg
        self.sample_rate = int(data_cfg.sample_rate)
        self.segment_samples = int(round(float(data_cfg.segment_seconds) * self.sample_rate))
        if self.segment_samples <= 0:
            raise ValueError("data.segment_seconds 必须大于 0")

        win_length = int(round(feature_cfg.win_length_ms * self.sample_rate / 1000.0))
        hop_length = int(round(feature_cfg.hop_length_ms * self.sample_rate / 1000.0))
        if win_length <= 0 or hop_length <= 0:
            raise ValueError("feature.win_length_ms / hop_length_ms 配置非法")

        self.feature_type = str(feature_cfg.type).strip().lower()
        mel_kwargs = {
            "n_fft": int(feature_cfg.n_fft),
            "win_length": win_length,
            "hop_length": hop_length,
            "f_min": float(feature_cfg.f_min),
            "f_max": feature_cfg.f_max,
            "n_mels": int(feature_cfg.n_mels),
        }
        if self.feature_type == "mfcc":
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=int(feature_cfg.n_mfcc),
                melkwargs=mel_kwargs,
            )
        elif self.feature_type in {"log_mel", "logmel", "mel"}:
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=mel_kwargs["n_fft"],
                win_length=mel_kwargs["win_length"],
                hop_length=mel_kwargs["hop_length"],
                f_min=mel_kwargs["f_min"],
                f_max=mel_kwargs["f_max"],
                n_mels=mel_kwargs["n_mels"],
                power=2.0,
            )
        else:
            raise ValueError("feature.type 仅支持: mfcc / log_mel")

        self.cmvn = bool(feature_cfg.cmvn)
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        self.musan_enable = bool(data_cfg.musan_enable)
        self.musan_prob = float(data_cfg.musan_prob)
        self.musan_snr_min_db = float(data_cfg.musan_snr_min_db)
        self.musan_snr_max_db = float(data_cfg.musan_snr_max_db)
        if self.musan_snr_min_db > self.musan_snr_max_db:
            raise ValueError("data.musan_snr_min_db 不能大于 data.musan_snr_max_db")
        self.musan_paths = self._scan_musan_paths() if self.musan_enable else []

    def _scan_musan_paths(self) -> List[Path]:
        extensions = _normalize_extensions(self.data_cfg.musan_extensions)
        paths: List[Path] = []
        for root in self.data_cfg.musan_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for file_path in root_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    paths.append(file_path)
        paths.sort()
        return paths

    def _resample_if_needed(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.sample_rate:
            return wave
        resampler = self._resamplers.get(sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            self._resamplers[sample_rate] = resampler
        return resampler(wave)

    def _load_wave(self, path: str | Path) -> torch.Tensor:
        wave, sample_rate = torchaudio.load(str(path))
        if wave.size(0) > 1:
            wave = wave.mean(dim=0, keepdim=True)
        wave = self._resample_if_needed(wave, int(sample_rate))
        return wave.squeeze(0)

    def _crop_or_pad(self, wave: torch.Tensor, training: bool) -> torch.Tensor:
        target_len = self.segment_samples
        cur_len = int(wave.numel())
        if cur_len == target_len:
            return wave
        if cur_len > target_len:
            max_start = cur_len - target_len
            if training:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2
            return wave[start : start + target_len]
        if cur_len <= 0:
            return torch.zeros(target_len, dtype=wave.dtype)
        return F.pad(wave, (0, target_len - cur_len))

    def _fit_noise_to_target(self, noise: torch.Tensor, target_len: int) -> torch.Tensor:
        if noise.numel() <= 0:
            return torch.zeros(target_len, dtype=noise.dtype)
        if noise.numel() < target_len:
            repeat = (target_len + noise.numel() - 1) // noise.numel()
            noise = noise.repeat(repeat)
        if noise.numel() > target_len:
            start = random.randint(0, int(noise.numel()) - target_len)
            noise = noise[start : start + target_len]
        return noise

    def _apply_musan(self, wave: torch.Tensor, training: bool) -> torch.Tensor:
        if not training or not self.musan_enable:
            return wave
        if not self.musan_paths:
            return wave
        if random.random() > self.musan_prob:
            return wave

        noise_path = random.choice(self.musan_paths)
        try:
            noise = self._load_wave(noise_path)
        except Exception:
            return wave

        noise = self._fit_noise_to_target(noise, target_len=int(wave.numel()))
        snr = random.uniform(self.musan_snr_min_db, self.musan_snr_max_db)

        speech_rms = wave.pow(2).mean().sqrt().clamp_min(1e-6)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-6)
        noise_scale = speech_rms / (10 ** (snr / 20.0) * noise_rms)
        mixed = wave + noise * noise_scale
        return mixed.clamp(-1.0, 1.0)

    def _to_feature(self, wave: torch.Tensor) -> torch.Tensor:
        wave = wave.unsqueeze(0)
        if self.feature_type == "mfcc":
            feature = self.transform(wave).squeeze(0)  # [F, T]
        else:
            mel = self.transform(wave).squeeze(0)  # [F, T]
            feature = torch.log(mel + 1e-6)

        if self.cmvn:
            mean = feature.mean(dim=1, keepdim=True)
            std = feature.std(dim=1, keepdim=True).clamp_min(1e-5)
            feature = (feature - mean) / std

        return feature.transpose(0, 1).contiguous().float()  # [T, F]

    def __call__(self, wav_path: str | Path, training: bool) -> torch.Tensor:
        wave = self._load_wave(wav_path)
        wave = self._crop_or_pad(wave, training=training)
        wave = self._apply_musan(wave, training=training)
        return self._to_feature(wave)


def scan_speaker_samples(data_cfg: DataConfig) -> List[SpeakerSample]:
    extensions = _normalize_extensions(data_cfg.extensions)
    if not extensions:
        raise ValueError("data.extensions 不能为空")

    samples_by_speaker: Dict[str, List[SpeakerSample]] = defaultdict(list)
    speaker_level = int(data_cfg.speaker_level)
    if speaker_level < 1:
        raise ValueError("data.speaker_level 必须 >= 1")

    for root in data_cfg.roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        for wav_path in root_path.rglob("*"):
            if not wav_path.is_file() or wav_path.suffix.lower() not in extensions:
                continue
            try:
                rel_parts = wav_path.relative_to(root_path).parts
            except ValueError:
                continue
            if len(rel_parts) <= speaker_level:
                continue

            speaker_id = rel_parts[-(speaker_level + 1)]
            samples_by_speaker[speaker_id].append(
                SpeakerSample(wav_path=str(wav_path), speaker_id=speaker_id)
            )

    if not samples_by_speaker:
        raise RuntimeError("未扫描到任何音频样本，请检查 data.roots / data.extensions 配置")

    max_samples = int(data_cfg.max_samples_per_speaker)
    rng = random.Random(int(data_cfg.speaker_sample_seed))
    selected: List[SpeakerSample] = []
    for speaker_id in sorted(samples_by_speaker.keys()):
        items = sorted(samples_by_speaker[speaker_id], key=lambda x: x.wav_path)
        if max_samples > 0 and len(items) > max_samples:
            pick_idx = sorted(rng.sample(range(len(items)), k=max_samples))
            items = [items[i] for i in pick_idx]
        selected.extend(items)
    return selected


def build_label_map(samples: Sequence[SpeakerSample]) -> Dict[str, int]:
    speakers = sorted({sample.speaker_id for sample in samples})
    return {speaker_id: idx for idx, speaker_id in enumerate(speakers)}


def _split_count_per_speaker(
    num_samples: int, ratios: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    if num_samples <= 1:
        return num_samples, 0, 0

    train_ratio, val_ratio, _ = ratios
    train_n = max(1, int(round(num_samples * train_ratio)))
    val_n = int(round(num_samples * val_ratio))

    if train_n + val_n >= num_samples:
        val_n = max(0, num_samples - train_n - 1)
    test_n = num_samples - train_n - val_n

    if test_n == 0 and num_samples > 1:
        if val_n > 0:
            val_n -= 1
        elif train_n > 1:
            train_n -= 1
        test_n = 1

    return train_n, val_n, test_n


def split_samples_per_speaker(
    samples: Sequence[SpeakerSample],
    ratios: Sequence[float],
    seed: int,
) -> Tuple[List[SpeakerSample], List[SpeakerSample], List[SpeakerSample]]:
    if len(ratios) != 3:
        raise ValueError("data.split_ratio 必须包含三个值: [train, val, test]")
    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0:
        raise ValueError("data.split_ratio 总和必须大于 0")
    norm_ratios = tuple(float(r) / ratio_sum for r in ratios)

    grouped: Dict[str, List[SpeakerSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.speaker_id].append(sample)

    train_samples: List[SpeakerSample] = []
    val_samples: List[SpeakerSample] = []
    test_samples: List[SpeakerSample] = []

    for speaker_id in sorted(grouped.keys()):
        speaker_items = list(grouped[speaker_id])
        speaker_rng = random.Random(_speaker_seed(seed, speaker_id))
        speaker_rng.shuffle(speaker_items)

        train_n, val_n, _ = _split_count_per_speaker(len(speaker_items), norm_ratios)
        train_samples.extend(speaker_items[:train_n])
        val_samples.extend(speaker_items[train_n : train_n + val_n])
        test_samples.extend(speaker_items[train_n + val_n :])

    return train_samples, val_samples, test_samples


class SpeakerFeatureDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SpeakerSample],
        label_map: Dict[str, int],
        extractor: AudioFeatureExtractor,
        training: bool,
    ) -> None:
        self.samples = list(samples)
        self.label_map = label_map
        self.extractor = extractor
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        feat = self.extractor(sample.wav_path, training=self.training)
        label = self.label_map[sample.speaker_id]
        return feat, label


def speaker_batch_collate(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    feats, labels = zip(*batch)
    feat_tensor = pad_sequence(feats, batch_first=True)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return feat_tensor, label_tensor
