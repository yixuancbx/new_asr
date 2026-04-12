from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm on channel dimension for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Res2Conv2d(nn.Module):
    """Res2Net-like multi-scale convolution for 2D speech features."""

    def __init__(
        self,
        channels: int,
        scale: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be >= 2")

        self.scale = scale
        self.width = max(1, channels // scale)
        branch_channels = self.width * scale
        padding = kernel_size // 2

        self.pre = nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.width,
                        self.width,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True),
                )
                for _ in range(scale - 1)
            ]
        )
        self.post = nn.Sequential(
            nn.Conv2d(branch_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        splits = torch.split(y, self.width, dim=1)
        outputs: List[torch.Tensor] = [splits[0]]
        for idx, conv in enumerate(self.convs, start=1):
            branch = splits[idx] + outputs[-1]
            outputs.append(conv(branch))

        y = torch.cat(outputs, dim=1)
        y = self.post(y)
        return self.dropout(y)


class SELayer2d(nn.Module):
    """Squeeze-and-excitation on channel axis."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class SFRes2Block(nn.Module):
    """Non-cascaded half-step residual Res2 block (SF-Res2Block style)."""

    def __init__(
        self,
        channels: int,
        scale: int = 6,
        half_step: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.res2 = Res2Conv2d(channels=channels, scale=scale, dropout=dropout)
        self.half_step = half_step
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.half_step * self.res2(x))


class SERes2Block(nn.Module):
    """SE-Res2 block."""

    def __init__(
        self,
        channels: int,
        scale: int = 6,
        reduction: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.res2 = Res2Conv2d(channels=channels, scale=scale, dropout=dropout)
        self.se = SELayer2d(channels=channels, reduction=reduction)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.res2(x)))


class HybridFeatureEncoder(nn.Module):
    """
    Hybrid feature encoder:
    Conv-BN-ReLU -> SF-Res2Block -> SE-Res2Block -> SF-Res2Block
    with dense-like residual bridges.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 512,
        scale: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.sf1 = SFRes2Block(channels=channels, scale=scale, dropout=dropout)
        self.se = SERes2Block(channels=channels, scale=scale, dropout=dropout)
        self.sf2 = SFRes2Block(channels=channels, scale=scale, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.sf1(x0)
        x2 = self.se(x1 + x0)
        x3 = self.sf2(x2 + x1)
        return x3


class DomainSeparableAttention1d(nn.Module):
    """Pointwise + depthwise + pointwise branch for 1D attention."""

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 7,
        dilation: int = 3,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive odd integer")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")

        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeFrequencyAttentionPooling(nn.Module):
    """
    Time-frequency attention pooling:
    average pooling over channel -> temporal/frequency branches -> outer product.
    """

    def __init__(
        self,
        branch_channels: int = 32,
        kernel_size: int = 7,
        dilation: int = 3,
    ) -> None:
        super().__init__()
        self.temporal_branch = DomainSeparableAttention1d(
            channels=branch_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.frequency_branch = DomainSeparableAttention1d(
            channels=branch_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = x.mean(dim=1)  # [B, T, F]
        temporal = pooled.mean(dim=2, keepdim=False).unsqueeze(1)  # [B, 1, T]
        frequency = pooled.mean(dim=1, keepdim=False).unsqueeze(1)  # [B, 1, F]

        t_attn = self.temporal_branch(temporal).squeeze(1)  # [B, T]
        f_attn = self.frequency_branch(frequency).squeeze(1)  # [B, F]

        tf_map = t_attn.unsqueeze(-1) * f_attn.unsqueeze(-2)  # [B, T, F]
        return x * tf_map.unsqueeze(1), tf_map


class FFN2d(nn.Module):
    """Position-wise feed-forward network in 2D form."""

    def __init__(self, channels: int, mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = channels * mult
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSelfAttention(nn.Module):
    """
    Self-attention over time axis.
    Frequency axis is preserved by broadcasting attention output.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, t_steps, f_bins = x.shape
        seq = x.mean(dim=3).permute(0, 2, 1)  # [B, T, C]
        seq_norm = self.norm(seq)
        attn_out, _ = self.attn(seq_norm, seq_norm, seq_norm, need_weights=False)
        attn_out = self.dropout(attn_out)
        return attn_out.permute(0, 2, 1).unsqueeze(-1).expand(bsz, channels, t_steps, f_bins)


class PointDepthwiseSeparableConv2d(nn.Module):
    """Pointwise-depthwise separable convolution block."""

    def __init__(
        self,
        channels: int,
        kernel_t: int = 17,
        kernel_f: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_t, kernel_f),
            padding=(kernel_t // 2, kernel_f // 2),
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)  # GLU
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)
        return self.dropout(x)


class TFAConformerBlock(nn.Module):
    """
    TFA-Conformer block:
    0.5*FFN -> MHSA -> PW-DW Conv -> TF-attention -> 0.5*FFN.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_t: int = 17,
        conv_kernel_f: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_ffn1 = LayerNorm2d(channels)
        self.norm_attn = LayerNorm2d(channels)
        self.norm_conv = LayerNorm2d(channels)
        self.norm_tfa = LayerNorm2d(channels)
        self.norm_ffn2 = LayerNorm2d(channels)
        self.out_norm = LayerNorm2d(channels)

        self.ffn1 = FFN2d(channels=channels, mult=ff_mult, dropout=dropout)
        self.attn = TemporalSelfAttention(channels=channels, num_heads=num_heads, dropout=dropout)
        self.conv = PointDepthwiseSeparableConv2d(
            channels=channels,
            kernel_t=conv_kernel_t,
            kernel_f=conv_kernel_f,
            dropout=dropout,
        )
        self.tfa_pool = TimeFrequencyAttentionPooling(
            branch_channels=32,
            kernel_size=7,
            dilation=3,
        )
        self.ffn2 = FFN2d(channels=channels, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))
        x = x + self.attn(self.norm_attn(x))
        x = x + self.conv(self.norm_conv(x))
        x = x + self.tfa_pool(self.norm_tfa(x))[0]
        x = x + 0.5 * self.ffn2(self.norm_ffn2(x))
        return self.out_norm(x)


class MultiScaleTFAEncoder(nn.Module):
    """Stack TFA-Conformer blocks and fuse intermediate scales."""

    def __init__(
        self,
        channels: int,
        num_blocks: int = 3,
        num_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_t: int = 17,
        conv_kernel_f: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TFAConformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    conv_kernel_t=conv_kernel_t,
                    conv_kernel_f=conv_kernel_f,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(num_blocks))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        scale_feats: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            scale_feats.append(x)

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        fused = sum(weight * feat for weight, feat in zip(scale_weights, scale_feats))
        return fused, scale_feats, scale_weights


class CompressionExcitationBalance(nn.Module):
    """Compression-excitation temporal balance module."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.mean(dim=3)  # [B, C, T]
        mu = h.mean(dim=2)  # [B, C]
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(mu), inplace=True)))  # [B, C]
        return x * gate[:, :, None, None], gate


@dataclass
class ModelConfig:
    num_speakers: int = 855
    in_channels: int = 1
    feature_channels: int = 512
    res2_scale: int = 6
    num_conformer_blocks: int = 3
    num_heads: int = 4
    ff_mult: int = 4
    conv_kernel_t: int = 17
    conv_kernel_f: int = 3
    ce_reduction: int = 8
    embedding_dim: int = 256
    dropout: float = 0.1


class TFAMultiScaleConformerSpeakerNet(nn.Module):
    """
    Multi-scale short-utterance speaker identification model based on TFA-Conformer.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = HybridFeatureEncoder(
            in_channels=cfg.in_channels,
            channels=cfg.feature_channels,
            scale=cfg.res2_scale,
            dropout=cfg.dropout,
        )
        self.tfa_encoder = MultiScaleTFAEncoder(
            channels=cfg.feature_channels,
            num_blocks=cfg.num_conformer_blocks,
            num_heads=cfg.num_heads,
            ff_mult=cfg.ff_mult,
            conv_kernel_t=cfg.conv_kernel_t,
            conv_kernel_f=cfg.conv_kernel_f,
            dropout=cfg.dropout,
        )
        self.ce_balance = CompressionExcitationBalance(
            channels=cfg.feature_channels,
            reduction=cfg.ce_reduction,
        )
        self.emb_proj = nn.LazyLinear(cfg.embedding_dim)
        self.emb_bn = nn.BatchNorm1d(cfg.embedding_dim)
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_speakers)

    @staticmethod
    def _stats_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)  # [B, C, F]

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feat.dim() == 3:
            x = feat.unsqueeze(1)
        elif feat.dim() == 4:
            x = feat
        else:
            raise ValueError("Input must be [B, T, F] or [B, 1, T, F].")

        if x.size(1) != self.cfg.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.cfg.in_channels}, got {x.size(1)}."
            )

        x = self.feature_encoder(x)
        x, scales, scale_weights = self.tfa_encoder(x)
        x, ce_gate = self.ce_balance(x)

        pooled = self._stats_pool(x).flatten(start_dim=1)
        embedding = self.emb_proj(pooled)
        embedding = self.emb_bn(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        cosine_logits = F.linear(embedding, F.normalize(self.classifier.weight, p=2, dim=1))

        return {
            "logits": logits,
            "cosine_logits": cosine_logits,
            "embedding": embedding,
            "scale_weights": scale_weights,
            "ce_gate": ce_gate,
            "final_feature_map": x,
            "multi_scale_features": torch.stack(scales, dim=1),
        }


def speaker_ce_loss(outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(outputs["logits"], targets)

