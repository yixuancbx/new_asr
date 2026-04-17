from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


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
        half_step: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.res2 = Res2Conv2d(channels=channels, scale=scale, dropout=dropout)
        self.se = SELayer2d(channels=channels, reduction=reduction)
        self.half_step = half_step
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.half_step * self.se(self.res2(x)))


class HybridFeatureEncoder(nn.Module):
    """
    Frame-level encoder:
    Conv-BN-ReLU -> SE-Res2Block -> SE-Res2Block -> SE-Res2Block.
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
        self.se1 = SERes2Block(channels=channels, scale=scale, dropout=dropout)
        self.se2 = SERes2Block(channels=channels, scale=scale, dropout=dropout)
        self.se3 = SERes2Block(channels=channels, scale=scale, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.se1(x)
        x = self.se2(x)
        x = self.se3(x)
        return x


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
    Serial flow block:
    MHSA(global) -> TF-attention(0.5 residual) ->
    PW-DW Conv(local, 0.5 residual) -> single FFN.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_t: int = 17,
        conv_kernel_f: int = 3,
        dropout: float = 0.1,
        use_tfa_pooling: bool = True,
        use_conformer_conv: bool = True,
    ) -> None:
        super().__init__()
        self.norm_attn = LayerNorm2d(channels)
        self.norm_tfa = LayerNorm2d(channels)
        self.norm_conv = LayerNorm2d(channels)
        self.norm_ffn = LayerNorm2d(channels)
        self.out_norm = LayerNorm2d(channels)
        self.use_tfa_pooling = bool(use_tfa_pooling)
        self.use_conformer_conv = bool(use_conformer_conv)

        self.attn = TemporalSelfAttention(channels=channels, num_heads=num_heads, dropout=dropout)
        self.tfa_pool = (
            TimeFrequencyAttentionPooling(
                branch_channels=32,
                kernel_size=7,
                dilation=3,
            )
            if self.use_tfa_pooling
            else None
        )
        self.conv = (
            PointDepthwiseSeparableConv2d(
                channels=channels,
                kernel_t=conv_kernel_t,
                kernel_f=conv_kernel_f,
                dropout=dropout,
            )
            if self.use_conformer_conv
            else None
        )
        self.ffn = FFN2d(channels=channels, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        if self.tfa_pool is not None:
            x = x + 0.5 * self.tfa_pool(self.norm_tfa(x))[0]
        if self.conv is not None:
            x = x + 0.5 * self.conv(self.norm_conv(x))
        x = x + self.ffn(self.norm_ffn(x))
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
        use_tfa_pooling: bool = True,
        use_conformer_conv: bool = True,
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
                    use_tfa_pooling=use_tfa_pooling,
                    use_conformer_conv=use_conformer_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(num_blocks))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        scale_feats: List[torch.Tensor] = []
        for block in self.blocks:
            # 开启梯度检查点，牺牲少量计算，换取更低显存占用
            if self.training:
                x = cp.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            scale_feats.append(x)

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        # 使用原地累加，降低多尺度融合阶段的峰值显存占用
        fused = torch.zeros_like(scale_feats[0])
        for weight, feat in zip(scale_weights, scale_feats):
            fused += weight * feat
        return fused, scale_feats, scale_weights


class CompressionExcitationBalance(nn.Module):
    """Compression-excitation temporal balance module."""

    def __init__(self, channels: int, reduction: int = 8, use_se: bool = True) -> None:
        super().__init__()
        self.use_se = bool(use_se)
        if self.use_se:
            hidden = max(8, channels // reduction)
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels)
        else:
            self.fc1 = None
            self.fc2 = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_se:
            gate = x.new_ones((x.size(0), x.size(1)))
            return x, gate
        h = x.mean(dim=3)  # [B, C, T]
        mu = h.mean(dim=2)  # [B, C]
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(mu), inplace=True)))  # [B, C]
        return x * gate[:, :, None, None], gate


class ClassifierSEBlock(nn.Module):
    """Extra SE block before classifier head."""

    def __init__(self, channels: int, reduction: int = 8, use_se: bool = True) -> None:
        super().__init__()
        self.use_se = bool(use_se)
        if self.use_se:
            hidden = max(8, channels // reduction)
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels)
        else:
            self.fc1 = None
            self.fc2 = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_se:
            gate = x.new_ones((x.size(0), x.size(1)))
            return x, gate
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3))), inplace=True)))
        return x * gate[:, :, None, None], gate


def _build_resnet18_backbone() -> nn.Module:
    try:
        tv_models = importlib.import_module("torchvision.models")
    except Exception as exc:
        raise ImportError(
            "视频分支依赖 torchvision，请安装：pip install torchvision"
        ) from exc
    try:
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        # 兼容旧版 torchvision（<0.13）的 pretrained 参数接口
        backbone = tv_models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    return backbone


class VideoAttentionAggregator(nn.Module):
    """Attention-based temporal aggregation for frame-wise video features."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        score = self.score(x).squeeze(-1)  # [B, T]
        if mask is not None:
            score = score.masked_fill(mask <= 0, -1e4)
        attn = torch.softmax(score, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class AdaptiveModalFusion(nn.Module):
    """Adaptive fusion gate between audio/video embeddings."""

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 256,
        audio_prior: float = 0.8,
    ) -> None:
        super().__init__()
        if not (0.0 < audio_prior < 1.0):
            raise ValueError("audio_prior 必须在 (0, 1) 区间内")

        self.gate_feature = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gate_proj = nn.Linear(hidden_dim, emb_dim)

        # 让 alpha 在训练初期更偏向音频分支，减少噪声视频对融合结果的扰动。
        safe_prior = min(max(float(audio_prior), 1e-4), 1.0 - 1e-4)
        prior_bias = math.log(safe_prior / (1.0 - safe_prior))
        nn.init.constant_(self.gate_proj.bias, prior_bias)

    def forward(self, audio_emb: torch.Tensor, video_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gate_feature(torch.cat([audio_emb, video_emb], dim=1))
        alpha = torch.sigmoid(self.gate_proj(hidden))
        fused = alpha * audio_emb + (1.0 - alpha) * video_emb
        return fused, alpha


@dataclass
class ModelConfig:
    # 主模型 / baseline 选择：
    # tfa_multiscale_conformer | ecapa_tdnn | resnet_xvector | mfa_conformer
    backbone_type: str = "tfa_multiscale_conformer"
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
    classifier_se_reduction: int = 8
    bottleneck_dim: int = 512
    embedding_dim: int = 1024
    video_backbone_dim: int = 512
    fusion_hidden_dim: int = 256
    fusion_audio_prior: float = 0.8
    use_audio_branch: bool = True
    use_video_branch: bool = True
    use_attention_aggregation: bool = True
    use_adaptive_fusion: bool = True
    use_conformer_conv: bool = True
    use_balance_se: bool = True
    use_tfa_pooling: bool = True
    dropout: float = 0.1
    # baseline 参数对齐开关：默认按当前 TFA 模型参数量自动搜索最接近配置
    auto_match_baseline_params: bool = True
    # >0 时强制使用该参数预算；<=0 时自动以当前 TFA 配置为预算
    baseline_target_params: int = 0
    baseline_ecapa_channels: int = 512
    baseline_resnet_base_channels: int = 48
    baseline_mfa_conformer_dim: int = 256
    baseline_mfa_num_blocks: int = 6


class TFAMultiScaleConformerSpeakerNet(nn.Module):
    """
    Audio-video multi-modal SID model.
    - Audio: SE-Res2 frame encoder + serial-flow TFA + SE classifier head
    - Video: ResNet18 ROI encoder + attention aggregation
    - Fusion: adaptive gate fusion (supports ablation by config switches)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_audio_branch = bool(cfg.use_audio_branch)
        self.use_video_branch = bool(cfg.use_video_branch)
        if not self.use_audio_branch and not self.use_video_branch:
            raise ValueError("至少需要启用一个模态分支：audio 或 video")

        if self.use_audio_branch:
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
                use_tfa_pooling=cfg.use_tfa_pooling,
                use_conformer_conv=cfg.use_conformer_conv,
            )
            self.ce_balance = CompressionExcitationBalance(
                channels=cfg.feature_channels,
                reduction=cfg.ce_reduction,
                use_se=cfg.use_balance_se,
            )
            self.classifier_se = ClassifierSEBlock(
                channels=cfg.feature_channels,
                reduction=cfg.classifier_se_reduction,
            )
            self.bottleneck_proj = nn.Linear(cfg.feature_channels, cfg.bottleneck_dim)
            self.emb_proj = nn.Linear(cfg.bottleneck_dim, cfg.embedding_dim)
        else:
            self.feature_encoder = None
            self.tfa_encoder = None
            self.ce_balance = None
            self.classifier_se = None
            self.bottleneck_proj = None
            self.emb_proj = None

        if self.use_video_branch:
            self.video_backbone = _build_resnet18_backbone()
            with torch.no_grad():
                backbone_dim = int(self.video_backbone(torch.zeros(1, 3, 112, 112)).shape[1])
            self.video_aggregator = VideoAttentionAggregator(in_dim=backbone_dim)
            self.video_proj = nn.Linear(backbone_dim, cfg.embedding_dim)
        else:
            self.video_backbone = None
            self.video_aggregator = None
            self.video_proj = None

        if self.use_audio_branch and self.use_video_branch:
            self.modal_fusion = AdaptiveModalFusion(
                emb_dim=cfg.embedding_dim,
                hidden_dim=cfg.fusion_hidden_dim,
                audio_prior=cfg.fusion_audio_prior,
            )
        else:
            self.modal_fusion = None

        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_speakers)

    @staticmethod
    def _stats_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))  # [B, C]

    def _encode_audio(self, audio_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_audio_branch:
            raise RuntimeError("audio 分支未启用")
        if audio_feat.dim() == 3:
            x = audio_feat.unsqueeze(1)
        elif audio_feat.dim() == 4:
            x = audio_feat
        else:
            raise ValueError("audio 输入必须是 [B, T, F] 或 [B, 1, T, F]")
        if x.size(1) != self.cfg.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.cfg.in_channels}, got {x.size(1)}."
            )

        x = self.feature_encoder(x)
        x, scales, scale_weights = self.tfa_encoder(x)
        x, ce_gate = self.ce_balance(x)
        x, cls_se_gate = self.classifier_se(x)

        pooled = self._stats_pool(x)
        compressed = self.bottleneck_proj(pooled)
        compressed = F.normalize(compressed, p=2, dim=1)
        audio_emb = self.emb_proj(compressed)
        audio_emb = F.normalize(audio_emb, p=2, dim=1)
        aux = {
            "scale_weights": scale_weights,
            "ce_gate": ce_gate,
            "classifier_se_gate": cls_se_gate,
            "final_feature_map": x,
            "multi_scale_features": torch.stack(scales, dim=1),
            "compressed_embedding": compressed,
        }
        return audio_emb, aux

    def _encode_video(
        self,
        video_feat: torch.Tensor,
        video_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_video_branch:
            raise RuntimeError("video 分支未启用")
        if video_feat.dim() == 4:
            video_feat = video_feat.unsqueeze(1)
        if video_feat.dim() != 5:
            raise ValueError("video 输入必须是 [B, T, 3, H, W] 或 [B, 3, H, W]")
        if video_mask is not None:
            video_mask = video_mask.float()

        bsz, num_frames, channels, h, w = video_feat.shape
        if channels != 3:
            raise ValueError(f"video 通道必须为 3，当前为 {channels}")

        flat = video_feat.reshape(bsz * num_frames, channels, h, w)
        frame_features = self.video_backbone(flat).view(bsz, num_frames, -1)

        if self.cfg.use_attention_aggregation and self.video_aggregator is not None:
            temporal_mask = None
            if video_mask is not None:
                temporal_mask = video_mask[:, None].expand(-1, num_frames) > 0
            pooled, attn_weights = self.video_aggregator(frame_features, mask=temporal_mask)
        else:
            pooled = frame_features.mean(dim=1)
            attn_weights = frame_features.new_full(
                (bsz, num_frames),
                fill_value=1.0 / max(num_frames, 1),
            )

        video_emb = self.video_proj(pooled)
        video_emb = F.normalize(video_emb, p=2, dim=1)
        if video_mask is not None:
            video_emb = video_emb * video_mask.unsqueeze(1)
        return video_emb, attn_weights

    def _fuse_modalities(
        self,
        audio_emb: torch.Tensor | None,
        video_emb: torch.Tensor | None,
        video_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio_emb is None and video_emb is None:
            raise ValueError("audio/video 至少输入一个模态")

        if audio_emb is None:
            alpha = torch.zeros_like(video_emb)
            return video_emb, alpha
        if video_emb is None:
            alpha = torch.ones_like(audio_emb)
            return audio_emb, alpha

        if self.cfg.use_adaptive_fusion and self.modal_fusion is not None:
            fused, alpha = self.modal_fusion(audio_emb, video_emb)
        else:
            alpha = torch.full_like(audio_emb, 0.5)
            fused = 0.5 * (audio_emb + video_emb)

        if video_mask is not None:
            valid = (video_mask > 0).float().unsqueeze(1)
            fused = valid * fused + (1.0 - valid) * audio_emb
            alpha = valid * alpha + (1.0 - valid) * torch.ones_like(alpha)
        return fused, alpha

    def forward(self, feat) -> Dict[str, torch.Tensor]:
        audio_feat = None
        video_feat = None
        video_mask = None
        if isinstance(feat, dict):
            audio_feat = feat.get("audio")
            video_feat = feat.get("video")
            video_mask = feat.get("video_mask")
        else:
            audio_feat = feat

        audio_emb = None
        audio_aux: Dict[str, torch.Tensor] = {}
        if self.use_audio_branch and audio_feat is not None:
            audio_emb, audio_aux = self._encode_audio(audio_feat)

        video_emb = None
        frame_attn = None
        if self.use_video_branch and video_feat is not None:
            video_emb, frame_attn = self._encode_video(video_feat, video_mask=video_mask)

        fused_embedding, fusion_alpha = self._fuse_modalities(
            audio_emb,
            video_emb,
            video_mask=video_mask,
        )
        logits = self.classifier(fused_embedding)
        cosine_logits = F.linear(
            fused_embedding,
            F.normalize(self.classifier.weight, p=2, dim=1),
        )
        probabilities = torch.softmax(logits, dim=1)

        empty = fused_embedding.new_empty(0)
        return {
            "logits": logits,
            "cosine_logits": cosine_logits,
            "probabilities": probabilities,
            "embedding": fused_embedding,
            "audio_embedding": audio_emb if audio_emb is not None else empty,
            "video_embedding": video_emb if video_emb is not None else empty,
            "fusion_alpha": fusion_alpha,
            "video_frame_attention": frame_attn if frame_attn is not None else empty,
            "compressed_embedding": audio_aux.get("compressed_embedding", empty),
            "scale_weights": audio_aux.get("scale_weights", empty),
            "ce_gate": audio_aux.get("ce_gate", empty),
            "classifier_se_gate": audio_aux.get("classifier_se_gate", empty),
            "final_feature_map": audio_aux.get("final_feature_map", empty),
            "multi_scale_features": audio_aux.get("multi_scale_features", empty),
        }


def speaker_ce_loss(outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(outputs["logits"], targets)

