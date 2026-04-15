from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_audio_tensor(feat, expected_channels: int) -> torch.Tensor:
    audio_feat = feat
    if isinstance(feat, dict):
        audio_feat = feat.get("audio")
        if audio_feat is None:
            raise ValueError("输入中缺少 `audio` 特征，baseline 模型无法前向计算")

    if audio_feat.dim() == 3:
        x = audio_feat.unsqueeze(1)  # [B, 1, T, F]
    elif audio_feat.dim() == 4:
        x = audio_feat
    else:
        raise ValueError("audio 输入必须是 [B, T, F] 或 [B, C, T, F]")

    if x.size(1) != expected_channels:
        if expected_channels == 1:
            x = x.mean(dim=1, keepdim=True)
        elif x.size(1) == 1 and expected_channels > 1:
            x = x.repeat(1, expected_channels, 1, 1)
        else:
            raise ValueError(
                f"输入通道不匹配：期望 {expected_channels}，实际 {x.size(1)}"
            )
    return x


class _SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(2)


class _SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class _AttentiveStatsPool1d(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 128, global_context: bool = True) -> None:
        super().__init__()
        self.global_context = bool(global_context)
        attn_in = channels * 3 if self.global_context else channels
        self.tdnn = nn.Conv1d(attn_in, hidden_channels, kernel_size=1)
        self.proj = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if self.global_context:
            mean = x.mean(dim=2, keepdim=True).expand_as(x)
            std = torch.sqrt(
                torch.clamp((x - mean).pow(2).mean(dim=2, keepdim=True), min=1e-5)
            ).expand_as(x)
            attn_in = torch.cat([x, mean, std], dim=1)
        else:
            attn_in = x

        alpha = self.proj(torch.tanh(self.tdnn(attn_in)))
        alpha = torch.softmax(alpha, dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x.pow(2), dim=2) - mean.pow(2)
        std = torch.sqrt(var.clamp(min=1e-5))
        return torch.cat([mean, std], dim=1)


class _AudioOnlySpeakerNet(nn.Module):
    def __init__(self, cfg, backbone_type: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_type = backbone_type
        self.use_audio_branch = True
        self.use_video_branch = False
        self.use_attention_aggregation = False
        self.use_adaptive_fusion = False
        self.use_conformer_conv = False
        self.use_balance_se = False
        self.use_tfa_pooling = False
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_speakers)

    def _build_outputs(
        self,
        embedding: torch.Tensor,
        aux: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        aux = aux or {}
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        cosine_logits = F.linear(
            embedding,
            F.normalize(self.classifier.weight, p=2, dim=1),
        )
        probabilities = torch.softmax(logits, dim=1)
        empty = embedding.new_empty(0)
        return {
            "logits": logits,
            "cosine_logits": cosine_logits,
            "probabilities": probabilities,
            "embedding": embedding,
            "audio_embedding": embedding,
            "video_embedding": aux.get("video_embedding", empty),
            "fusion_alpha": aux.get("fusion_alpha", torch.ones_like(embedding)),
            "video_frame_attention": aux.get("video_frame_attention", empty),
            "compressed_embedding": aux.get("compressed_embedding", empty),
            "scale_weights": aux.get("scale_weights", empty),
            "ce_gate": aux.get("ce_gate", empty),
            "classifier_se_gate": aux.get("classifier_se_gate", empty),
            "final_feature_map": aux.get("final_feature_map", empty),
            "multi_scale_features": aux.get("multi_scale_features", empty),
        }


class _Res2Conv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        scale = max(2, int(scale))
        width = max(1, channels // scale)
        branch_channels = width * scale
        padding = dilation * (kernel_size - 1) // 2
        self.width = width
        self.scale = scale
        self.pre = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.SiLU(inplace=True),
        )
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        width,
                        width,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(width),
                    nn.SiLU(inplace=True),
                )
                for _ in range(scale - 1)
            ]
        )
        self.post = nn.Sequential(
            nn.Conv1d(branch_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        splits = torch.split(y, self.width, dim=1)
        out: List[torch.Tensor] = [splits[0]]
        for idx, conv in enumerate(self.convs, start=1):
            branch = splits[idx] + out[-1]
            out.append(conv(branch))
        y = torch.cat(out, dim=1)
        y = self.post(y)
        return self.dropout(y)


class _SERes2Block1d(nn.Module):
    def __init__(
        self,
        channels: int,
        scale: int,
        dilation: int,
        dropout: float,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
        )
        self.res2 = _Res2Conv1d(
            channels=channels,
            scale=scale,
            kernel_size=3,
            dilation=dilation,
            dropout=dropout,
        )
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = _SEBlock1d(channels, reduction=reduction)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        y = self.res2(y)
        y = self.post(y)
        y = self.se(y)
        return self.act(x + y)


class BaselineECAPASpeakerNet(_AudioOnlySpeakerNet):
    """ECAPA-TDNN 风格 baseline，统一到当前训练/ArcFace 接口。"""

    def __init__(self, cfg, channels: int) -> None:
        super().__init__(cfg=cfg, backbone_type="ecapa_tdnn")
        channels = max(64, int(channels))
        mfa_channels = channels * 3
        self.audio_backbone = nn.ModuleDict(
            {
                "stem2d": nn.Sequential(
                    nn.Conv2d(
                        cfg.in_channels,
                        channels,
                        kernel_size=(5, 3),
                        padding=(2, 1),
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ),
                "layer1": _SERes2Block1d(
                    channels=channels,
                    scale=cfg.res2_scale,
                    dilation=2,
                    dropout=cfg.dropout,
                ),
                "layer2": _SERes2Block1d(
                    channels=channels,
                    scale=cfg.res2_scale,
                    dilation=3,
                    dropout=cfg.dropout,
                ),
                "layer3": _SERes2Block1d(
                    channels=channels,
                    scale=cfg.res2_scale,
                    dilation=4,
                    dropout=cfg.dropout,
                ),
                "mfa": nn.Sequential(
                    nn.Conv1d(mfa_channels, mfa_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(mfa_channels),
                    nn.SiLU(inplace=True),
                    nn.Dropout(p=cfg.dropout),
                ),
                "pool": _AttentiveStatsPool1d(
                    channels=mfa_channels,
                    hidden_channels=max(64, mfa_channels // 8),
                    global_context=True,
                ),
                "pool_bn": nn.BatchNorm1d(mfa_channels * 2),
            }
        )
        self.bottleneck_proj = nn.Linear(mfa_channels * 2, cfg.bottleneck_dim)
        self.emb_proj = nn.Linear(cfg.bottleneck_dim, cfg.embedding_dim)
        self.dropout = nn.Dropout(p=cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, feat) -> Dict[str, torch.Tensor]:
        x = _resolve_audio_tensor(feat, expected_channels=self.cfg.in_channels)
        x = self.audio_backbone["stem2d"](x)
        x = x.mean(dim=3)  # [B, C, T]

        x1 = self.audio_backbone["layer1"](x)
        x2 = self.audio_backbone["layer2"](x + x1)
        x3 = self.audio_backbone["layer3"](x + x1 + x2)

        mfa_feat = torch.cat([x1, x2, x3], dim=1)
        mfa_feat = self.audio_backbone["mfa"](mfa_feat)
        pooled = self.audio_backbone["pool"](mfa_feat)
        pooled = self.audio_backbone["pool_bn"](pooled)

        compressed = self.bottleneck_proj(self.dropout(pooled))
        compressed = F.normalize(compressed, p=2, dim=1)
        embedding = self.emb_proj(compressed)
        return self._build_outputs(
            embedding=embedding,
            aux={
                "compressed_embedding": compressed,
                "final_feature_map": mfa_feat,
            },
        )


class _ResNetBasicBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(stride, stride),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = _SEBlock2d(out_channels, reduction=8) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, stride), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.se(y)
        y = self.dropout(y)
        return self.act(y + residual)


class BaselineResNetXvectorSpeakerNet(_AudioOnlySpeakerNet):
    """ResNet-xvector 风格 baseline，输入输出完全兼容主训练框架。"""

    def __init__(self, cfg, base_channels: int) -> None:
        super().__init__(cfg=cfg, backbone_type="resnet_xvector")
        base_channels = max(16, int(base_channels))
        stage_blocks = (2, 2, 2, 2)
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.audio_backbone = nn.ModuleDict(
            {
                "stem": nn.Sequential(
                    nn.Conv2d(
                        cfg.in_channels,
                        channels[0],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels[0]),
                    nn.SiLU(inplace=True),
                ),
                "layer1": self._make_stage(
                    in_channels=channels[0],
                    out_channels=channels[0],
                    blocks=stage_blocks[0],
                    stride=1,
                    dropout=cfg.dropout,
                ),
                "layer2": self._make_stage(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    blocks=stage_blocks[1],
                    stride=2,
                    dropout=cfg.dropout,
                ),
                "layer3": self._make_stage(
                    in_channels=channels[1],
                    out_channels=channels[2],
                    blocks=stage_blocks[2],
                    stride=2,
                    dropout=cfg.dropout,
                ),
                "layer4": self._make_stage(
                    in_channels=channels[2],
                    out_channels=channels[3],
                    blocks=stage_blocks[3],
                    stride=2,
                    dropout=cfg.dropout,
                ),
                "pool": _AttentiveStatsPool1d(
                    channels=channels[3],
                    hidden_channels=max(64, channels[3] // 2),
                    global_context=False,
                ),
                "pool_bn": nn.BatchNorm1d(channels[3] * 2),
            }
        )
        self.bottleneck_proj = nn.Linear(channels[3] * 2, cfg.bottleneck_dim)
        self.emb_proj = nn.Linear(cfg.bottleneck_dim, cfg.embedding_dim)
        self.dropout = nn.Dropout(p=cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [
            _ResNetBasicBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dropout=dropout,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                _ResNetBasicBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, feat) -> Dict[str, torch.Tensor]:
        x = _resolve_audio_tensor(feat, expected_channels=self.cfg.in_channels)
        x = self.audio_backbone["stem"](x)
        x = self.audio_backbone["layer1"](x)
        x = self.audio_backbone["layer2"](x)
        x = self.audio_backbone["layer3"](x)
        x = self.audio_backbone["layer4"](x)
        feat_map = x

        # x: [B, C, F', T'] -> [B, C, T']
        x = x.mean(dim=2)
        pooled = self.audio_backbone["pool"](x)
        pooled = self.audio_backbone["pool_bn"](pooled)

        compressed = self.bottleneck_proj(self.dropout(pooled))
        compressed = F.normalize(compressed, p=2, dim=1)
        embedding = self.emb_proj(compressed)
        return self._build_outputs(
            embedding=embedding,
            aux={
                "compressed_embedding": compressed,
                "final_feature_map": feat_map,
            },
        )


class _FeedForwardModule(nn.Module):
    def __init__(self, dim: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        hidden = max(dim, dim * ff_mult)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.silu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y


class _ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        kernel_size = max(3, int(kernel_size) | 1)  # force odd kernel size
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        y = self.norm(x).transpose(1, 2)  # [B, D, T]
        y = self.pw1(y)
        y, gate = y.chunk(2, dim=1)
        y = y * torch.sigmoid(gate)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class _ConformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ffn1 = _FeedForwardModule(dim=dim, ff_mult=ff_mult, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.conv = _ConformerConvModule(
            dim=dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.ffn2 = _FeedForwardModule(dim=dim, ff_mult=ff_mult, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        q = self.attn_norm(x)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.out_norm(x)


class BaselineMFAConformerSpeakerNet(_AudioOnlySpeakerNet):
    """MFA-Conformer 风格 baseline，保留多层特征聚合并对齐统一输出格式。"""

    def __init__(self, cfg, model_dim: int, num_blocks: int) -> None:
        super().__init__(cfg=cfg, backbone_type="mfa_conformer")
        model_dim = max(64, int(model_dim))
        num_blocks = max(2, int(num_blocks))
        num_heads = max(1, int(cfg.num_heads))
        while model_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.audio_backbone = nn.ModuleDict(
            {
                "frontend": nn.Sequential(
                    nn.Conv2d(cfg.in_channels, model_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(model_dim),
                    nn.SiLU(inplace=True),
                    nn.Dropout2d(p=cfg.dropout),
                ),
                "pool": _AttentiveStatsPool1d(
                    channels=model_dim,
                    hidden_channels=max(64, model_dim // 2),
                    global_context=True,
                ),
                "pool_bn": nn.BatchNorm1d(model_dim * 2),
            }
        )
        self.blocks = nn.ModuleList(
            [
                _ConformerEncoderBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    ff_mult=cfg.ff_mult,
                    conv_kernel_size=cfg.conv_kernel_t,
                    dropout=cfg.dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(num_blocks))
        self.bottleneck_proj = nn.Linear(model_dim * 2, cfg.bottleneck_dim)
        self.emb_proj = nn.Linear(cfg.bottleneck_dim, cfg.embedding_dim)
        self.dropout = nn.Dropout(p=cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, feat) -> Dict[str, torch.Tensor]:
        x = _resolve_audio_tensor(feat, expected_channels=self.cfg.in_channels)
        x = self.audio_backbone["frontend"](x)  # [B, D, T, F]
        x = x.mean(dim=3).transpose(1, 2)  # [B, T, D]

        scales: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            scales.append(x.transpose(1, 2))  # [B, D, T]

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        fused = torch.zeros_like(scales[0])
        for weight, feat_scale in zip(scale_weights, scales):
            fused += weight * feat_scale

        pooled = self.audio_backbone["pool"](fused)
        pooled = self.audio_backbone["pool_bn"](pooled)

        compressed = self.bottleneck_proj(self.dropout(pooled))
        compressed = F.normalize(compressed, p=2, dim=1)
        embedding = self.emb_proj(compressed)
        return self._build_outputs(
            embedding=embedding,
            aux={
                "compressed_embedding": compressed,
                "scale_weights": scale_weights,
                "final_feature_map": fused,
                "multi_scale_features": torch.stack(scales, dim=1),
            },
        )


__all__ = [
    "BaselineECAPASpeakerNet",
    "BaselineResNetXvectorSpeakerNet",
    "BaselineMFAConformerSpeakerNet",
]
