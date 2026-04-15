# 基于时频注意力 Conformer 的多尺度短语音说话人识别

本项目实现了论文图示对应的模型与训练流程：

- 帧级特征编码器：`SE-Res2Block × 3` 堆叠
- `TFA-Conformer` 模块堆叠与多尺度融合
- 压缩激励时域均衡模块
- 视频分支：`ResNet18` ROI 编码 + 注意力聚合
- 自适应音视频融合（可切换为简单平均用于消融）
- `Warmup + Cosine/Step` 学习率调度
- MUSAN 噪声数据增强（训练阶段）
- 模态随机丢弃（训练阶段随机置零音频/视频分支）
- 说话人分类训练（ACC / Precision / Recall / F1）

## 目录结构

```text
new_asr/
├─ configs/
│  └─ paper_experiment.yaml
├─ src/
│  └─ tfa_conformer_sid/
│     ├─ config.py
│     ├─ dataio.py
│     ├─ engine/
│     │  └─ trainer.py
│     ├─ models/
│     │  └─ tfa_multiscale_conformer.py
│     └─ utils/
│        ├─ metrics.py
│        └─ seed.py
├─ train.py
├─ evaluate.py
└─ main.py
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据组织

默认按“音频父目录名即说话人 ID”读取数据（`speaker_level: 1`）：

```text
data/TIMIT/<speaker_id>/*.wav
data/ST-CMDS/<speaker_id>/*.wav
data/musan/**/*.(wav|flac)   # 可选，用于噪声增强
data/video_roi/<speaker_id>/*.mp4  # 可选，用于视频分支
```

可在 `configs/paper_experiment.yaml` 中修改：

- `data.roots`
- `data.extensions`
- `data.speaker_level`
- `data.max_samples_per_speaker`（每位说话人最多保留多少条音频，默认 6）
- `data.speaker_sample_seed`（说话人内样本超限时的采样随机种子）
- `data.musan_*`（MUSAN 增强开关、路径、SNR 范围、增强概率）
- `data.video_*`（视频分支开关、路径、帧数、ROI 尺寸）
- `train.modality_drop_*`（训练阶段音频/视频整分支随机置零概率）
- `model.use_*`（分支级开关 + 论文 `-Conv/-SE/-TFA` 消融开关）

## 训练

```bash
python train.py --config configs/paper_experiment.yaml
```

现已支持在同一训练脚本中切换 3 个 baseline（统一复用 ArcFace/CosFace 训练头）：

- `model.backbone_type: "ecapa_tdnn"`
- `model.backbone_type: "resnet_xvector"`
- `model.backbone_type: "mfa_conformer"`

为了做更严格控制变量，默认启用 `model.auto_match_baseline_params: true`，会自动将 baseline 的通道宽度搜索到与当前 TFA 配置最接近的参数量（也可用 `model.baseline_target_params` 手动指定预算）。

支持主流分类损失配置（`train.loss_type`）：

- `arcface`：AAM-Softmax（默认）
- `cosface`：AM-Softmax
- `ce`：普通 Softmax 交叉熵

可恢复训练：

```bash
python train.py --config configs/paper_experiment.yaml --resume runs/<run_name>/last.pt
```

论文消融实验可通过以下配置开关直接控制：

- `-Conv`：`model.use_conformer_conv: false`
- `-SE`：`model.use_balance_se: false`（关闭块级特征均衡模块的 SE 门控）
- `-TFA`：`model.use_tfa_pooling: false`

## 评估

```bash
python evaluate.py --config configs/paper_experiment.yaml --checkpoint runs/<run_name>/best.pt
```

## 说明

- 配置中已按论文实验思路提供默认参数（16kHz、2.5s、25ms/10ms、n_fft=256、batch size=64、Adam + Warmup + Cosine）。
- `model.num_speakers` 会在训练时自动以扫描到的说话人数覆盖，避免手动不一致。

