# 基于时频注意力 Conformer 的多尺度短语音说话人识别

本项目实现了论文图示对应的模型与训练流程：

- 帧级特征编码器：`SE-Res2Block × 3` 堆叠
- `TFA-Conformer` 模块堆叠与多尺度融合
- 压缩激励时域均衡模块
- `Warmup + Cosine/Step` 学习率调度
- MUSAN 噪声数据增强（训练阶段）
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
```

可在 `configs/paper_experiment.yaml` 中修改：

- `data.roots`
- `data.extensions`
- `data.speaker_level`
- `data.max_samples_per_speaker`（每位说话人最多保留多少条音频，默认 6）
- `data.speaker_sample_seed`（说话人内样本超限时的采样随机种子）
- `data.musan_*`（MUSAN 增强开关、路径、SNR 范围、增强概率）

## 训练

```bash
python train.py --config configs/paper_experiment.yaml
```

支持主流分类损失配置（`train.loss_type`）：

- `arcface`：AAM-Softmax（默认）
- `cosface`：AM-Softmax
- `ce`：普通 Softmax 交叉熵

可恢复训练：

```bash
python train.py --config configs/paper_experiment.yaml --resume runs/<run_name>/last.pt
```

## 评估

```bash
python evaluate.py --config configs/paper_experiment.yaml --checkpoint runs/<run_name>/best.pt
```

## 说明

- 配置中已按论文实验思路提供默认参数（16kHz、2.5s、25ms/10ms、n_fft=256、batch size=64、Adam + Warmup + Cosine）。
- `model.num_speakers` 会在训练时自动以扫描到的说话人数覆盖，避免手动不一致。

