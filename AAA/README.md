# Sign Language Translation Model

基于 CE-CSL 数据集的手语翻译模型，实现从手语视频到中文文本的端到端翻译。

## 目录

- [项目概述](#项目概述)
- [功能特点](#功能特点)
- [安装依赖](#安装依赖)
- [数据集结构](#数据集结构)
- [快速开始](#快速开始)
  - [训练](#训练)
  - [测试](#测试)
  - [推理](#推理)
- [模型架构](#模型架构)
- [配置参数](#配置参数)
- [评估指标](#评估指标)
- [项目结构](#项目结构)
- [参考文档](#参考文档)

## 项目概述

本项目实现了一个完整的手语翻译系统，基于深度学习技术将手语视频自动转换为中文文本。

### 技术栈

- **PyTorch**: 深度学习框架
- **ResNet-50**: 空间特征提取
- **Bi-LSTM**: 时序建模
- **CTC Loss**: 序列对齐
- **Transformer**: 端到端翻译

## 功能特点

1. **多任务学习**: 同时支持 Gloss 识别和中文翻译
2. **混合精度训练**: 支持 FP16 加速训练
3. **灵活配置**: 支持命令行参数和配置文件
4. **多种解码策略**: 支持 Greedy 和 Beam Search
5. **完整评估**: BLEU, WER, CER 指标计算

## 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 数据集结构

```
dataset/CE-CSL/CE-CSL/
├── label/
│   ├── train.csv      # 训练集标注
│   ├── dev.csv        # 验证集标注
│   └── test.csv       # 测试集标注
└── video/
    ├── train/
    │   ├── A/         # 翻译者 A
    │   ├── B/
    │   └── ... (A-L)
    ├── dev/
    │   └── A-L/
    └── test/
        └── A-L/
```

### CSV 文件格式

```
Number,Translator,Chinese Sentences,Gloss,Note
train-00001,A,2023年高考到了。,2/0/2/3/高/考/时间/到/。,
```

- `Number`: 视频编号
- `Translator`: 翻译者标识 (A-L)
- `Chinese Sentences`: 中文句子
- `Gloss`: 手语词序列 (用 `/` 分隔)

## 快速开始

### 训练

```bash
# 基本训练
python train.py --data-root /path/to/dataset --epochs 50 --batch-size 8

# 指定输出目录
python train.py --data-root /path/to/dataset --output-dir ./output --epochs 50

# 使用混合精度
python train.py --data-root /path/to/dataset --use-amp --epochs 50

# 自定义学习率
python train.py --data-root /path/to/dataset --learning-rate 1e-4 --epochs 50
```

### 测试

```bash
# 评估验证集
python test.py --checkpoint ./output/model_best.pth --eval

# 评估测试集
python test.py --checkpoint ./output/model_best.pth --eval --data-root /path/to/dataset

# 评估指定样本数
python test.py --checkpoint ./output/model_best.pth --eval --num-samples 500
```

### 推理

```bash
# 单个视频推理
python test.py --checkpoint ./output/model_best.pth --test-video video.mp4

# 单张图片推理
python test.py --checkpoint ./output/model_best.pth --test-image image.jpg

# 批量推理目录
python test.py --checkpoint ./output/model_best.pth --test-dir ./test_videos

# 使用 Beam Search
python test.py --checkpoint ./output/model_best.pth --test-video video.mp4 --beam-search
```

## 模型架构

```
Input Video (T frames)
    │
    ▼
┌─────────────────────┐
│  ResNet-50          │  空间特征提取
│  (Pretrained)       │
│  Output: (B, T, 512)│
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Bi-LSTM (2层)      │  时序建模
│  Hidden: 512        │
│  Output: (B, T, 1024)│
└─────────────────────┘
    │
    ├───┐
    │   │
    ▼   ▼
┌──────────┐  ┌─────────────────┐
│  CTC      │  │  Transformer    │
│  Classifier│  │  Encoder-Decoder│
│  Gloss识别 │  │  中文翻译       │
└──────────┘  └─────────────────┘
```

## 配置参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 8 | 批次大小 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--grad-clip-norm` | 1.0 | 梯度裁剪 |
| `--use-amp` | True | 混合精度训练 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resnet-model` | resnet50 | ResNet 模型 |
| `--lstm-hidden-size` | 512 | LSTM 隐藏层大小 |
| `--lstm-num-layers` | 2 | LSTM 层数 |
| `--vocab-size` | 1000 | 词汇表大小 |

## 评估指标

- **BLEU**: 翻译质量评估 (1-4 gram)
- **WER**: 词错误率 (Gloss 识别)
- **CER**: 字符错误率

## 项目结构

```
AAA/
├── config.py           # 配置文件
├── dataset.py          # 数据集处理
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── requirements.txt   # 依赖列表
├── README.md          # 项目说明
├── models/
│   ├── __init__.py
│   ├── resnet.py      # ResNet 特征提取
│   ├── bilstm.py      # BiLSTM 时序建模
│   ├── ctc.py         # CTC 损失和解码
│   ├── transformer.py  # Transformer 翻译
│   └── translator.py  # 完整翻译模型
└── output/            # 输出目录
    ├── exp_*/         # 实验目录
    │   ├── checkpoint_latest.pth
    │   ├── model_best.pth
    │   ├── config.json
    │   └── training.log
```

## 模型检查点格式

```python
checkpoint = {
    'epoch': int,              # 当前轮数
    'model_state_dict': dict, # 模型权重
    'optimizer_state_dict': dict, # 优化器状态
    'best_bleu': float,       # 最佳 BLEU
    'best_loss': float,        # 最佳损失
}
```

## 常见问题

### 1. 显存不足

```bash
# 减小批次大小
python train.py --batch-size 4

# 减小序列长度
python train.py --num-frames 16
```

### 2. 训练不稳定

```bash
# 关闭混合精度
python train.py --use-amp False

# 降低学习率
python train.py --learning-rate 5e-5
```

### 3. 模型不收敛

```bash
# 增加训练轮数
python train.py --epochs 100

# 使用学习率预热
python train.py --learning-rate 1e-4 --epochs 100
```

## 参考文档

- [执行.md](执行.md) - 详细训练方案
- [手语翻译模型训练方案.docx](手语翻译模型训练方案.docx) - 完整技术文档

## 许可证

本项目仅用于学术研究。

## 引用

如果使用本项目代码，请引用相关论文。




