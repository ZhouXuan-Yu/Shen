# 项目全景 & Agent 上手指南

> **用途**：当 AI Agent 丢失对话记忆、或新 Agent 首次接手本项目时，阅读本文档即可快速了解项目背景、结构、约束、当前进度，并能立即开始工作。

---

## 相关文档引用

本文档为快速上手概览。如需深入了解，请阅读以下文档：

- **实验设计权威文档**：[`docs/开发计划_实验计划.md`](./开发计划_实验计划.md)
  - 包含：实验矩阵、协议详情、公平性原则、3天执行顺序、验收标准
- **开发日志**：[`docs/开发计划_开发记录.md`](./开发计划_开发记录.md)
  - 包含：里程碑进度、按日期追加的开发记录、配置约定
- **项目说明**：[`Readme.md`](../Readme.md)
  - 包含：项目背景、教师硬约束原文、需求范围

> **建议阅读顺序**：本文档 → 开发计划_实验计划.md → 开发计划_开发记录.md

---

## 0. 一句话定位

**本项目是本科毕业论文**，主题为《简化版 ResNet 在小样本手语识别中的应用》，核心任务是：**使用现成的轻量 CNN（ResNet-18/34、MobileNetV2 等）在 few-shot 手语识别任务上做公平对比实验，输出模型选型建议**。

---

## 1. 硬约束（必须严格遵守）

| 约束类型 | 具体要求 |
|---------|---------|
| **禁止改模型结构** | 不允许修改 ResNet/MobileNet 等骨干网络的任何结构（不加注意力、不换 block、不做 3D/时序结构创新）。唯一允许的修改是**替换最后分类层**以适配类别数。 |
| **禁止设计新模型** | 不允许提出"新模型/新算法"，只能使用 torchvision/timm 等现成实现。 |
| **训练策略可调** | 允许调整超参（学习率、weight decay、增强强度等），但必须在验证集上决定、完整记录搜索过程。 |
| **可复现性** | 所有实验必须固定随机种子、保存配置原文、记录环境信息。 |
| **语言输出** | 所有面向用户的文档/计划/解释必须使用**中文**；代码标识符保持英文。 |

---

## 2. 项目目标与交付物

### 2.1 核心目标

1. 在 **AUTSL**（土耳其手语，226 类，~36k 视频）数据集上，完成 **few-shot（K=1/5）分类对比实验**。
2. 对比模型：**ResNet-18、ResNet-34、MobileNetV2**（必须）；ShuffleNetV2、EfficientNet-B0（可选扩展）。
3. 输出：主对比表（Acc±Std + 资源指标）、收敛曲线、过拟合分析图、性能-资源权衡图、模型选型建议。

### 2.2 硬件与时间约束

- **GPU**：RTX 3050 4GB（移动版）
- **时间预算**：3 天内完成 MVP

### 2.3 交付物清单

- `results/tables/table_main.csv`：主对比表
- `results/figures/`：收敛曲线、过拟合图、权衡图
- `results/summaries/`：模型选型建议文本

---

## 3. 数据集信息（AUTSL）

| 属性 | 值 |
|-----|---|
| **名称** | AUTSL（Ankara University Turkish Sign Language Dataset） |
| **来源** | ChaLearn LAP CVPR'21 Challenge |
| **类别数** | 226 |
| **总样本数** | ~36,302 视频 |
| **模态** | RGB 视频（`*_color.mp4`）+ 深度视频（`*_depth.mp4`，本项目只用 RGB） |
| **分辨率** | 512×512 |
| **官方划分** | train（31 signers）/ val（6 signers）/ test（6 signers） |
| **标签格式** | CSV：`signerX_sampleY, label`（label 为 0-225 的类别 ID） |
| **下载页** | <https://chalearnlap.cvc.uab.es/dataset/40/description/> |

### 3.1 解压密钥

- Train：`MdG3z6Eh1t`
- Validation：`bhRY5B9zS2`
- Test / Test labels：`ds6Kvdus3o`
- Validation labels：`zYX5W7fZ`

### 3.2 数据存放位置

```text
data/raw/
├── train/           # 训练集视频（只用 *_color.mp4）
├── val/             # 验证集视频
├── test/            # 测试集视频
├── train_labels.csv # 训练集标签
├── val_labels.csv   # 验证集标签
├── test_labels.csv  # 测试集标签
└── class_id.csv     # 类别ID→土耳其语/英语映射
```

---

## 4. 项目目录结构

```text
project_root/
├── data/
│   ├── raw/              # 原始数据（解压后的视频+标签）
│   ├── processed/        # 处理后数据（manifest.jsonl、label2id.json、stats.json）
│   └── splits/           # few-shot 索引文件（kshot_K*_seed*.json、val_seed*.json）
├── configs/
│   ├── dataset/          # 数据集配置（路径、抽帧策略等）
│   ├── model/            # 模型配置（名称、权重、冻结策略）
│   ├── train/            # 训练配置（lr、epoch、aug、amp 等）
│   └── eval/             # 评估配置（Protocol A/B 参数）
├── src/                  # 代码主体（dataset、model、utils、metrics 等）
├── tools/                # CLI 入口脚本
│   ├── prepare_dataset.py   # 扫描数据→生成 manifest/stats
│   ├── make_fewshot_splits.py # 生成 K-shot/val 索引
│   ├── train.py             # 训练脚本（待实现）
│   ├── eval.py              # 评估脚本（待实现）
│   ├── benchmark.py         # 资源评测脚本（待实现）
│   └── summarize_results.py # 汇总结果脚本（待实现）
├── runs/                 # 单次运行产物（按 run_id 组织）
├── results/
│   ├── tables/           # 汇总表（CSV）
│   ├── figures/          # 图（PNG/PDF）
│   └── summaries/        # 汇总文本
├── docs/
│   ├── 开发计划_实验计划.md    # 实验设计权威文档（详细协议、矩阵、验收标准）
│   ├── 开发计划_开发记录.md    # 开发日志（按日期追加）
│   └── 项目全景_Agent上手指南.md  # 本文档
├── .windsurf/rules/      # Windsurf/Cascade 规则文件
└── Readme.md             # 项目说明与快速链接
```

---

## 5. 关键文档速查

| 文档 | 路径 | 用途 |
|-----|------|-----|
| **实验计划（权威）** | [`docs/开发计划_实验计划.md`](./开发计划_实验计划.md) | 实验设计、协议、矩阵、3天执行顺序、验收标准 |
| **开发日志** | [`docs/开发计划_开发记录.md`](./开发计划_开发记录.md) | 工程落地记录、里程碑、按日期追加 |
| **本文档** | `docs/项目全景_Agent上手指南.md` | Agent 快速上手概览 |
| **Readme** | [`Readme.md`](../Readme.md) | 项目背景、教师硬约束、快速链接 |

---

## 6. 已实现的脚本

### 6.1 `tools/prepare_dataset.py`

**功能**：扫描 `data/raw/` 目录，生成 manifest 与统计文件。

**当前支持**：

- 目录结构扫描模式：`data/raw/<split>/<label>/*` 或 `data/raw/<label>/*`
- 支持图片（jpg/png/bmp/webp）和视频（mp4/avi/mov/mkv/webm/m4v）

**待扩展**：

- 支持 `--label-file` 参数，从 AUTSL 的 `train_labels.csv` 导入标签（避免按目录组织）

**输出**：

- `data/processed/manifest.jsonl`：样本清单
- `data/processed/label2id.json`：类别→ID 映射
- `data/processed/id2label.json`：ID→类别映射
- `data/processed/stats.json`：统计信息

### 6.2 `tools/make_fewshot_splits.py`

**功能**：基于 manifest 生成 K-shot 训练索引与 val 索引。

**参数**：

- `--manifest`：manifest 路径（默认 `data/processed/manifest.jsonl`）
- `--k`：K-shot 的 K（默认 5）
- `--seed`：随机种子（默认 0）
- `--val-per-class`：从 train 抽取的 val 样本数/类（默认 2）

**输出**：

- `data/splits/val_seed{seed}.json`
- `data/splits/kshot_K{k}_seed{seed}.json`

---

## 7. 实验设计摘要

### 7.1 模型清单

| 优先级 | 模型 | 来源 |
|-------|------|-----|
| P0（必须） | ResNet-18 | torchvision/timm |
| P0（必须） | ResNet-34 | torchvision/timm |
| P0（必须） | MobileNetV2 | torchvision |
| P1（扩展） | ShuffleNetV2 1.0x | torchvision |
| P1（扩展） | EfficientNet-B0 | torchvision/timm |

### 7.2 Few-shot 设置

- **K-shot**：K=1, K=5
- **Seeds**：3 次（0/1/2），最终可扩展到 5 次
- **预训练**：统一使用 ImageNet 预训练权重

### 7.3 输入处理（不改结构）

- 视频抽帧 → 图像输入 2D CNN
- MVP：`T_train=1`（每视频 1 帧），`T_test=3`
- 输入尺寸：224×224（显存紧张可统一降到 192/160）
- 视频级推理：对同一视频抽 T 帧分别前向，logits 取平均

### 7.4 训练策略

- Loss：Cross-Entropy
- 分类头：仅替换最后分类层
- 冻结策略：阶段A（训头 5 epoch）→ 阶段B（全网训练至早停，上限 50 epoch）
- 优化器：AdamW，lr=1e-3（训头）/ 3e-4（全网）
- 学习率策略：Cosine decay，warmup 3 epoch
- Batch size：32（OOM 则统一下调）
- AMP：开启
- Early stop：patience 10

### 7.5 评估协议

- **Protocol A**（主）：固定划分全量测试，跨 seeds 报告 mean±std
- **Protocol B**（补充）：任务抽样评估（N-way，N=5/10，Q=15，T=200 任务）

### 7.6 评估指标

- 精度：Top-1（必须）、Top-5（推荐）、mean±std
- 稳定性：seeds 维度 Std、95% CI
- 类别维度：class-wise acc、混淆矩阵
- 资源：Params、FLOPs、Latency、Throughput、Peak VRAM

---

## 8. 当前进度与待办

### 8.1 已完成

- [x] 项目目录结构创建
- [x] `tools/prepare_dataset.py` 基础版本
- [x] `tools/make_fewshot_splits.py` 基础版本
- [x] 数据集选定（AUTSL）
- [x] 数据集下载完成

### 8.2 进行中

- [ ] 实现 `tools/train.py`（训练脚本）
- [ ] 实现数据加载器（支持视频抽帧）

### 8.3 待办（按优先级）

1. **M1 数据集定稿** ✅ 已完成
   - [x] 解压数据到 `data/raw/`（train/val/test 共 36,302 RGB 视频）
   - [x] 扩展 `prepare_dataset.py` 支持 `--autsl` 模式
   - [x] 运行 `prepare_dataset.py` 生成 manifest 与统计（226 类别）
   - [x] 运行 `make_fewshot_splits.py` 生成 K-shot/val 索引（K=1/5 × seed=0/1/2）

2. **M2 训练闭环**
   - [ ] 实现 `tools/train.py`
   - [ ] 实现数据加载器（支持视频抽帧）
   - [ ] 跑通 ResNet-18 + K=5 + seed=0

3. **M3 评估闭环**
   - [ ] 实现 `tools/eval.py`
   - [ ] Protocol A 评估
   - [ ] Protocol B 评估（可选）

4. **M4 资源评测**
   - [ ] 实现 `tools/benchmark.py`

5. **M5 结果汇总**
   - [ ] 实现 `tools/summarize_results.py`
   - [ ] 生成主对比表与图

---

## 9. 常用命令速查

```bash
# 进入项目目录
cd "d:\Project\Application of a Simplified ResNet in Few-Shot Sign Language Recognition"

# 生成 manifest 与统计（AUTSL 模式）
python tools/prepare_dataset.py --autsl

# 生成 K-shot 索引
python tools/make_fewshot_splits.py --k 1 --seed 0
python tools/make_fewshot_splits.py --k 5 --seed 0

# 训练（待实现）
python tools/train.py --config configs/train/default.yaml --model resnet18 --k 5 --seed 0

# 评估（待实现）
python tools/eval.py --checkpoint runs/<run_id>/checkpoints/best.pt --protocol a

# 资源评测（待实现）
python tools/benchmark.py --model resnet18 --input-size 224 --batch 1
```

---

## 10. Agent 接手检查清单

当你（AI Agent）首次接手或丢失记忆后，请按以下步骤确认状态：

1. **阅读本文档**：了解项目背景、约束、结构（5分钟）
2. **查看 [`docs/开发计划_实验计划.md`](./开发计划_实验计划.md)**：了解详细实验设计、协议、矩阵（10分钟）
3. **查看 [`docs/开发计划_开发记录.md`](./开发计划_开发记录.md)**：了解当前进度与历史记录（5分钟）
4. **检查 `data/raw/` 目录**：确认数据是否已解压
5. **检查 `data/processed/` 目录**：确认 manifest 是否已生成
6. **检查 `data/splits/` 目录**：确认 few-shot 索引是否已生成
7. **根据待办清单继续工作**

---

## 11. 联系与版本

- **项目类型**：本科毕业论文
- **文档版本**：v1.1（2026-01-04）
- **最后更新**：M1 数据集定稿完成，manifest 与 K-shot 索引已生成

---

> **重要提醒**：本项目的核心约束是**不改模型结构**。任何涉及修改 ResNet/MobileNet 内部结构的建议都是违反约束的，请严格遵守。
