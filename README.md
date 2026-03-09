# HandTalk AI - 基于深度视觉模型的连续手语识别与跨模态检索系统

> 一个端到端的智能手语识别与检索系统，实现从手语视频/图像到中文文本的自动识别，以及基于语义相似度的文本-视频跨模态检索。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![Nuxt](https://img.shields.io/badge/Nuxt-3.9.3-00DC82.svg)](https://nuxt.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目概述

HandTalk AI 是一个完整的智能手语识别与检索系统，旨在解决聋哑人群与健听人群之间的无障碍沟通问题。系统采用端到端的深度学习架构，融合计算机视觉、自然语言处理和机器学习技术，实现了：

- **端到端连续手语识别**：从手语视频/图像自动识别为中文文本
- **跨模态检索**：基于语义相似度的文本-视频检索系统
- **智能词典搜索**：支持关键词匹配、语义搜索、同义词扩展的混合搜索
- **实时识别服务**：支持 HTTP 轮询与 WebSocket 双模式的实时识别

## ✨ 核心特性

### 1. 深度学习模型
- **CTC 序列识别模型**：基于 ResNet18 + 1D-CNN + 双向LSTM 的混合架构
- **端到端训练**：使用 CTC Loss 解决序列对齐问题，无需帧级标注
- **迁移学习**：使用预训练 ResNet18（ImageNet）进行特征提取
- **时序建模**：1D-CNN 进行局部时序建模，BiLSTM 进行全局时序建模

### 2. 多模态数据处理
- **视频处理**：支持视频帧采样、时序对齐、特征提取
- **图像处理**：支持图像预处理、归一化、数据增强
- **文本处理**：使用 BGE-small-zh-v1.5 进行中文文本向量化（384维）
- **跨模态检索**：文本-视频语义相似度匹配

### 3. 智能系统架构
- **前后端分离**：FastAPI 后端 + Nuxt 3 前端
- **RESTful API**：完整的 API 接口设计
- **实时通信**：WebSocket 支持实时识别
- **状态管理**：Pinia 进行前端状态管理

## 🏗️ 项目结构

```
Shen/
├── AAA/                          # 手语翻译模型训练
│   ├── models/                   # 模型定义（ResNet, BiLSTM, CTC, Transformer）
│   ├── train.py                  # 训练脚本
│   ├── test.py                   # 测试脚本
│   ├── dataset.py                # 数据集处理
│   └── README.md                 # 详细文档
│
├── Project/
│   ├── Back/                     # 后端服务（FastAPI）
│   │   ├── app/                  # 应用主目录
│   │   │   ├── main.py           # FastAPI 主应用
│   │   │   ├── recognizer.py     # 手语识别推理模块
│   │   │   ├── ctc_service.py    # CTC 识别服务
│   │   │   └── rag_index.py      # 文本-视频检索索引
│   │   ├── scripts/              # 训练与工具脚本
│   │   ├── model/                # 模型检查点
│   │   ├── data/                 # 数据索引
│   │   └── requirements.txt      # Python 依赖
│   │
│   └── Front/                    # 前端应用（Nuxt 3）
│       ├── pages/                # 页面路由
│       │   ├── index.vue         # 首页
│       │   ├── recognize.vue     # 实时翻译
│       │   ├── translate.vue     # 上传翻译
│       │   ├── video-translate.vue # 视频翻译
│       │   ├── dictionary.vue    # 手语词典
│       │   └── auth/             # 认证页面
│       ├── components/           # Vue 组件
│       ├── composables/          # 组合式函数
│       ├── stores/               # Pinia 状态管理
│       └── package.json          # Node.js 依赖
│
├── ShenCode/                     # 实验代码
│   ├── src/                      # 源代码
│   ├── scripts/                  # 实验脚本
│   ├── configs/                  # 配置文件
│   └── results/                  # 实验结果
│
├── dataset/                      # 数据集（CE-CSL）
│   └── CE-CSL/                   # 中文手语数据集
│
└── README.md                     # 本文件
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.11+
- **Node.js**: 18+
- **CUDA**: 可选（用于 GPU 加速）
- **Conda**: 推荐用于 Python 环境管理

### 1. 后端服务启动

```bash
# 创建 Conda 环境
conda create -n Shen python=3.11 -y
conda activate Shen

# 安装依赖
cd Project/Back
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

后端服务将在 `http://localhost:8000` 启动，API 文档访问 `http://localhost:8000/docs`

### 2. 前端应用启动

```bash
# 进入前端目录
cd Project/Front

# 安装依赖
npm install
# 或使用 pnpm（推荐）
pnpm install

# 启动开发服务器
npm run dev
# 或
pnpm dev
```

前端应用将在 `http://localhost:3000` 启动

### 3. 模型训练（可选）

```bash
# 训练 CTC 模型
cd AAA
python train.py --data-root /path/to/dataset --epochs 50 --batch-size 8

# 或使用后端训练脚本
cd Project/Back
python scripts/train_model_ctc.py
```

## 📚 核心功能

### 1. 手语识别

- **图像识别**：上传单张手语图片，识别为中文文本
- **视频识别**：上传手语视频，识别为中文文本序列
- **实时识别**：通过摄像头实时捕获手语动作并识别

### 2. 词典搜索

- **关键词搜索**：支持中文、拼音匹配
- **语义搜索**：基于句向量的语义相似度搜索
- **混合搜索**：融合关键词和语义搜索，提升召回率
- **分类筛选**：按手语类别筛选词汇

### 3. 文本-视频检索

- **跨模态检索**：输入中文文本，检索相关手语视频
- **语义匹配**：基于向量相似度的语义级别匹配
- **Top-K 检索**：返回最相关的 K 个视频结果

## 🔧 技术栈

### 后端技术
- **框架**: FastAPI
- **深度学习**: PyTorch 2.1.0
- **模型架构**: ResNet18 + 1D-CNN + BiLSTM + CTC
- **文本向量化**: sentence-transformers (BGE-small-zh-v1.5)
- **图像处理**: OpenCV, Pillow
- **数据处理**: NumPy, Pandas

### 前端技术
- **框架**: Nuxt 3 + Vue 3 Composition API
- **语言**: TypeScript
- **样式**: Tailwind CSS
- **状态管理**: Pinia
- **工具库**: VueUse
- **图表**: ECharts

## 📊 模型架构

```
输入视频 (T 帧)
    │
    ▼
┌─────────────────────┐
│  ResNet18           │  空间特征提取
│  (Pretrained)       │  输出: (B, T, 512)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  1D-CNN             │  局部时序建模
│  (保持时间维度)     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  BiLSTM (2层)       │  全局时序建模
│  Hidden: 256        │  输出: (B, T, 512)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  CTC Classifier     │  序列对齐与解码
│  (Greedy Decode)    │  输出: 中文文本
└─────────────────────┘
```

## 📖 API 文档

### 手语识别接口

- `POST /api/v1/recognize/image` - 识别图片中的手语
- `POST /api/v1/recognize/video` - 识别视频中的手语
- `GET /api/v1/recognize/status` - 获取识别服务状态

### 词典搜索接口

- `GET /api/v1/dictionary/search?query=关键词&search_type=hybrid&top_k=20` - 搜索词汇
- `GET /api/v1/dictionary/list?page=1&limit=100` - 获取词汇列表
- `GET /api/v1/dictionary/{id}` - 获取单个词汇详情
- `GET /api/v1/dictionary/categories/list` - 获取所有分类

### 文本-视频检索接口

- `POST /api/v1/rag/search` - 文本到视频检索

详细 API 文档请访问：`http://localhost:8000/docs`

## 🎯 数据集

本项目使用 **CE-CSL (Chinese Sign Language)** 数据集：

- **训练集**: 包含多个翻译者的手语视频
- **验证集**: 用于模型验证
- **测试集**: 用于最终评估
- **标注格式**: CSV 格式，包含视频编号、翻译者、中文句子、Gloss 序列

数据集结构：
```
dataset/CE-CSL/CE-CSL/
├── label/
│   ├── train.csv      # 训练集标注
│   ├── dev.csv        # 验证集标注
│   └── test.csv       # 测试集标注
└── video/
    ├── train/         # 训练视频（A-L 翻译者）
    ├── dev/           # 验证视频
    └── test/          # 测试视频
```

## 📈 实验结果

- **模型架构**: ResNet18 + 1D-CNN + BiLSTM + CTC
- **训练策略**: Adam 优化器，学习率 1e-4，梯度裁剪，早停机制
- **评估指标**: CTC Loss, 验证集准确率
- **检索性能**: Top-K 准确率，余弦相似度匹配

详细实验结果请参考 `Project/results/` 目录。

## 🔬 研究贡献

### 技术创新点

1. **混合时序建模架构**：结合 1D-CNN 的局部特征提取与 BiLSTM 的全局时序建模
2. **端到端训练范式**：使用 CTC Loss 实现无需帧级标注的序列识别
3. **跨模态检索系统**：将文本查询与视频内容映射到统一的向量空间
4. **多模式搜索融合**：设计关键词匹配、语义搜索、同义词扩展的混合策略

### 研究方向

- **多模态数据分析**：处理视频、图像、文本三种模态，实现跨模态检索
- **深度学习建模**：CTC 模型、时序建模、迁移学习
- **智能系统设计**：前后端分离、服务化架构、实时处理
- **计算机视觉**：手语识别、动作识别、特征提取

## 📝 文档

- [后端启动指南](Project/Back/启动.md)
- [前端项目文档](Project/Front/README.md)
- [模型训练文档](AAA/README.md)
- [项目科研化重构报告](研究生申请-项目科研化重构报告.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅用于学术研究。

## 🙏 致谢

- **数据集**: CE-CSL (Chinese Sign Language Dataset)
- **预训练模型**: 
  - ResNet18 (ImageNet)
  - BGE-small-zh-v1.5 (sentence-transformers)
- **框架**: PyTorch, FastAPI, Nuxt 3

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

---

**项目版本**: v1.0.0  
**最后更新**: 2025-01-15
