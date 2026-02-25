"""
手语翻译模型训练配置
基于 CE-CSL 数据集的配置参数

参考 AAA/执行.md 文档实现:
- ResNet-50: 空间特征提取
- Bi-LSTM: 时序建模
- CTC Loss: 序列对齐
- Transformer: 端到端翻译
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ==================== 路径配置 ====================
@dataclass
class PathConfig:
    """路径配置"""
    # 数据根目录
    DATA_ROOT: str = 'D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL'
    
    # 标签目录
    LABEL_DIR: str = os.path.join(DATA_ROOT, 'label')
    
    # 视频目录
    VIDEO_DIR: str = os.path.join(DATA_ROOT, 'video')
    
    # 输出目录
    OUTPUT_DIR: str = './output'
    
    # 模型保存目录
    MODEL_DIR: str = os.path.join(OUTPUT_DIR, 'models')
    
    # 日志目录
    LOG_DIR: str = os.path.join(OUTPUT_DIR, 'logs')
    
    # 结果目录
    RESULTS_DIR: str = os.path.join(OUTPUT_DIR, 'results')
    
    def __post_init__(self):
        """初始化创建目录"""
        for dir_path in [self.OUTPUT_DIR, self.MODEL_DIR, self.LOG_DIR, self.RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)


# ==================== 数据配置 ====================
@dataclass
class DataConfig:
    """数据配置"""
    # 训练集
    TRAIN_CSV: str = 'train.csv'
    TRAIN_VIDEO_DIR: str = 'train'
    
    # 验证集
    VAL_CSV: str = 'dev.csv'
    VAL_VIDEO_DIR: str = 'dev'
    
    # 测试集
    TEST_CSV: str = 'test.csv'
    TEST_VIDEO_DIR: str = 'test'
    
    # 翻译者目录 (A-L)
    TRANSLATORS: List[str] = field(default_factory=lambda: list('ABCDEFGHIJKL'))
    
    # 视频采样帧数
    NUM_FRAMES: int = 32
    
    # 图像尺寸
    IMG_SIZE: int = 224
    
    # 是否缓存帧到内存（内存充足时启用）
    CACHE_FRAMES: bool = False
    
    # 数据集划分比例
    TRAIN_RATIO: float = 0.9
    
    # 最小词汇出现次数
    MIN_GLOSS_COUNT: int = 2


# ==================== 模型配置 ====================
@dataclass
class ModelConfig:
    """模型配置"""
    # ==================== ResNet 配置 ====================
    RESNET_MODEL: str = 'resnet50'
    RESNET_PRETRAINED: bool = True
    RESNET_OUTPUT_DIM: int = 512  # ResNet特征降维后的维度
    
    # 冻结层数 (可选)
    RESNET_FREEZE_LAYERS: int = 10
    
    # ==================== BiLSTM 配置 ====================
    LSTM_HIDDEN_SIZE: int = 512
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.3
    LSTM_BIDIRECTIONAL: bool = True
    
    # ==================== Transformer 配置 ====================
    # Transformer 编码器
    TRANSFORMER_DIM: int = 512
    TRANSFORMER_NHEAD: int = 8
    TRANSFORMER_NUM_ENCODER_LAYERS: int = 3
    TRANSFORMER_NUM_DECODER_LAYERS: int = 3
    TRANSFORMER_DROPOUT: float = 0.1
    TRANSFORMER_DIM_FEEDFORWARD: int = 2048
    
    # 位置编码
    MAX_SEQ_LEN: int = 500
    TRANSFORMER_POS_ENCODING: str = 'sine'  # 'sine' or 'learnable'
    
    # ==================== 词汇表配置 ====================
    VOCAB_SIZE: int = 2000  # 词汇表大小
    MIN_WORD_COUNT: int = 2
    
    # 特殊 token
    PAD_TOKEN: str = '<pad>'
    SOS_TOKEN: str = '<sos>'
    EOS_TOKEN: str = '<eos>'
    UNK_TOKEN: str = '<unk>'
    BLANK_TOKEN: str = '<blank>'  # CTC blank
    
    PAD_IDX: int = 0
    SOS_IDX: int = 1
    EOS_IDX: int = 2
    UNK_IDX: int = 3
    BLANK_IDX: int = 4


# ==================== 训练配置 ====================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 批次大小
    BATCH_SIZE: int = 8
    
    # 训练轮数
    NUM_EPOCHS: int = 50
    
    # 学习率
    LEARNING_RATE: float = 1e-4
    
    # 学习率调度
    LR_SCHEDULER: str = 'cosine'  # 'cosine', 'step', 'plateau', 'warmup_cosine'
    LR_WARMUP_EPOCHS: int = 5
    LR_STEP_SIZE: int = 20
    LR_GAMMA: float = 0.5
    
    # 优化器
    OPTIMIZER: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    WEIGHT_DECAY: float = 1e-4
    
    # 梯度裁剪
    GRAD_CLIP_NORM: float = 1.0
    
    # 混合精度训练
    USE_AMP: bool = True
    
    # 梯度累积 (用于小显存)
    GRAD_ACCUM_STEPS: int = 1
    
    # 标签平滑
    LABEL_SMOOTHING: float = 0.1
    
    # 随机种子
    SEED: int = 42
    
    # 评估间隔
    EVAL_INTERVAL: int = 1
    
    # 保存间隔
    SAVE_INTERVAL: int = 5
    
    # 早停
    EARLY_STOPPING_PATIENCE: int = 10
    
    # 最佳模型指标
    BEST_METRIC: str = 'bleu'  # 'loss', 'bleu', 'wer'


# ==================== 推理配置 ====================
@dataclass
class InferenceConfig:
    """推理配置"""
    # 批处理大小
    BATCH_SIZE: int = 1
    
    # 解码策略
    DECODING_STRATEGY: str = 'beam_search'  # 'greedy', 'beam_search'
    BEAM_SIZE: int = 5
    
    # 最大生成长度
    MAX_DECODING_LEN: int = 100
    
    # 长度惩罚
    LENGTH_PENALTY: float = 0.6
    
    # 重复惩罚
    REPETITION_PENALTY: float = 1.0
    
    # CTC 置信度阈值
    CTC_THRESHOLD: float = 0.5
    
    # 推理设备
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 主配置类 ====================
class Config:
    """主配置类，整合所有配置"""
    
    paths = PathConfig()
    data = DataConfig()
    model = ModelConfig()
    training = TrainingConfig()
    inference = InferenceConfig()
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def print_info(self):
        """打印配置信息"""
        print("=" * 70)
        print("  Sign Language Translation - Configuration")
        print("=" * 70)
        print(f"\n[Device]")
        print(f"  Device: {self.DEVICE}")
        print(f"  GPUs: {self.NUM_GPUS}")
        
        print(f"\n[Data]")
        print(f"  Data Root: {self.paths.DATA_ROOT}")
        print(f"  Num Frames: {self.data.NUM_FRAMES}")
        print(f"  Image Size: {self.data.IMG_SIZE}")
        
        print(f"\n[Model]")
        print(f"  ResNet: {self.model.RESNET_MODEL}")
        print(f"  LSTM Hidden: {self.model.LSTM_HIDDEN_SIZE}")
        print(f"  Transformer Dim: {self.model.TRANSFORMER_DIM}")
        
        print(f"\n[Training]")
        print(f"  Epochs: {self.training.NUM_EPOCHS}")
        print(f"  Batch Size: {self.training.BATCH_SIZE}")
        print(f"  Learning Rate: {self.training.LEARNING_RATE}")
        print(f"  AMP: {self.training.USE_AMP}")
        
        print("=" * 70)
    
    def save(self, save_path: str):
        """保存配置到文件"""
        import json
        
        config_dict = {
            'paths': {
                'DATA_ROOT': self.paths.DATA_ROOT,
                'OUTPUT_DIR': self.paths.OUTPUT_DIR,
            },
            'data': {
                'NUM_FRAMES': self.data.NUM_FRAMES,
                'IMG_SIZE': self.data.IMG_SIZE,
                'TRAIN_RATIO': self.data.TRAIN_RATIO,
            },
            'model': {
                'RESNET_MODEL': self.model.RESNET_MODEL,
                'LSTM_HIDDEN_SIZE': self.model.LSTM_HIDDEN_SIZE,
                'TRANSFORMER_DIM': self.model.TRANSFORMER_DIM,
                'VOCAB_SIZE': self.model.VOCAB_SIZE,
            },
            'training': {
                'NUM_EPOCHS': self.training.NUM_EPOCHS,
                'BATCH_SIZE': self.training.BATCH_SIZE,
                'LEARNING_RATE': self.training.LEARNING_RATE,
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Config saved to: {save_path}")


# 全局配置实例
cfg = Config()




