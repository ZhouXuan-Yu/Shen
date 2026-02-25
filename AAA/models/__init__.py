"""
手语翻译模型包
Sign Language Translation Model Package

模块:
- resnet: ResNet 视觉特征提取器
- bilstm: BiLSTM 时序建模模块
- ctc: CTC 损失和解码模块
- transformer: Transformer 翻译模块
- translator: 完整的手语翻译模型
"""

from .resnet import ResNetFeatureExtractor, VideoResNet
from .bilstm import BidirectionalLSTM, StackedBidirectionalLSTM, AttentionPooling
from .ctc import CTCLoss, greedy_decode_ctc, beam_search_decode_ctc, calculate_wer, calculate_cer
from .transformer import (
    PositionalEncoding,
    LearnedPositionalEncoding,
    MultiHeadAttention,
    TransformerEncoder,
    TransformerDecoder,
    TransformerTranslator,
)
from .translator import SignLanguageTranslator, count_parameters

__all__ = [
    # ResNet
    'ResNetFeatureExtractor',
    'VideoResNet',
    
    # BiLSTM
    'BidirectionalLSTM',
    'StackedBidirectionalLSTM',
    'AttentionPooling',
    
    # CTC
    'CTCLoss',
    'greedy_decode_ctc',
    'beam_search_decode_ctc',
    'calculate_wer',
    'calculate_cer',
    
    # Transformer
    'PositionalEncoding',
    'LearnedPositionalEncoding',
    'MultiHeadAttention',
    'TransformerEncoder',
    'TransformerDecoder',
    'TransformerTranslator',
    
    # Translator
    'SignLanguageTranslator',
    'count_parameters',
]

