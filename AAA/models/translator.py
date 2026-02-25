"""
完整的手语翻译模型

参考 AAA/执行.md 文档实现:
- ResNet-50: 空间特征提取
- Bi-LSTM: 时序建模
- CTC Loss: 序列对齐 (用于 Gloss 识别)
- Transformer: 端到端翻译 (用于中文生成)

整体架构:
1. 视频帧 -> ResNet-50 -> 视觉特征 (B, T, 512)
2. 视觉特征 -> BiLSTM -> 时序特征 (B, T, 1024)
3. 时序特征 -> CTC Classifier -> Gloss 序列
4. 时序特征 -> Transformer Encoder -> 上下文表示
5. 上下文表示 -> Transformer Decoder -> 中文翻译
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入子模块
from .resnet import ResNetFeatureExtractor
from .bilstm import StackedBidirectionalLSTM, AttentionPooling
from .transformer import TransformerEncoder, TransformerDecoder, TransformerTranslator
from .ctc import CTCLoss


class SignLanguageTranslator(nn.Module):
    """
    手语翻译完整模型
    
    特点:
    1. 多任务学习: CTC 识别 + Transformer 翻译
    2. 端到端训练
    3. 可切换训练/推理模式
    """
    
    def __init__(
        self,
        num_gloss_classes: int,
        num_chinese_classes: int,
        resnet_model: str = 'resnet50',
        resnet_pretrained: bool = True,
        resnet_output_dim: int = 512,
        resnet_freeze_layers: int = 10,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        transformer_d_model: int = 512,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 3,
        transformer_dim_feedforward: int = 2048,
        transformer_dropout: float = 0.1,
        use_ctc: bool = True,
        use_transformer: bool = True,
    ):
        """
        Args:
            num_gloss_classes: Gloss 词汇表大小
            num_chinese_classes: 中文词汇表大小
            resnet_model: ResNet 模型名称
            resnet_pretrained: 是否使用预训练权重
            resnet_output_dim: ResNet 输出维度
            resnet_freeze_layers: 冻结层数
            lstm_hidden_size: LSTM 隐藏层大小
            lstm_num_layers: LSTM 层数
            lstm_dropout: LSTM dropout
            transformer_d_model: Transformer 维度
            transformer_nhead: 注意力头数
            transformer_num_layers: Transformer 层数
            transformer_dim_feedforward: FFN 维度
            transformer_dropout: Transformer dropout
            use_ctc: 是否使用 CTC 损失
            use_transformer: 是否使用 Transformer 翻译
        """
        super().__init__()
        
        self.use_ctc = use_ctc
        self.use_transformer = use_transformer
        self.num_gloss_classes = num_gloss_classes
        
        # ==================== 1. 视觉特征提取 ====================
        self.resnet = ResNetFeatureExtractor(
            model_name=resnet_model,
            pretrained=resnet_pretrained,
            output_dim=resnet_output_dim,
            freeze_layers=resnet_freeze_layers,
            dropout=0.0,
        )
        
        # ==================== 2. 时序建模 ====================
        self.lstm = StackedBidirectionalLSTM(
            input_size=resnet_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
        )
        
        # ==================== 3. CTC 分类 (可选) ====================
        if use_ctc:
            self.ctc_classifier = nn.Linear(lstm_hidden_size * 2, num_gloss_classes)
            self.ctc_loss = CTCLoss(blank=0)
        
        # ==================== 4. Transformer 翻译 (可选) ====================
        if use_transformer:
            self.transformer = TransformerTranslator(
                vocab_size=num_chinese_classes,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_encoder_layers=transformer_num_layers,
                num_decoder_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
            )
            
            # 投影层: LSTM 输出 -> Transformer 输入
            self.project_to_transformer = nn.Linear(
                lstm_hidden_size * 2, 
                transformer_d_model
            )
        
        # ==================== 5. 初始化 ====================
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def extract_features(
        self,
        video_frames: torch.Tensor,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取视频特征
        
        Args:
            video_frames: (B, C, T, H, W) 视频帧
            video_lengths: (B,) 实际帧数
            
        Returns:
            features: (B, T, feature_dim) 视觉特征
            pooled: (B, feature_dim) 池化特征
        """
        # ResNet 特征提取
        features = self.resnet(video_frames)  # (B, T, 512)
        
        # BiLSTM 时序建模
        lstm_out, hidden_states = self.lstm(features, video_lengths)
        
        # hidden_states 形状: (num_layers, B, hidden_size * 2)
        # 取最后一层的 hidden state
        h_n = hidden_states  # (num_layers, B, hidden_size * 2)
        # 取最后一层
        h_last = h_n[-1]  # (B, hidden_size * 2)
        
        # 池化: 使用最后一层的 hidden state
        pooled = h_last  # (B, hidden_size * 2)
        
        return lstm_out, pooled
    
    def forward_ctc(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CTC 前向传播
        
        Args:
            features: (B, T, hidden_size * 2) LSTM 输出
            targets: 目标标签
            input_lengths: 输入长度
            target_lengths: 目标长度
            
        Returns:
            loss: CTC 损失
            logits: 分类logits (T, B, num_classes)
        """
        # 分类
        logits = self.ctc_classifier(features)  # (B, T, num_classes)
        
        # 转换为 (T, B, C)
        logits = logits.permute(1, 0, 2)
        
        # CTC 损失
        loss = self.ctc_loss(logits, targets, input_lengths, target_lengths)
        
        return loss, logits
    
    def forward_transformer(
        self,
        features: torch.Tensor,
        tgt: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transformer 翻译前向传播
        
        Args:
            features: (B, T, hidden_size * 2) LSTM 输出
            tgt: (B, L) 目标序列
            tgt_padding_mask: 目标 padding 掩码
            src_padding_mask: 源 padding 掩码
            
        Returns:
            output: (B, L, vocab_size) 预测 logits
        """
        # 投影到 Transformer 维度
        src = self.project_to_transformer(features)  # (B, T, d_model)
        
        # Transformer 翻译
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )
        
        return output
    
    def forward(
        self,
        video_frames: torch.Tensor,
        video_lengths: Optional[torch.Tensor] = None,
        gloss_targets: Optional[torch.Tensor] = None,
        gloss_target_lengths: Optional[torch.Tensor] = None,
        chinese_targets: Optional[torch.Tensor] = None,
        chinese_target_lengths: Optional[torch.Tensor] = None,
        chinese_target_padding_mask: Optional[torch.Tensor] = None,
        mode: str = 'both',  # 'ctc', 'transformer', 'both'
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            video_frames: (B, C, T, H, W) 视频帧
            video_lengths: (B,) 实际帧数
            gloss_targets: Gloss 目标标签
            gloss_target_lengths: Gloss 目标长度
            chinese_targets: 中文目标序列
            chinese_target_lengths: 中文目标长度
            chinese_target_padding_mask: 中文目标 padding 掩码
            mode: 训练模式
            
        Returns:
            losses: 损失字典
            logits: 输出 logits
        """
        # 提取特征
        features, pooled = self.extract_features(video_frames, video_lengths)
        
        results = {
            'features': features,
            'pooled': pooled,
        }
        
        losses = {}
        
        # CTC 损失
        if mode in ['ctc', 'both'] and self.use_ctc and gloss_targets is not None:
            input_lengths = video_lengths if video_lengths is not None else \
                torch.full((video_frames.size(0),), video_frames.size(2), dtype=torch.long)
            
            ctc_loss, ctc_logits = self.forward_ctc(
                features,
                gloss_targets,
                input_lengths,
                gloss_target_lengths,
            )
            losses['ctc_loss'] = ctc_loss
            results['ctc_logits'] = ctc_logits
        
        # Transformer 翻译损失
        if mode in ['transformer', 'both'] and self.use_transformer and chinese_targets is not None:
            transformer_output = self.forward_transformer(
                features,
                chinese_targets,
                tgt_padding_mask=chinese_target_padding_mask,
            )
            
            # 计算交叉熵损失
            output = transformer_output.contiguous().view(-1, transformer_output.size(-1))
            targets = chinese_targets.contiguous().view(-1)
            
            translation_loss = F.cross_entropy(
                output, 
                targets,
                ignore_index=0,  # pad index
            )
            losses['translation_loss'] = translation_loss
            results['transformer_output'] = transformer_output
        
        # 总损失
        if losses:
            total_loss = sum(losses.values())
            losses['total_loss'] = total_loss
        
        results['losses'] = losses
        
        return results
    
    def generate(
        self,
        video_frames: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 100,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        生成中文翻译
        
        Args:
            video_frames: (B, C, T, H, W) 视频帧
            sos_idx: 开始符号索引
            eos_idx: 结束符号索引
            max_len: 最大生成长度
            video_lengths: 实际帧数
            
        Returns:
            results: 生成结果字典
        """
        # 提取特征
        features, pooled = self.extract_features(video_frames, video_lengths)
        
        # CTC 解码 (如果启用)
        ctc_predictions = None
        if self.use_ctc:
            logits = self.ctc_classifier(features)  # (B, T, num_classes)
            logits = logits.permute(1, 0, 2)  # (T, B, num_classes)
            ctc_predictions = logits
        
        # Transformer 生成
        translations = None
        if self.use_transformer:
            # 投影
            src = self.project_to_transformer(features)  # (B, T, d_model)
            
            # 生成
            translations = self.transformer.generate(
                src=src,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
                device=video_frames.device,
            )
        
        return {
            'features': features,
            'pooled': pooled,
            'ctc_predictions': ctc_predictions,
            'translations': translations,
        }
    
    def unfreeze_resnet(self, unfreeze_ratio: float = 0.5):
        """解冻 ResNet 某些层"""
        if hasattr(self.resnet, 'unfreeze_last_n_layers'):
            self.resnet.unfreeze_last_n_layers(int(10 * unfreeze_ratio))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Sign Language Translator')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--gloss-classes', type=int, default=1000)
    parser.add_argument('--chinese-classes', type=int, default=5000)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing Sign Language Translator")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SignLanguageTranslator(
        num_gloss_classes=args.gloss_classes,
        num_chinese_classes=args.chinese_classes,
        resnet_model='resnet50',
        resnet_pretrained=False,
        resnet_output_dim=512,
        lstm_hidden_size=512,
        use_ctc=True,
        use_transformer=True,
    ).to(device)
    
    total, trainable = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Size: {total * 4 / 1024 / 1024:.2f} MB")
    
    # 模拟输入
    B = args.batch_size
    T = args.num_frames
    frames = torch.randn(B, 3, T, 224, 224).to(device)
    video_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    
    # Gloss 目标
    gloss_targets = torch.randint(1, 100, (B * 5,)).to(device)
    gloss_lengths = torch.tensor([5] * B, dtype=torch.long).to(device)
    
    # 中文目标
    chinese_ids = torch.randint(1, 1000, (B, 20)).to(device)
    chinese_lengths = torch.tensor([20] * B, dtype=torch.long).to(device)
    chinese_padding_mask = torch.zeros(B, 20, dtype=torch.bool).to(device)
    
    # 测试前向传播
    print("\n1. Testing forward (CTC + Transformer):")
    result = model(
        video_frames=frames,
        video_lengths=video_lengths,
        gloss_targets=gloss_targets,
        gloss_target_lengths=gloss_lengths,
        chinese_targets=chinese_ids,
        chinese_target_lengths=chinese_lengths,
        chinese_target_padding_mask=chinese_padding_mask,
        mode='both',
    )
    
    print(f"  Features shape: {result['features'].shape}")
    print(f"  Losses: {result['losses']}")
    
    # 测试生成
    print("\n2. Testing generation:")
    gen_result = model.generate(
        video_frames=frames,
        sos_idx=1,
        eos_idx=2,
        max_len=50,
        video_lengths=video_lengths,
    )
    
    print(f"  CTC predictions shape: {gen_result['ctc_predictions'].shape if gen_result['ctc_predictions'] is not None else None}")
    print(f"  Translations shape: {gen_result['translations'].shape if gen_result['translations'] is not None else None}")
    
    print("\n" + "=" * 70)
    print("Sign Language Translator test passed!")
    print("=" * 70)

