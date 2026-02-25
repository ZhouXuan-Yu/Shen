"""
Transformer 翻译模块

参考 AAA/执行.md 文档实现:
- Transformer Encoder: 多头自注意力
- Transformer Decoder: 自回归生成
- 位置编码: 正弦位置编码或可学习位置编码

架构说明:
- 输入: (B, T, feature_dim) - 编码器特征序列
- 输出: (B, L, vocab_size) - 翻译结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    使用正弦和余弦函数生成位置编码:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 注册位置编码 buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            
        Returns:
            (B, L, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    可学习位置编码
    
    将位置编码作为可学习的参数
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            
        Returns:
            (B, L, d_model)
        """
        seq_len = x.size(1)
        
        # 创建位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, L, d_model)
        
        return self.dropout(x + pos_emb)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    特点:
    1. 并行计算多个注意力头
    2. 缩放点积注意力
    3. Dropout 正则化
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, L, d_model)
            key: (B, S, d_model)
            value: (B, S, d_model)
            key_padding_mask: (B, S) True for padding
            attn_mask: (L, S) True for mask
            
        Returns:
            output: (B, L, d_model)
            attention: (B, nhead, L, S)
        """
        B, L, _ = query.shape
        _, S, _ = key.shape
        
        # 线性投影并分割为多头
        q = self.w_q(query).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, L, h)
        k = self.w_k(key).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        attn = torch.matmul(q / self.scale, k.transpose(-2, -1))  # (B, nhead, L, S)
        
        # 应用掩码
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)
        
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        output = torch.matmul(attn, v)  # (B, nhead, L, h)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.w_o(output)
        
        return output, attn


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    Transformer 中的 FFN 子层
    """
    
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        # 自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, dim_feedforward)
        
        # FFN
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 自注意力 + 残差连接
        residual = src
        src_attn, _ = self.self_attn(
            src, src, src,
            key_padding_mask=src_padding_mask,
        )
        src = self.norm1(residual + self.dropout(src_attn))
        
        # FFN + 残差连接
        residual = src
        src = self.norm2(residual + self.ffn(src))
        
        return src


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder 层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 添加位置编码
        src = self.pos_encoding(src)
        
        # 通过各层
        for layer in self.layers:
            src = layer(src, src_padding_mask)
        
        # 最终归一化
        src = self.norm(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        # 自注意力 (Masked)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, dim_feedforward)
        
        # 交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout, dim_feedforward)
        
        # FFN
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 自注意力
        residual = tgt
        tgt_attn, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_padding_mask,
        )
        tgt = self.norm1(residual + self.dropout(tgt_attn))
        
        # 交叉注意力
        residual = tgt
        cross_attn, attn_weights = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_padding_mask,
        )
        tgt = self.norm2(residual + self.dropout(cross_attn))
        
        # FFN
        residual = tgt
        tgt = self.norm3(residual + self.ffn(tgt))
        
        return tgt, attn_weights


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token 嵌入
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder 层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.norm = nn.LayerNorm(d_model)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成后续掩码 (防止看到未来信息)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tgt: 目标序列 (B, L)
            memory: 编码器输出 (B, S, d_model)
            tgt_mask: 目标序列掩码
            tgt_padding_mask: 目标 padding 掩码
            memory_padding_mask: 编码器 padding 掩码
            
        Returns:
            output: (B, L, vocab_size)
            attention_weights: 最后一层的注意力权重
        """
        # Token 嵌入
        tgt = self.tok_embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        tgt = self.pos_encoding(tgt)
        
        # 生成掩码
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # 通过各层
        attention_weights = None
        for layer in self.layers:
            tgt, attention_weights = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )
        
        # 输出投影
        output = self.fc_out(self.norm(tgt))
        
        return output, attention_weights
    
    def generate(
        self,
        sos_idx: int,
        eos_idx: int,
        memory: torch.Tensor,
        max_len: int = 100,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            sos_idx: 开始符号索引
            eos_idx: 结束符号索引
            memory: 编码器输出
            max_len: 最大生成长度
            device: 设备
            
        Returns:
            generated: 生成的序列 (L,)
        """
        if device is None:
            device = memory.device
        
        generated = [sos_idx]
        
        for _ in range(max_len):
            tgt = torch.tensor([generated], device=device)
            
            with torch.no_grad():
                output, _ = self.forward(
                    tgt,
                    memory,
                    tgt_mask=None,
                )
            
            next_token = output.argmax(dim=-1)[:, -1].item()
            
            if next_token == eos_idx:
                break
            
            generated.append(next_token)
        
        return generated[1:]  # 移除 sos


class TransformerTranslator(nn.Module):
    """
    Transformer 翻译模型
    
    完整的编码器-解码器架构
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 编码器
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: 源序列 (B, S, d_model)
            tgt: 目标序列 (B, L)
            src_padding_mask: 源 padding 掩码
            tgt_padding_mask: 目标 padding 掩码
            
        Returns:
            output: (B, L, vocab_size)
        """
        # 编码
        memory = self.encoder(src, src_padding_mask)
        
        # 解码
        output, _ = self.decoder(
            tgt, memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
        )
        
        return output
    
    def encode(self, src: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码源序列"""
        return self.encoder(src, src_padding_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """解码目标序列"""
        output, _ = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        return output
    
    def generate(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 100,
        device: torch.device = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            src: 源序列 (B, S, d_model)
            sos_idx: 开始符号索引
            eos_idx: 结束符号索引
            max_len: 最大生成长度
            device: 设备
            src_padding_mask: 源 padding 掩码
            
        Returns:
            generated: 生成的序列 (B, L)
        """
        if device is None:
            device = src.device
        
        batch_size = src.size(0)
        
        # 编码
        memory = self.encoder(src, src_padding_mask)
        
        # 自回归生成
        generated = []
        
        for b in range(batch_size):
            gen = self.decoder.generate(
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                memory=memory[b:b+1],
                max_len=max_len,
                device=device,
            )
            generated.append(gen)
        
        # 填充到相同长度
        max_gen_len = max(len(g) for g in generated)
        padded = torch.zeros(batch_size, max_gen_len, dtype=torch.long, device=device)
        
        for b, g in enumerate(generated):
            padded[b, :len(g)] = torch.tensor(g, device=device)
        
        return padded


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Transformer Module')
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-length', type=int, default=32)
    parser.add_argument('--tgt-length', type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing Transformer Module")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 Transformer Encoder
    print("\n1. Testing TransformerEncoder:")
    encoder = TransformerEncoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    
    src = torch.randn(args.batch_size, args.seq_length, args.d_model).to(device)
    src_padding_mask = torch.zeros(args.batch_size, args.seq_length, dtype=torch.bool).to(device)
    src_padding_mask[:, -5:] = True  # 最后 5 个是 padding
    
    memory = encoder(src, src_padding_mask)
    print(f"  Input:  {src.shape}")
    print(f"  Output: {memory.shape}")
    
    # 测试 Transformer Decoder
    print("\n2. Testing TransformerDecoder:")
    decoder = TransformerDecoder(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    
    tgt = torch.randint(0, args.vocab_size, (args.batch_size, args.tgt_length)).to(device)
    tgt_padding_mask = torch.zeros(args.batch_size, args.tgt_length, dtype=torch.bool).to(device)
    
    output, attn = decoder(tgt, memory, tgt_padding_mask=tgt_padding_mask)
    print(f"  Input tgt:  {tgt.shape}")
    print(f"  Input memory: {memory.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Attention shape: {attn.shape}")
    
    # 测试完整翻译模型
    print("\n3. Testing TransformerTranslator:")
    translator = TransformerTranslator(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
    ).to(device)
    
    output = translator(src, tgt, src_padding_mask, tgt_padding_mask)
    print(f"  Output: {output.shape}")
    
    # 测试生成
    print("\n4. Testing Generation:")
    sos_idx = 1
    eos_idx = 2
    generated = translator.generate(src, sos_idx, eos_idx, max_len=30, device=device)
    print(f"  Generated shape: {generated.shape}")
    
    # 统计参数
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\nParameters:")
    print(f"  Encoder: {count_params(encoder):,}")
    print(f"  Decoder: {count_params(decoder):,}")
    print(f"  Translator: {count_params(translator):,}")
    
    print("\n" + "=" * 70)
    print("Transformer Module test passed!")
    print("=" * 70)

