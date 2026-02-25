"""
BiLSTM 时序建模模块

参考 AAA/执行.md 文档实现:
- 双向 LSTM 用于时序建模
- 支持多层堆叠
- Dropout 正则化

架构说明:
- 输入: (B, T, feature_dim) - 视觉特征序列
- 输出: (B, T, hidden_size*2) - 双向特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class BidirectionalLSTM(nn.Module):
    """
    双向 LSTM 层
    
    特点:
    1. 双向读取序列，捕获前向和后向上下文
    2. 支持多层堆叠
    3. Dropout 正则化
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度 (双向，所以实际输出是 2*hidden_size)
            num_layers: LSTM 层数
            dropout: 层间 dropout 比率
            batch_first: 是否批次维度在前
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入序列 (B, T, input_size) 或 (T, B, input_size)
            lengths: 序列长度 (可选，用于 packing)
            
        Returns:
            output: LSTM 输出 (B, T, hidden_size*2)
            (h_n, c_n): 最后一层的隐藏状态
        """
        if lengths is not None:
            # 使用 packed sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, 
                lengths.cpu(), 
                batch_first=self.batch_first, 
                enforce_sorted=False
            )
        
        # LSTM 前向传播
        output, (h_n, c_n) = self.lstm(x)
        
        if lengths is not None:
            # 解包
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, 
                batch_first=self.batch_first
            )
        
        # 层归一化
        output = self.layer_norm(output)
        
        # Dropout
        output = self.dropout(output)
        
        # 拼接双向 hidden state
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (num_layers, B, hidden_size*2)
        c_n = torch.cat([c_n[0], c_n[1]], dim=1)
        
        return output, (h_n, c_n)


class StackedBidirectionalLSTM(nn.Module):
    """
    堆叠双向 LSTM
    
    特点:
    1. 多层 BiLSTM 堆叠
    2. 残差连接
    3. 层归一化
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # 多层 BiLSTM
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * 2
            layer_dropout = dropout if i < num_layers - 1 else 0
            
            self.layers.append(
                BidirectionalLSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    dropout=layer_dropout,
                    batch_first=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
            self.dropouts.append(nn.Dropout(dropout))
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, input_size)
            lengths: (B,)
            
        Returns:
            output: (B, T, hidden_size*2)
            hidden_states: 最后一层的 hidden states
        """
        hidden_states = None
        
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            residual = x
            
            x, hidden_states = layer(x, lengths)
            
            # 残差连接
            if x.shape == residual.shape:
                x = x + residual
            
            # 层归一化
            x = norm(x)
            
            # Dropout
            if i < self.num_layers - 1:
                x = dropout(x)
        
        return x, hidden_states


class AttentionPooling(nn.Module):
    """
    注意力池化层
    
    用于对序列进行加权聚合
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            mask: (B, T), True 表示 padding
            
        Returns:
            pooled: (B, input_dim)
        """
        # 计算注意力权重
        attn = self.attention(x).squeeze(-1)  # (B, T)
        
        # 应用掩码
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        
        # Softmax
        weights = F.softmax(attn, dim=-1)  # (B, T)
        
        # 加权求和
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, input_dim)
        
        return pooled


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BiLSTM Module')
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-length', type=int, default=32)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing BiLSTM Module")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 BidirectionalLSTM
    print("\n1. Testing BidirectionalLSTM:")
    bilstm = BidirectionalLSTM(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=1,
        dropout=0.3,
    ).to(device)
    
    x = torch.randn(args.batch_size, args.seq_length, args.input_size).to(device)
    lengths = torch.randint(10, args.seq_length + 1, (args.batch_size,)).to(device)
    
    output, (h_n, c_n) = bilstm(x, lengths)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  h_n: {h_n.shape}, c_n: {c_n.shape}")
    
    # 测试 StackedBidirectionalLSTM
    print("\n2. Testing StackedBidirectionalLSTM:")
    stacked_bilstm = StackedBidirectionalLSTM(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.3,
    ).to(device)
    
    output, hidden_states = stacked_bilstm(x, lengths)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    
    # 测试 AttentionPooling
    print("\n3. Testing AttentionPooling:")
    attn_pool = AttentionPooling(args.input_size * 2).to(device)
    pooled = attn_pool(output, mask=(lengths.unsqueeze(1) < args.seq_length))
    print(f"  Input:  {output.shape}")
    print(f"  Pooled: {pooled.shape}")
    
    # 统计参数
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\nParameters:")
    print(f"  BidirectionalLSTM: {count_params(bilstm):,}")
    print(f"  StackedBidirectionalLSTM: {count_params(stacked_bilstm):,}")
    print(f"  AttentionPooling: {count_params(attn_pool):,}")
    
    print("\n" + "=" * 70)
    print("BiLSTM Module test passed!")
    print("=" * 70)
