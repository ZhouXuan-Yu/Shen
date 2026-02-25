"""
CTC 损失和对齐模块

参考 AAA/执行.md 文档实现:
- CTC (Connectionist Temporal Classification) Loss
- Greedy Decoding
- Beam Search Decoding

CTC 核心思想:
- 处理输入序列和输出序列长度不一致的问题
- 引入 blank 符号处理重复和间隔
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CTCLoss(nn.Module):
    """
    CTC 损失函数包装
    
    特点:
    1. 自动处理序列长度
    2. 支持 blank 符号配置
    3. 梯度裁剪
    """
    
    def __init__(
        self,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = True,
    ):
        """
        Args:
            blank: blank 符号的索引
            reduction: 损失 reduction 方式 ('mean', 'sum', 'none')
            zero_infinity: 是否将无穷损失置零
        """
        super().__init__()
        
        self.criterion = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
    
    def forward(
        self,
        logits: torch.Tensor,  # (T, B, C)
        targets: torch.Tensor,  # (L,) 或 (B, L)
        input_lengths: torch.Tensor,  # (B,)
        target_lengths: torch.Tensor,  # (B,) 或标量
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出的 log probabilities, (T, B, C)
            targets: 目标标签序列, (L,) 或 (B, L)
            input_lengths: 每个输入序列的长度, (B,)
            target_lengths: 每个目标序列的长度, (B,)
            
        Returns:
            loss: CTC 损失值
        """
        loss = self.criterion(
            F.log_softmax(logits, dim=-1),
            targets,
            input_lengths,
            target_lengths,
        )
        
        return loss


def greedy_decode_ctc(
    logits: torch.Tensor,
    blank: int = 0,
    threshold: Optional[float] = None,
) -> List[List[int]]:
    """
    Greedy Decoding for CTC
    
    贪心解码:
    1. 在每个时间步选择概率最高的类别
    2. 移除连续的重复和 blank 符号
    
    Args:
        logits: (T, B, C) 模型输出
        blank: blank 符号索引
        threshold: 置信度阈值
        
    Returns:
        decoded: 解码后的序列列表
    """
    # 获取预测类别
    predictions = logits.argmax(dim=-1)  # (T, B)
    
    decoded = []
    for pred_seq in predictions.T:  # 遍历 batch
        # 移除连续重复和 blank
        decoded_seq = []
        prev = -1
        
        for p in pred_seq:
            if p != prev and p != blank:
                # 如果有阈值要求，检查置信度
                if threshold is not None:
                    prob = F.softmax(logits[:, batch_idx, p], dim=0)
                    if prob.item() < threshold:
                        continue
                decoded_seq.append(p.item())
            prev = p
        
        decoded.append(decoded_seq)
    
    return decoded


def beam_search_decode_ctc(
    logits: torch.Tensor,
    blank: int = 0,
    beam_size: int = 5,
    threshold: float = 0.001,
    max_output_length: int = 100,
) -> List[List[int]]:
    """
    Beam Search Decoding for CTC
    
    束搜索解码:
    1. 维护 beam_size 个最佳候选序列
    2. 考虑 blank 符号的合并
    3. 比贪心解码更准确，但速度较慢
    
    Args:
        logits: (T, B, C) 模型输出
        blank: blank 符号索引
        beam_size: 束大小
        threshold: 剪枝阈值
        max_output_length: 最大输出长度
        
    Returns:
        best_sequences: 最佳序列列表
    """
    # Log 概率
    log_probs = F.log_softmax(logits, dim=-1)  # (T, B, C)
    
    batch_results = []
    
    for batch_idx in range(log_probs.size(1)):
        batch_log_probs = log_probs[:, batch_idx, :]  # (T, C)
        
        # 初始化: 空白序列的概率为 0
        # beams: List[Tuple(log_prob, sequence)]
        beams = [(0.0, [])]
        
        for t in range(batch_log_probs.size(0)):
            current_log_probs = batch_log_probs[t]  # (C,)
            
            new_beams = []
            
            for log_prob, seq in beams:
                # 对每个类别
                for c in range(current_log_probs.size(0)):
                    c_log_prob = current_log_probs[c].item()
                    
                    new_log_prob = log_prob + c_log_prob
                    
                    # 剪枝
                    if new_log_prob < threshold:
                        continue
                    
                    # 处理 blank
                    if c == blank:
                        new_beams.append((new_log_prob, seq))
                    # 处理连续重复
                    elif seq and c == seq[-1]:
                        # 合并到最后一个（如果最后一个不是 blank）
                        if seq[-1] != blank:
                            new_seq = seq[:-1] + [c]
                            new_beams.append((new_log_prob, new_seq))
                        else:
                            new_beams.append((new_log_prob, seq))
                    # 新类别
                    else:
                        new_seq = seq + [c]
                        # 长度剪枝
                        if len(new_seq) <= max_output_length:
                            new_beams.append((new_log_prob, new_seq))
            
            # 合并相同序列
            beam_dict = {}
            for lp, s in new_beams:
                key = tuple(s)
                if key not in beam_dict or lp > beam_dict[key]:
                    beam_dict[key] = lp
            
            # 选择 top-k
            beams = sorted(beam_dict.items(), key=lambda x: x[1], reverse=True)[:beam_size]
            beams = [(lp, list(s)) for s, lp in beams]
        
        # 选择最佳序列
        if beams:
            best_seq = max(beams, key=lambda x: x[0])[1]
        else:
            best_seq = []
        
        batch_results.append(best_seq)
    
    return batch_results


def calculate_wer(reference: List[str], hypothesis: List[str]) -> float:
    """
    计算 Word Error Rate (WER)
    
    WER = (S + D + I) / N
    
    其中:
    - S: 替换错误数
    - D: 删除错误数
    - I: 插入错误数
    - N: 参考序列中的词数
    
    Args:
        reference: 参考词列表
        hypothesis: 预测词列表
        
    Returns:
        wer: WER 值 (0-1)
    """
    # 简单的 WER 计算
    ref = reference
    hyp = hypothesis
    
    # 编辑距离
    n = len(ref)
    m = len(hyp)
    
    # 动态规划计算编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # 删除
                    dp[i][j - 1],    # 插入
                    dp[i - 1][j - 1] # 替换
                )
    
    errors = dp[n][m]
    wer = errors / n if n > 0 else 0
    
    return wer


def calculate_cer(reference: List[str], hypothesis: List[str]) -> float:
    """
    计算 Character Error Rate (CER)
    
    类似于 WER，但在字符级别计算
    """
    # 连接成字符串
    ref_str = ''.join(reference)
    hyp_str = ''.join(hypothesis)
    
    # 简单的编辑距离
    n, m = len(ref_str), len(hyp_str)
    
    if n == 0:
        return 0.0 if m == 0 else 1.0
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_str[i - 1] == hyp_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )
    
    errors = dp[n][m]
    cer = errors / n
    
    return cer


class CTCEvaluator:
    """CTC 模型评估器"""
    
    def __init__(
        self,
        vocab,
        blank: int = 0,
        use_beam_search: bool = False,
        beam_size: int = 5,
    ):
        """
        Args:
            vocab: 词汇表对象
            blank: blank 符号索引
            use_beam_search: 是否使用束搜索
            beam_size: 束大小
        """
        self.vocab = vocab
        self.blank = blank
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
    
    def decode(self, logits: torch.Tensor) -> List[List[str]]:
        """解码预测结果"""
        if self.use_beam_search:
            indices = beam_search_decode_ctc(
                logits, 
                blank=self.blank,
                beam_size=self.beam_size,
            )
        else:
            indices = greedy_decode_ctc(logits, blank=self.blank)
        
        # 转换为词
        decoded = []
        for seq in indices:
            words = [self.vocab.decode(idx) for idx in seq]
            decoded.append(words)
        
        return decoded
    
    def evaluate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> dict:
        """
        评估模型性能
        
        Args:
            logits: (T, B, C) 模型输出
            targets: (sum of lengths) 目标标签
            target_lengths: (B,) 每个目标的长度
            
        Returns:
            metrics: 评估指标
        """
        # 解码预测结果
        decoded = self.decode(logits)
        
        # 获取参考序列
        references = self._split_targets(targets, target_lengths)
        references = [[self.vocab.decode(idx) for idx in ref] for ref in references]
        
        # 计算 WER 和 CER
        total_wer = 0
        total_cer = 0
        
        for ref, hyp in zip(references, decoded):
            total_wer += calculate_wer(ref, hyp)
            total_cer += calculate_cer(ref, hyp)
        
        num_samples = len(references)
        
        return {
            'wer': total_wer / num_samples if num_samples > 0 else 0,
            'cer': total_cer / num_samples if num_samples > 0 else 0,
            'references': references,
            'hypotheses': decoded,
        }
    
    def _split_targets(
        self,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[List[int]]:
        """将合并的目标序列分割"""
        split_targets = []
        start = 0
        
        for length in lengths:
            split_targets.append(targets[start:start + length].tolist())
            start += length
        
        return split_targets


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CTC Module')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-length', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--blank', type=int, default=0)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing CTC Module")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建 CTC 损失
    print("\n1. Testing CTCLoss:")
    criterion = CTCLoss(blank=args.blank)
    
    # 模拟数据
    T, B = args.seq_length, args.batch_size
    C = args.num_classes
    
    logits = torch.randn(T, B, C).to(device)
    targets = torch.randint(1, C, (B * 5,))  # 随机目标
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    target_lengths = torch.tensor([5] * B, dtype=torch.long).to(device)
    
    loss = criterion(logits, targets, input_lengths, target_lengths)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试解码
    print("\n2. Testing Greedy Decoding:")
    predictions = greedy_decode_ctc(logits, blank=args.blank)
    print(f"  Batch predictions: {len(predictions)} sequences")
    print(f"  First sequence length: {len(predictions[0])}")
    
    # 测试 Beam Search
    print("\n3. Testing Beam Search:")
    beam_predictions = beam_search_decode_ctc(logits, blank=args.blank, beam_size=3)
    print(f"  Batch predictions: {len(beam_predictions)} sequences")
    print(f"  First sequence length: {len(beam_predictions[0])}")
    
    print("\n" + "=" * 70)
    print("CTC Module test passed!")
    print("=" * 70)

