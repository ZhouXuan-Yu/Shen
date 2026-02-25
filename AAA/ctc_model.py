"""
阶段二：1D-CNN + BiLSTM + CTC 模型与训练器

对应 `AAA/2.md` 中的时序建模部分：
- 输入: (B, T, 512) 变长特征序列
- 1D-CNN 时序卷积 + MaxPool1d 将时间维减半
- BiLSTM 建模长程依赖
- 线性层输出到 CTC 类别空间，使用 nn.CTCLoss 训练
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 训练时显示进度条；若未安装 tqdm，则退化为普通迭代
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


class TemporalConvBiLSTM(nn.Module):
    """
    1D-CNN + BiLSTM 时序分类模型（为字符级 CTC 专门调整的安全结构）

    设计要点：
    - 输入特征维度默认为 512（来自 ResNet18），形状 (B, T, 512)
    - 1D-CNN 仅做“特征平滑 + 局部时序建模”，**绝不改变时间长度 T**
      * kernel_size=3, padding=1, stride=1 -> 输出长度与输入完全一致
      * 不使用 MaxPool / stride=2，避免 T 过小导致 CTC 报错（T < L）
    - BiLSTM 使用 pack_padded_sequence 提升变长序列计算效率
    - 分类层输出维度为 num_classes（= len(vocab) + 1，包含 1 个 blank）
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 1000,
        dropout: float = 0.1,
        num_cnn_layers: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 1D 时序卷积层：在时间维上做卷积，但不改变长度
        conv_layers: list[nn.Module] = []

        # 第 1 层卷积：kernel_size=3, padding=1, stride=1 保证 T 不变
        conv_layers.append(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        conv_layers.append(nn.BatchNorm1d(input_size))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.Dropout(dropout))

        # 可选的第 2 层卷积，同样保持长度不变
        if num_cnn_layers >= 2:
            conv_layers.append(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=input_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_layers.append(nn.BatchNorm1d(input_size))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(dropout))

        self.temporal_conv = nn.Sequential(*conv_layers)

        # BiLSTM：input_size=512, hidden_size=256, num_layers=2, bidirectional=True
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 分类层（输出维度 = 词表大小，含 blank）
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(
        self, x: torch.Tensor, feature_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)
            feature_lengths: (B,) 原始序列长度 T

        Returns:
            log_probs: (T, B, num_classes)
            output_lengths: (B,) 与输入相同的长度 T（不做降采样）
        """
        # Conv1d 期望输入 (B, C, T)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.temporal_conv(x)  # (B, C, T) —— 长度完全不变
        x = x.permute(0, 2, 1)  # (B, T, C)

        # output_lengths 与原始 feature_lengths 完全一致
        output_lengths = feature_lengths.clone()

        # pack_padded_sequence 需要 CPU 上的长度
        packed = nn.utils.rnn.pack_padded_sequence(
            x, output_lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (B, T, 2H)

        logits = self.fc(lstm_out)  # (B, T, C)

        # CTC 要求 (T, B, C)
        logits = logits.permute(1, 0, 2)
        log_probs = self.log_softmax(logits)

        return log_probs, output_lengths


@dataclass
class TrainStats:
    loss: float
    token_accuracy: float | None = None  # token 级别准确率（CTC 解码后 vs 标签）
    seq_accuracy: float | None = None    # 整句准确率（完全匹配的比例）


class CTCTrainer:
    """CTC 训练器

    - 使用 AdamW + weight decay
    - GPU 上默认启用 AMP 混合精度，提升训练稳定性与速度（常见于语音/视频 CTC 训练）
    """

    def __init__(
        self,
        model: TemporalConvBiLSTM,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        # 使用 AdamW，并加入适度的 weight_decay 做 L2 正则，缓解过拟合
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # AMP 混合精度（仅在 CUDA 上启用），参考语音 / 手语 CTC 常规配置
        self.use_amp = self.device.type == "cuda"
        # 兼容旧版 PyTorch：使用 torch.cuda.amp.GradScaler 接口
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @staticmethod
    def _ctc_greedy_ids(
        log_probs: torch.Tensor,
        out_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """
        计算一个 batch 内的 CTC greedy 解码准确率（在 ID 级别，不依赖词表）：
        - 折叠重复 + 去 blank（ID=0）
        - 与标签序列逐 token 对齐，统计：
          - token_correct: 预测 token == 标签 token 的个数
          - token_total: 标签 token 总数
          - seq_correct: 整句完全匹配的样本数
          - num_seqs: 有效样本数
        """
        with torch.no_grad():
            B = log_probs.shape[1]
            token_correct = 0
            token_total = 0
            seq_correct = 0
            num_seqs = 0

            for b in range(B):
                T_b = int(out_lengths[b].item())
                lp = log_probs[:T_b, b, :]  # (T_b, C)
                best_path = lp.argmax(dim=-1).tolist()

                # CTC 折叠：去重复 + 去 blank
                pred_seq: List[int] = []
                prev: int | None = None
                for idx in best_path:
                    if idx == 0:  # blank
                        prev = None
                        continue
                    if prev is not None and idx == prev:
                        continue
                    pred_seq.append(idx)
                    prev = idx

                # 这里的 labels 是 (B, L_max) 的 padding 序列
                L_b = int(label_lengths[b].item())
                tgt_seq = labels[b, :L_b].tolist()

                if L_b == 0:
                    continue

                num_seqs += 1
                token_total += L_b

                # token 级别：按最短长度对齐后统计完全相等的 token 数
                min_len = min(len(pred_seq), len(tgt_seq))
                for i in range(min_len):
                    if pred_seq[i] == tgt_seq[i]:
                        token_correct += 1

                # 整句级别：ID 序列完全相同则记 1
                if pred_seq == tgt_seq:
                    seq_correct += 1

        return token_correct, token_total, seq_correct, num_seqs

    def train_epoch(self, dataloader) -> TrainStats:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_token_correct = 0
        total_token_count = 0
        total_seq_correct = 0
        total_seq_count = 0

        for batch in tqdm(dataloader, desc="Train", unit="batch"):
            features = batch["features"].to(self.device)          # (B, T, C)
            # (B, L_max) 的 padding 标签
            labels = batch["labels"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)  # (B,)
            label_lengths = batch["label_lengths"].to(self.device)      # (B,)

            self.optimizer.zero_grad()

            # AMP 前向与反向
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                log_probs, out_lengths = self.model(features, feature_lengths)
                # nn.CTCLoss 支持 targets 为 (B, L_max)，配合 label_lengths 使用
                loss = self.criterion(
                    log_probs, labels, out_lengths, label_lengths
                )

            # 统计当前 batch 的准确率
            (
                token_correct,
                token_total,
                seq_correct,
                num_seqs,
            ) = self._ctc_greedy_ids(
                log_probs.detach(), out_lengths, labels, label_lengths
            )
            total_token_correct += token_correct
            total_token_count += token_total
            total_seq_correct += seq_correct
            total_seq_count += num_seqs

            if self.use_amp:
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        token_acc = (
            float(total_token_correct) / float(total_token_count)
            if total_token_count > 0
            else 0.0
        )
        seq_acc = (
            float(total_seq_correct) / float(total_seq_count)
            if total_seq_count > 0
            else 0.0
        )
        return TrainStats(loss=avg_loss, token_accuracy=token_acc, seq_accuracy=seq_acc)

    @torch.no_grad()
    def evaluate(self, dataloader) -> TrainStats:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_token_correct = 0
        total_token_count = 0
        total_seq_correct = 0
        total_seq_count = 0

        for batch in tqdm(dataloader, desc="Val", unit="batch"):
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            log_probs, out_lengths = self.model(features, feature_lengths)

            loss = self.criterion(
                log_probs, labels, out_lengths, label_lengths
            )
            total_loss += loss.item()
            num_batches += 1

            (
                token_correct,
                token_total,
                seq_correct,
                num_seqs,
            ) = self._ctc_greedy_ids(
                log_probs, out_lengths, labels, label_lengths
            )
            total_token_correct += token_correct
            total_token_count += token_total
            total_seq_correct += seq_correct
            total_seq_count += num_seqs

        avg_loss = total_loss / max(num_batches, 1)
        token_acc = (
            float(total_token_correct) / float(total_token_count)
            if total_token_count > 0
            else 0.0
        )
        seq_acc = (
            float(total_seq_correct) / float(total_seq_count)
            if total_seq_count > 0
            else 0.0
        )
        return TrainStats(loss=avg_loss, token_accuracy=token_acc, seq_accuracy=seq_acc)


