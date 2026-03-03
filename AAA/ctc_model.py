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
        # 统一将模型中的 dropout 设置为 0.2，以在欠拟合阶段释放一定拟合能力
        #（包括 1D-CNN 与 BiLSTM 内部的 dropout）
        dropout: float = 0.2,
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
        # AdamW：学习率由外部传入，允许 weight_decay 由调用方显式控制
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # AMP 混合精度：
        # - 在 CUDA 设备上默认启用，以提升训练速度与稳定性
        # - 在 CPU 上自动禁用
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 调试标记：用于在训练的第一个 epoch 的第一个 batch 上打印核心张量信息
        self._debug_print_done: bool = False

    @staticmethod
    def _ctc_greedy_ids(
        log_probs: torch.Tensor,
        out_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        epoch: int | None = None,
        batch_idx: int | None = None,
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

                # 透视镜：每 10 个 epoch，在第一个 batch 的第一个样本上打印一次预测 vs 标签
                if (
                    epoch is not None
                    and batch_idx is not None
                    and epoch % 10 == 0
                    and batch_idx == 0
                    and b == 0
                ):
                    print(f"\n[DEBUG Epoch {epoch}] 序列对比:")
                    print(f"  预测 IDs (pred_seq): {pred_seq}")
                    print(f"  真实 IDs (tgt_seq): {tgt_seq}")

                # token 级别：按最短长度对齐后统计完全相等的 token 数
                min_len = min(len(pred_seq), len(tgt_seq))
                for i in range(min_len):
                    if pred_seq[i] == tgt_seq[i]:
                        token_correct += 1

                # 整句级别：ID 序列完全相同则记 1
                if pred_seq == tgt_seq:
                    seq_correct += 1

        return token_correct, token_total, seq_correct, num_seqs

    def train_epoch(self, dataloader, epoch: int | None = None) -> TrainStats:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_token_correct = 0
        total_token_count = 0
        total_seq_correct = 0
        total_seq_count = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train", unit="batch")):
            # ----------------------------------------------------------
            # 1. 从 batch 中取出特征与标签，并搬到指定设备上
            #    features: (B, T, C)  变长帧特征（已在 collate_fn 中对齐为同一 T）
            #    labels:   (B, L_max) padding 后的字符标签序列
            #    feature_lengths: (B,) 每个样本的真实帧长度 T_i
            #    label_lengths:   (B,) 每个样本的真实标签长度 L_i
            # ----------------------------------------------------------
            features = batch["features"].to(self.device)  # (B, T, C)
            labels = batch["labels"].to(self.device)      # (B, L_max)
            feature_lengths = batch["feature_lengths"].to(self.device)  # (B,)
            label_lengths = batch["label_lengths"].to(self.device)      # (B,)

            # ----------------------------------------------------------
            # 2. 【核心防御】过滤异常样本：要求时间步长 T 严格大于标签长度 L
            #    对于 CTC 来说，若时间步数 T <= 标签长度 L，则：
            #    - 无法在时间轴上插入足够的 blank
            #    - 会导致 loss = inf 或 NaN，进而拖垮整个 batch 的反向传播
            #
            #    这里的逻辑：
            #    - 先构造一个布尔掩码 valid_mask = (feature_lengths > label_lengths)
            #    - 若整个 batch 都是无效样本，则直接跳过该 batch
            #    - 否则仅保留合法样本子集进行后续前向 & 反向计算
            # ----------------------------------------------------------
            valid_mask = feature_lengths > label_lengths
            if not valid_mask.any():
                # 整个 batch 都不满足 T > L，直接跳过，防止 CTC 报错
                print("⚠️ 发现异常样本：该 Batch 内所有样本均满足 T <= L，已跳过该 Batch")
                continue

            if not valid_mask.all():
                # 部分样本非法，仅保留合法子集；这在极少数降采样过强或标签异常时发生
                features = features[valid_mask]
                labels = labels[valid_mask]
                feature_lengths = feature_lengths[valid_mask]
                label_lengths = label_lengths[valid_mask]
                print("⚠️ 发现部分异常样本 T <= L，已在当前 Batch 中剔除这些样本")

            self.optimizer.zero_grad()

            # ----------------------------------------------------------
            # 3. 前向传播 + CTC Loss 计算（支持 AMP 混合精度）
            #    model(features, feature_lengths) 会返回：
            #       - log_probs:    (T_max, B_eff, num_classes)
            #       - out_lengths:  (B_eff,)  每个样本真实的时间步长 T_i
            #    其中 B_eff 是过滤异常样本后剩余的 batch 大小。
            # ----------------------------------------------------------
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                log_probs, out_lengths = self.model(features, feature_lengths)

                # 额外安全检查（正常情况下 out_lengths == feature_lengths）：
                # 再次确保 out_lengths > label_lengths，从输入到输出全程满足 CTC 约束
                if not (out_lengths > label_lengths).all():
                    print("⚠️ 发现模型输出长度 T_out <= 标签长度 L，已跳过该 Batch")
                    continue

                # ------------------------------------------------------
                # 调试监控（仅在第一个 epoch 的第一个有效 batch 上触发一次）：
                # 核对 CTC 相关核心张量的形状与长度关系：
                #   - log_probs 的 shape
                #   - labels 的 shape 及前若干个标签 ID
                #   - out_lengths 与 label_lengths 的取值
                # ------------------------------------------------------
                if not self._debug_print_done:
                    print("[DEBUG] log_probs shape:", tuple(log_probs.shape))
                    print("[DEBUG] labels shape:", tuple(labels.shape))
                    # 仅打印第一个样本的前若干标签，避免输出过长
                    if labels.shape[0] > 0:
                        first_len = int(label_lengths[0].item())
                        preview = labels[0, : min(first_len, 32)].tolist()
                        print("[DEBUG] first labels (truncated):", preview)
                        print("[DEBUG] first label length:", first_len)
                    print("[DEBUG] out_lengths:", out_lengths.tolist())
                    print("[DEBUG] label_lengths:", label_lengths.tolist())
                    self._debug_print_done = True

                # nn.CTCLoss 支持 targets 为 (B, L_max)，配合 label_lengths 使用
                loss = self.criterion(
                    log_probs,  # (T_max, B_eff, C)
                    labels,     # (B_eff, L_max)
                    out_lengths,
                    label_lengths,
                )

            # ----------------------------------------------------------
            # 4. 统计当前 batch 的 CTC greedy 解码准确率
            #    - token_accuracy：逐 token 对齐后，预测 ID 与标签 ID 完全相等的比例
            #    - seq_accuracy： 整句 ID 序列完全匹配的比例
            # ----------------------------------------------------------
            (
                token_correct,
                token_total,
                seq_correct,
                num_seqs,
            ) = self._ctc_greedy_ids(
                log_probs.detach(),
                out_lengths,
                labels,
                label_lengths,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            total_token_correct += token_correct
            total_token_count += token_total
            total_seq_correct += seq_correct
            total_seq_count += num_seqs

            # ----------------------------------------------------------
            # 5. 反向传播 + 梯度裁剪 + 参数更新
            #
            # 【核心防御】梯度裁剪（Gradient Clipping）：
            # - CTC 在训练早期对齐尚不稳定时，极易出现梯度爆炸
            # - 若不加限制，参数会被巨大梯度一步“打崩”，模型长期卡在输出全 blank 的坏局部最优
            # - 这里强制约束所有参数梯度的 L2 范数不超过 max_norm=5.0
            #
            # 对于 AMP 混合精度：
            # - 先使用 scaler.scale(loss).backward() 计算缩放后的梯度
            # - 再通过 scaler.unscale_(optimizer) 将梯度还原到实值
            # - 然后再做 clip_grad_norm_，这样裁剪的是“真实梯度”
            # ----------------------------------------------------------
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # 将梯度从缩放空间还原到真实空间，便于正确执行梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
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
    def evaluate(self, dataloader, epoch: int | None = None) -> TrainStats:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_token_correct = 0
        total_token_count = 0
        total_seq_correct = 0
        total_seq_count = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Val", unit="batch")):
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            # 与 train_epoch 保持一致的安全过滤逻辑：
            # 1) 先过滤掉 T <= L 的异常样本，避免 CTC 在验证阶段产生无意义的巨大 loss
            valid_mask = feature_lengths > label_lengths
            if not valid_mask.any():
                # 整个 batch 都是异常样本，则直接跳过该 batch
                print("⚠️ [Val] 发现异常样本：该 Batch 内所有样本均满足 T <= L，已跳过该 Batch")
                continue

            if not valid_mask.all():
                # 仅保留合法子集
                features = features[valid_mask]
                labels = labels[valid_mask]
                feature_lengths = feature_lengths[valid_mask]
                label_lengths = label_lengths[valid_mask]
                print("⚠️ [Val] 发现部分异常样本 T <= L，已在当前 Batch 中剔除这些样本")

            log_probs, out_lengths = self.model(features, feature_lengths)

            # 再次确保 out_lengths > label_lengths，从输入到输出全程满足 CTC 约束
            if not (out_lengths > label_lengths).all():
                print("⚠️ [Val] 发现模型输出长度 T_out <= 标签长度 L，已跳过该 Batch")
                continue

            loss = self.criterion(log_probs, labels, out_lengths, label_lengths)
            total_loss += loss.item()
            num_batches += 1

            (
                token_correct,
                token_total,
                seq_correct,
                num_seqs,
            ) = self._ctc_greedy_ids(
                log_probs,
                out_lengths,
                labels,
                label_lengths,
                epoch=epoch,
                batch_idx=batch_idx,
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


