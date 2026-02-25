"""
阶段二：CTC 训练用特征数据集与词表

对应 `AAA/2.md` 中的动态 Padding + 排序方案：
- 读取阶段一提取的 .npy 特征 (T, 512)，T 为变长
- 从 CE-CSL 的 `train.csv` / `dev.csv` / `test.csv` 中读取 Gloss
- 使用 `collate_fn`:
  - 按特征长度从大到小排序 (pack_padded_sequence 要求)
  - 动态 Padding 对齐到 batch 内最长序列
  - 将标签拼接为 1D 向量，配合 CTC Loss 使用
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Vocabulary:
    """CTC 用词表（0 号为 blank）"""

    def __init__(self):
        # 0: blank, 1: unk，其余为实际 gloss token
        self.token2id: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self.id2token: Dict[int, str] = {0: "<blank>", 1: "<unk>"}
        self.n_tokens: int = 2

    def build_vocab(self, gloss_sequences: List[List[str]]) -> None:
        """从分词后的 gloss 序列列表构建词表"""
        for seq in gloss_sequences:
            for tok in seq:
                if tok not in self.token2id:
                    idx = self.n_tokens
                    self.token2id[tok] = idx
                    self.id2token[idx] = tok
                    self.n_tokens += 1

    def encode(self, tokens: List[str]) -> List[int]:
        """将 token 列表编码为 ID 列表（不包含 blank）"""
        return [self.token2id.get(t, self.token2id["<unk>"]) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """将 ID 列表解码为 token 列表（跳过 blank）"""
        tokens: List[str] = []
        for i in ids:
            if i == 0:  # blank
                continue
            tokens.append(self.id2token.get(i, "<unk>"))
        return tokens

    def __len__(self) -> int:
        return self.n_tokens


@dataclass
class SampleItem:
    feature_path: str
    gloss_ids: List[int]
    number: str
    translator: str


class CSLFeatureDataset(Dataset):
    """
    以 .npy 特征为输入的 CE-CSL 数据集

    每个样本：
    - features: (T, 512) FloatTensor
    - gloss_ids: (L,) LongTensor
    """

    def __init__(
        self,
        features_dir: str,
        label_csv: str,
        vocab: Vocabulary,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.features_dir = features_dir
        self.vocab = vocab
        self.split = split

        # 兼容 CSV 编码与可能存在的多余首行
        df = None
        for enc in ("utf-8", "gbk"):
            try:
                tmp = pd.read_csv(label_csv, encoding=enc)
            except UnicodeDecodeError:
                continue

            if "Number" not in tmp.columns and "Column1" in tmp.columns:
                tmp = pd.read_csv(label_csv, encoding=enc, header=1)

            if "Number" in tmp.columns and "Translator" in tmp.columns:
                df = tmp
                break

        if df is None:
            raise RuntimeError(f"无法在 {label_csv} 中找到 Number/Translator 列")

        self.samples: List[SampleItem] = []
        for _, row in df.iterrows():
            number = str(row["Number"])          # e.g. train-00001 / dev-00001
            translator = str(row["Translator"])  # A-L
            gloss_str = str(row["Gloss"])

            # Gloss 以 '/' 分隔
            tokens = [t.strip() for t in gloss_str.split("/") if t.strip()]
            if not tokens:
                continue

            gloss_ids = self.vocab.encode(tokens)

            feat_name = f"{number}_{translator}.npy"
            feat_path = os.path.join(self.features_dir, feat_name)
            if not os.path.exists(feat_path):
                # 对应视频尚未提取特征，跳过
                continue

            self.samples.append(
                SampleItem(
                    feature_path=feat_path,
                    gloss_ids=gloss_ids,
                    number=number,
                    translator=translator,
                )
            )

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        print(
            f"[CSLFeatureDataset] split={split}, "
            f"features_dir={features_dir}, 有效样本数={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        feats = np.load(item.feature_path)  # (T, 512)
        feats_tensor = torch.from_numpy(feats).float()

        # ---- 特征级数据增强（仅训练集使用）----
        # 参考语音 / 手语识别中的 SpecAugment 思路：
        # - 对时间维做随机遮挡（time masking），增强对局部缺失的鲁棒性
        # - 对特征通道做随机遮挡（feature masking），缓解过拟合
        if self.split == "train":
            T, C = feats_tensor.shape
            if T > 0:
                # 时间遮挡：随机选一段时间置零
                time_mask_ratio = 0.15  # 最多遮掉 15% 帧
                max_time_mask = max(1, int(T * time_mask_ratio))
                # 以一定概率做一次 time mask
                if torch.rand(1).item() < 0.5:
                    mask_len = torch.randint(1, max_time_mask + 1, (1,)).item()
                    start = torch.randint(0, max(1, T - mask_len + 1), (1,)).item()
                    feats_tensor[start : start + mask_len, :] = 0.0

                # 特征通道遮挡：随机屏蔽部分维度
                num_feat_mask = max(1, C // 16)  # 遮挡约 1/16 的通道
                if torch.rand(1).item() < 0.5:
                    mask_channels = torch.randperm(C)[:num_feat_mask]
                    feats_tensor[:, mask_channels] = 0.0
        # ------------------------------------

        gloss_ids_tensor = torch.tensor(item.gloss_ids, dtype=torch.long)
        return feats_tensor, gloss_ids_tensor


def collate_fn(batch):
    """
    CTC 训练用的批处理函数

    返回：
        features: (B, T_max, C)
        labels: (sum_L,)
        feature_lengths: (B,)
        label_lengths: (B,)
    """
    # 按序列长度从大到小排序（pack_padded_sequence 要求）
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    features, labels = zip(*batch)

    feature_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    # 动态 padding (Batch, T_max, C)
    padded_features = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True
    )

    # CTC 需要 1D 连续标签
    labels_concat = torch.cat(labels, dim=0)

    return {
        "features": padded_features,         # (B, T_max, C)
        "labels": labels_concat,            # (sum_L,)
        "feature_lengths": feature_lengths, # (B,)
        "label_lengths": label_lengths,     # (B,)
    }


