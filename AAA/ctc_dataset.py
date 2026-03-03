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
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# 在 CTC 中，我们需要“语义保持 + 变体编号保留”的 Gloss 级标签，而不是单字拆分。
# 例如：
#   原始 Gloss: "一定1/身体/保养/好/要/。"
#   目标序列:   ["一定1", "身体", "保养", "好", "要"]
# 因此这里严格按 '/' 切分，并过滤句末标点与空字符串。
PUNCTUATIONS = {"。", "，", "？", "！", ".", ",", "?", "!"}


def process_gloss(gloss_str: str) -> List[str]:
    """
    将原始 Gloss 字符串解析为「Gloss 级」标记序列：

    1. 使用 '/' 作为唯一分隔符进行切分；
    2. 去除前后空白与空字符串；
    3. 过滤中英文句末标点（。 ， ？ ！ . , ? !）。

    示例：
        输入:  "一定1/身体/保养/好/要/。"
        输出:  ["一定1", "身体", "保养", "好", "要"]
    """
    if not isinstance(gloss_str, str):
        gloss_str = str(gloss_str)

    raw_tokens = gloss_str.split("/")
    tokens: List[str] = []
    for tok in raw_tokens:
        t = tok.strip()
        if not t:
            continue
        if t in PUNCTUATIONS:
            continue
        tokens.append(t)
    return tokens


class Vocabulary:
    """
    CTC 用 Gloss 级词表（0 号为 blank, 1 号为 unk）

    与 nn.CTCLoss 的约定：
    - blank 索引必须为 0（blank=0）
    - 其余有效 token 索引从 2 开始，1 预留为 <unk>
    """

    def __init__(self):
        # 0: blank, 1: unk，其余为实际 Gloss token
        self.token2id: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self.id2token: Dict[int, str] = {0: "<blank>", 1: "<unk>"}
        self.n_tokens: int = 2

    def build_vocab(self, token_sequences: List[List[str]]) -> None:
        """
        从「Gloss 级 token 序列列表」构建词表。

        参数：
        - token_sequences: 形如 [['一定1','身体','保养'], ['很好'], ...]
        """
        for seq in token_sequences:
            for tok in seq:
                if tok not in self.token2id:
                    idx = self.n_tokens
                    self.token2id[tok] = idx
                    self.id2token[idx] = tok
                    self.n_tokens += 1

    def encode(self, tokens: List[str]) -> List[int]:
        """将 token 列表编码为 ID 列表（空白仍由 CTC 在时间维度产生）"""
        return [self.token2id.get(t, self.token2id["<unk>"]) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """
        将 ID 列表解码为字符列表（自动跳过 blank=0）。

        解码时：
        - 0 -> 直接跳过
        - 其它未知索引用 <unk> 代替
        """
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

            # 1) Gloss -> Gloss 级 token 序列（严格保留变体编号，如 "一定1"）
            token_list = process_gloss(gloss_str)
            if not token_list:
                continue

            # 2) token 序列 -> 索引序列
            gloss_ids = self.vocab.encode(token_list)

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        feats = np.load(item.feature_path)  # (T, 512)
        feats_tensor = torch.from_numpy(feats).float()

        gloss_ids_tensor = torch.tensor(item.gloss_ids, dtype=torch.long)
        return feats_tensor, gloss_ids_tensor


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    CTC 训练用的批处理函数（字符级标签 + 动态对齐）。

    返回：
        features: (B, T_max, C)    经 pad_sequence 补零后的特征
        labels: (B, L_max)         经 pad_sequence 补 0(<blank>) 的标签
        feature_lengths: (B,)      每个序列真实帧数 T_i
        label_lengths: (B,)        每个序列真实标签长度 L_i

    关键点：
    - 使用 torch.nn.utils.rnn.pad_sequence 对变长序列进行补齐；
    - labels 使用 0 补齐，和 CTC 的 blank=0 保持一致；
    - 不在这里做 1D 拼接，而是交给 CTC Loss 处理 (B,L) + label_lengths。
    """
    # 可以按时长排序，提高后续 pack_padded_sequence 的效率（不强制）
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    features, labels = zip(*batch)

    feature_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    # (B, T_max, C)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    # (B, L_max)，用 blank=0 补齐
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        "features": padded_features,
        "labels": padded_labels,
        "feature_lengths": feature_lengths,
        "label_lengths": label_lengths,
    }


