"""
在训练特征 (.npy) 上做 CTC 模型的记忆能力检查。

用法（在 AAA 目录下）：
------------------------------------------------
1）指定样本（推荐）：
    python AAA/Atest/debug_infer_on_train_feats.py ^
        --number train-00099 ^
        --translator A

2）从 CSV 中按索引选样本（0 开始）：
    python AAA/Atest/debug_infer_on_train_feats.py ^
        --index 0

脚本会：
- 直接从 train_features 读取对应的 .npy 特征（与训练时完全一致）；
- 从 train.csv 读取对应的 Gloss，并用 process_gloss 得到 Gloss token 序列；
- 加载 output/ctc_lstm/best_model.pt；
- 在该特征上做一次前向 + CTC greedy 解码；
- 打印：
  - 标签 token 序列
  - 预测 token 序列
  - 逐 token 对齐的对比与简单准确率
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# 确保可以从 AAA 目录导入 ctc_dataset / ctc_model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AAA_ROOT = os.path.dirname(CURRENT_DIR)
if AAA_ROOT not in sys.path:
    sys.path.insert(0, AAA_ROOT)

from ctc_dataset import Vocabulary, process_gloss
from ctc_model import TemporalConvBiLSTM


DATA_ROOT = r"D:\Aprogress\Shen\dataset\CE-CSL\CE-CSL"
LABEL_DIR = os.path.join(DATA_ROOT, "label")
TRAIN_FEATURES_DIR = os.path.join(DATA_ROOT, "train_features")
# 默认权重放在 AAA/output/... 下。用绝对路径，避免因当前工作目录不同导致找不到文件。
DEFAULT_CKPT = os.path.join(AAA_ROOT, "output", "ctc_lstm", "best_model.pt")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[TemporalConvBiLSTM, Vocabulary]:
    """复用训练时的配置，加载模型与词表。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型权重文件：{checkpoint_path}\n"
            f"请先运行 train_lstm.py 完成训练，并确认 best_model.pt 已保存。"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab: Vocabulary = ckpt["vocab"]
    cfg = ckpt["config"]

    model = TemporalConvBiLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocab


def ctc_greedy_decode_ids(
    log_probs: torch.Tensor,
) -> List[int]:
    """
    和训练时一致的 CTC greedy 解码（只在 ID 级，不依赖词表）：
    - 每个时间步取 argmax ID
    - 折叠重复
    - 去掉 blank=0
    """
    log_probs = log_probs.squeeze(1)  # (T, C)
    best_path = log_probs.argmax(dim=-1).tolist()

    collapsed: List[int] = []
    prev = None
    for idx in best_path:
        if idx == 0:  # blank
            prev = None
            continue
        if prev is not None and idx == prev:
            continue
        collapsed.append(idx)
        prev = idx
    return collapsed


def compare_sequences(pred_ids: List[int], tgt_ids: List[int]) -> None:
    """逐 token 对齐输出预测 vs 标签以及简单准确率。"""
    min_len = min(len(pred_ids), len(tgt_ids))
    token_correct = 0
    print("\n==== 逐 token 对齐（索引对比）====")
    print("idx\tpred_id\tgold_id\tmatch")
    for i in range(min_len):
        # Windows 下很多终端默认是 GBK，打印 ✓/✗ 容易触发 UnicodeEncodeError
        match = "OK" if pred_ids[i] == tgt_ids[i] else "NO"
        if pred_ids[i] == tgt_ids[i]:
            token_correct += 1
        print(f"{i}\t{pred_ids[i]}\t{tgt_ids[i]}\t{match}")

    token_acc = token_correct / max(len(tgt_ids), 1)
    print(f"\nToken-level accuracy (对齐到标签长度): {token_acc * 100:.2f}%")
    print(f"预测序列长度: {len(pred_ids)}, 标签长度: {len(tgt_ids)}")
    print(f"预测 ID 列表: {pred_ids}")
    print(f"标签  ID 列表: {tgt_ids}")


def debug_on_sample(
    number: str,
    translator: str,
    checkpoint_path: str = DEFAULT_CKPT,
) -> None:
    """
    使用指定 Number 和 Translator 的训练样本，直接在 .npy 特征上做推理对比。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型与词表
    model, vocab = load_checkpoint(checkpoint_path, device)
    print(f"词表大小（含 <blank>, <unk>）: {len(vocab)}")

    # 2. 构造特征路径并加载 .npy
    feat_name = f"{number}_{translator}.npy"
    feat_path = os.path.join(TRAIN_FEATURES_DIR, feat_name)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"找不到特征文件: {feat_path}")

    feats = np.load(feat_path)  # (T, 512)
    if feats.ndim != 2:
        raise ValueError(f"特征形状应为 (T, 512)，当前为 {feats.shape}")
    print(f"加载特征: {feat_path}, 形状: {feats.shape}")

    # 3. 从 train.csv 中读取对应标签，并转为 Gloss 级 ID 序列
    train_csv = os.path.join(LABEL_DIR, "train.csv")
    df = None
    for enc in ("utf-8", "gbk"):
        try:
            tmp = pd.read_csv(train_csv, encoding=enc)
        except UnicodeDecodeError:
            continue

        if "Number" not in tmp.columns and "Column1" in tmp.columns:
            tmp = pd.read_csv(train_csv, encoding=enc, header=1)

        if "Number" in tmp.columns and "Translator" in tmp.columns:
            df = tmp
            break

    if df is None:
        raise RuntimeError(f"无法在 {train_csv} 中找到 Number/Translator 列")

    row = df[(df["Number"].astype(str) == number) & (df["Translator"].astype(str) == translator)]
    if row.empty:
        raise RuntimeError(f"在 train.csv 中找不到样本: Number={number}, Translator={translator}")

    gloss_str = str(row.iloc[0]["Gloss"])
    gold_tokens = process_gloss(gloss_str)
    gold_ids = vocab.encode(gold_tokens)

    print("\n==== 标签（Gloss 级）====")
    print(f"原始 Gloss: {gloss_str}")
    print(f"Gloss tokens: {gold_tokens}")
    print(f"Gloss token IDs: {gold_ids}")

    # 4. 前向 + CTC 解码
    with torch.no_grad():
        feats_tensor = torch.from_numpy(feats).float().unsqueeze(0).to(device)  # (1, T, C)
        feat_lens = torch.tensor([feats.shape[0]], dtype=torch.long).to(device)

        log_probs, out_lens = model(feats_tensor, feat_lens)

        # 简单 blank 统计，和 demo_infer 中保持一致，检查是否几乎全 blank
        lp = log_probs.squeeze(1)
        probs = lp.exp()
        blank_probs = probs[:, 0]
        blank_mean = blank_probs.mean().item()
        blank_high_ratio = (blank_probs > 0.9).float().mean().item()
        best_ids_all = probs.argmax(dim=-1)
        blank_argmax_ratio = (best_ids_all == 0).float().mean().item()
        print(
            f"\n[Debug] T'={lp.shape[0]}, 平均blank概率={blank_mean:.3f}, "
            f"blank>0.9比例={blank_high_ratio:.3f}, "
            f"argmax为blank比例={blank_argmax_ratio:.3f}"
        )

        pred_ids = ctc_greedy_decode_ids(log_probs)
        pred_tokens = [vocab.id2token.get(i, "<unk>") for i in pred_ids]

    print("\n==== 预测（Gloss 级）====")
    print(f"预测 token IDs: {pred_ids}")
    print(f"预测 tokens   : {pred_tokens}")
    print(f"预测拼接句子  : {''.join(pred_tokens) if pred_tokens else '(空预测)'}")

    # 5. 逐 token 对齐对比
    compare_sequences(pred_ids, gold_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在训练特征 (.npy) 上做 CTC 记忆能力检查"
    )
    parser.add_argument(
        "--number",
        type=str,
        default=None,
        help="样本编号，如 train-00099（优先使用此方式）",
    )
    parser.add_argument(
        "--translator",
        type=str,
        default=None,
        help="翻译员标记，如 A、B、C 等（与特征文件名中的后缀一致）",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="可选：从 train.csv 中按索引选样本（0 开始），当未提供 --number 时使用",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT,
        help=f"模型权重路径（默认：{DEFAULT_CKPT}）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    number = args.number
    translator = args.translator

    # 如果没提供 number，就用 index 从 CSV 里选一个样本
    if number is None or translator is None:
        if args.index is None:
            raise SystemExit(
                "必须至少提供以下两种方式之一：\n"
                "  1）--number train-00099 --translator A\n"
                "  2）--index 0（脚本会从 train.csv 里取第 index 条样本）"
            )

        train_csv = os.path.join(LABEL_DIR, "train.csv")
        df = None
        for enc in ("utf-8", "gbk"):
            try:
                tmp = pd.read_csv(train_csv, encoding=enc)
            except UnicodeDecodeError:
                continue

            if "Number" not in tmp.columns and "Column1" in tmp.columns:
                tmp = pd.read_csv(train_csv, encoding=enc, header=1)

            if "Number" in tmp.columns and "Translator" in tmp.columns:
                df = tmp
                break

        if df is None:
            raise RuntimeError(f"无法在 {train_csv} 中找到 Number/Translator 列")

        if args.index < 0 or args.index >= len(df):
            raise IndexError(f"--index 超出范围：0 <= index < {len(df)}")

        row = df.iloc[args.index]
        number = str(row["Number"])
        translator = str(row["Translator"])
        print(f"从 index={args.index} 选中样本: Number={number}, Translator={translator}")

    print(f"\n=== 使用训练样本进行记忆检查 ===")
    print(f"Number     : {number}")
    print(f"Translator : {translator}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"特征目录   : {TRAIN_FEATURES_DIR}")

    debug_on_sample(number, translator, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

