"""
阶段二：1D-CNN + BiLSTM + CTC 训练脚本

总结 `AAA/2.md` 方案并做了工程化落地与小规模测试配置：
- 阶段一：`extract_features.py` 预先提取 ResNet18 帧特征 (.npy, 形状 (T, 512))
- 阶段二：本脚本加载 .npy 特征，使用 1D-CNN + BiLSTM + CTC 训练连续手语识别模型

为方便“先跑通测试一下”，默认训练轮数设为 1，且支持每个划分限制最大样本数。
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from ctc_dataset import CSLFeatureDataset, Vocabulary, collate_fn, process_gloss
from ctc_model import TemporalConvBiLSTM, CTCTrainer


@dataclass
class Config:
    # 路径配置
    DATA_ROOT: str = r"D:\Aprogress\Shen\dataset\CE-CSL\CE-CSL"
    LABEL_DIR: str = os.path.join(DATA_ROOT, "label")
    TRAIN_FEATURES: str = os.path.join(DATA_ROOT, "train_features")
    VAL_FEATURES: str = os.path.join(DATA_ROOT, "val_features")

    OUTPUT_DIR: str = os.path.join("output", "ctc_lstm")

    # 模型参数（Gloss 级 CTC，默认容量用于正式训练）
    INPUT_SIZE: int = 512
    HIDDEN_SIZE: int = 256
    NUM_LAYERS: int = 2
    # 正式训练阶段：统一使用适中的 dropout 以提升泛化（作用于 1D-CNN 与 BiLSTM）
    DROPOUT: float = 0.2

    # 训练参数（正式训练）
    # - 建议根据显存大小在 4~32 之间调整 batch_size
    BATCH_SIZE: int = 8
    # AdamW 初始学习率 + 权重衰减（配合余弦退火与 warmup）
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    # 正式训练轮数
    EPOCHS: int = 250
    # 学习率 warmup 轮数（前 WARMUP_EPOCHS 采用线性 warmup）
    WARMUP_EPOCHS: int = 10
    # 早停机制配置：patience=20，且前 MIN_EPOCHS_NO_EARLY_STOP 轮绝不触发早停
    PATIENCE: int = 20
    MIN_EPOCHS_NO_EARLY_STOP: int = 250
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 学习率调度与早停在正式训练阶段开启（见 main 中的 warmup + CosineAnnealingLR）

    # 每个划分最多使用多少样本：
    # - 正式训练阶段使用全量数据，因此设置为 None
    MAX_TRAIN_SAMPLES: int | None = None
    MAX_VAL_SAMPLES: int | None = None


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


def _safe_torch_save(obj: Dict[str, Any], path: str, desc: str = "") -> None:
    """
    更健壮的保存函数：
    - 在 Windows 上，若目标文件被其它进程占用（错误码 1224），直接 torch.save 会抛异常中断训练
    - 这里捕获这类异常，给出清晰提示并继续训练，避免整轮训练浪费
    """
    try:
        torch.save(obj, path)
    except RuntimeError as e:
        msg = str(e)
        # inline_container.cc:745 open file failed with error code: 1224
        if "open file failed" in msg and "1224" in msg:
            print(
                f"[警告] 保存 {desc or path} 失败：{e}\n"
                f"  可能原因：文件正被其它程序占用（例如被编辑器/杀毒软件/同步工具打开）。\n"
                f"  建议：关闭所有可能占用 {path} 的程序后，再重新启动训练或断点续训。"
            )
            return
        raise


def build_vocabulary() -> Vocabulary:
    """
    从 train/dev 标签构建 **Gloss 级** 词表。

    对应规划中的「Gloss -> token 序列」：
    - 先用 process_gloss 将 "一定1/身体/保养/好/要/。" 变成 ['一定1','身体','保养','好','要']
    - 再用 Vocabulary.build_vocab 统计所有出现的 Gloss token，分配索引。

    词表约定：
    - 0: <blank>（CTC blank）
    - 1: <unk>
    - 2..N: 实际 Gloss token
    """
    import pandas as pd

    vocab = Vocabulary()
    token_seqs = []

    for split in ["train", "dev"]:
        csv_path = os.path.join(cfg.LABEL_DIR, f"{split}.csv")
        df = None
        for enc in ("utf-8", "gbk"):
            try:
                tmp = pd.read_csv(csv_path, encoding=enc)
            except UnicodeDecodeError:
                continue

            if "Number" not in tmp.columns and "Column1" in tmp.columns:
                tmp = pd.read_csv(csv_path, encoding=enc, header=1)

            if "Number" in tmp.columns and "Translator" in tmp.columns:
                df = tmp
                break

        if df is None:
            raise RuntimeError(f"无法在 {csv_path} 中找到 Number/Translator 列")

        for gloss_str in df["Gloss"]:
            tokens = process_gloss(gloss_str)
            if tokens:
                token_seqs.append(tokens)

    vocab.build_vocab(token_seqs)
    print(f"Gloss-level vocabulary size (including <blank>, <unk>): {len(vocab)}")
    return vocab


def _build_ckpt_config(vocab: Vocabulary) -> Dict[str, Any]:
    """打包一份用于 checkpoint 的基础配置（模型结构相关）。"""
    return {
        "input_size": cfg.INPUT_SIZE,
        "hidden_size": cfg.HIDDEN_SIZE,
        "num_layers": cfg.NUM_LAYERS,
        "num_classes": len(vocab),
    }


def main(resume: bool = False) -> None:
    print(f"使用设备: {cfg.DEVICE}")
    print(f"数据根目录: {cfg.DATA_ROOT}")

    # 1. 构建词表
    vocab = build_vocabulary()

    # 2. 数据集与 DataLoader
    train_dataset = CSLFeatureDataset(
        features_dir=cfg.TRAIN_FEATURES,
        label_csv=os.path.join(cfg.LABEL_DIR, "train.csv"),
        vocab=vocab,
        split="train",
        max_samples=cfg.MAX_TRAIN_SAMPLES,
    )

    val_dataset = CSLFeatureDataset(
        features_dir=cfg.VAL_FEATURES,
        label_csv=os.path.join(cfg.LABEL_DIR, "dev.csv"),
        vocab=vocab,
        split="dev",
        max_samples=cfg.MAX_VAL_SAMPLES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("训练集为空，请先运行阶段一：python extract_features.py")
        return

    # 3. 创建模型
    model = TemporalConvBiLSTM(
        input_size=cfg.INPUT_SIZE,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        num_classes=len(vocab),
        dropout=cfg.DROPOUT,
    )

    # 训练器：AdamW + AMP（Cosine 学习率调度在此脚本中管理）
    trainer = CTCTrainer(
        model,
        device=cfg.DEVICE,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
    last_ckpt_path = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint.pt")
    start_epoch = 0
    epochs_no_improve = 0

    # 学习率调度器：前 cfg.WARMUP_EPOCHS 轮手动线性 warmup，之后使用 CosineAnnealingLR
    from torch.optim.lr_scheduler import CosineAnnealingLR

    cosine_epochs = max(cfg.EPOCHS - cfg.WARMUP_EPOCHS, 1)
    scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=cosine_epochs,
    )

    # 3.1 可选：从上一次的 checkpoint 继续训练（不再保存/恢复 scheduler 与早停状态）
    if resume and os.path.exists(last_ckpt_path):
        print(f"[断点续训] 加载 checkpoint: {last_ckpt_path}")
        ckpt = torch.load(
            last_ckpt_path,
            map_location=cfg.DEVICE,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(
            f" 从第 {start_epoch + 1} 个 epoch 开始继续训练 "
            f"(当前 best_val_loss={best_val_loss:.4f})"
        )

    # 4. 正式训练循环：AdamW + AMP + 余弦退火 + early stopping
    for epoch in range(start_epoch, cfg.EPOCHS):
        # 学习率调度：前 cfg.WARMUP_EPOCHS 轮线性 warmup，之后进入 Cosine 退火阶段
        if epoch < cfg.WARMUP_EPOCHS:
            warmup_factor = float(epoch + 1) / float(max(cfg.WARMUP_EPOCHS, 1))
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = cfg.LR * warmup_factor
        else:
            scheduler.step()

        print(f"\n===== Epoch {epoch + 1}/{cfg.EPOCHS} =====")
        train_stats = trainer.train_epoch(train_loader, epoch=epoch)
        val_stats = trainer.evaluate(val_loader, epoch=epoch)

        print(f"Train CTC Loss: {train_stats.loss:.4f}")
        print(f" Val  CTC Loss: {val_stats.loss:.4f}")

        # 基于 CTC greedy 解码的简单准确率指标，便于快速观察训练是否有效
        if train_stats.token_accuracy is not None:
            print(
                f"Train token acc: {train_stats.token_accuracy * 100:.2f}%, "
                f"seq acc: {train_stats.seq_accuracy * 100:.2f}%"
            )
        if val_stats.token_accuracy is not None:
            print(
                f" Val  token acc: {val_stats.token_accuracy * 100:.2f}%, "
                f"seq acc: {val_stats.seq_accuracy * 100:.2f}%"
            )

        # 保存最好模型 & 更新 early stopping 统计
        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            epochs_no_improve = 0
            _safe_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": _build_ckpt_config(vocab),
                },
                best_path,
                desc="最佳模型 best_model.pt",
            )
            print(f"[保存最佳模型] {best_path} (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # 另存一份「最近一次状态」用于断点续训
        _safe_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "vocab": vocab,
                "config": _build_ckpt_config(vocab),
            },
            last_ckpt_path,
            desc="断点续训 checkpoint last_checkpoint.pt",
        )

        # 早停机制：patience=cfg.PATIENCE，且前 cfg.MIN_EPOCHS_NO_EARLY_STOP 轮绝不触发早停
        current_epoch_idx = epoch + 1  # 以 1 开始计数
        if (
            current_epoch_idx > cfg.MIN_EPOCHS_NO_EARLY_STOP
            and epochs_no_improve >= cfg.PATIENCE
        ):
            print(
                f"\n[Early Stopping] 连续 {epochs_no_improve} 个 epoch "
                f"验证集未提升，且已超过前 {cfg.MIN_EPOCHS_NO_EARLY_STOP} 轮硬限制，提前停止训练。"
            )
            break

    print("\n训练完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="阶段二：1D-CNN + BiLSTM + CTC 训练脚本"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 output/ctc_lstm/last_checkpoint.pt 断点续训",
    )
    args = parser.parse_args()

    main(resume=args.resume)

