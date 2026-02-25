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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctc_dataset import CSLFeatureDataset, Vocabulary, collate_fn, process_gloss_to_chars
from ctc_model import TemporalConvBiLSTM, CTCTrainer


@dataclass
class Config:
    # 路径配置
    DATA_ROOT: str = r"D:\Aprogress\Shen\dataset\CE-CSL\CE-CSL"
    LABEL_DIR: str = os.path.join(DATA_ROOT, "label")
    TRAIN_FEATURES: str = os.path.join(DATA_ROOT, "train_features")
    VAL_FEATURES: str = os.path.join(DATA_ROOT, "val_features")

    OUTPUT_DIR: str = os.path.join("output", "ctc_lstm")

    # 模型参数（字符级 CTC，略微减小隐藏维度与层数，便于稳定收敛）
    INPUT_SIZE: int = 512
    HIDDEN_SIZE: int = 256
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.1

    # 训练参数
    # - 字符级标签后，标签序列变长，优化相对更难，可适当减小 batch_size
    # - 学习率使用 1e-3（或 5e-4）是 CTC 训练的常见选择
    BATCH_SIZE: int = 24
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 80
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 学习率调度与早停
    # - factor 较小：每次降低到原来的 1/10，配合更长的 patience
    # - patience 稍大：给模型更多 epoch 在同一学习率下收敛
    LR_FACTOR: float = 0.1
    LR_PATIENCE: int = 5          # 若验证集 loss 多轮不下降，则降低 LR
    MIN_LR: float = 1e-5
    # 早停耐心加长一些，避免 CTC 训练前期震荡时过早停止
    EARLY_STOP_PATIENCE: int = 15  # 若验证集在若干轮内无提升，则提前停止

    # 每个划分最多使用多少样本：
    # - 之前为了“先跑通”，只取 200/50 个样本，模型几乎学不到东西；
    # - 现在默认用全部样本进行正式训练。
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
    从 train/dev 标签构建**字符级**词表。

    对应规划中的「Gloss -> 单字序列」：
    - 先用 process_gloss_to_chars 将 "小/生命/到/家/。" 变成 ['小','生','命','到','家','。']
    - 再用 Vocabulary.build_vocab 统计所有出现的字符，分配索引。

    词表约定：
    - 0: <blank>（CTC blank）
    - 1: <unk>
    - 2..N: 实际汉字 / 标点
    """
    import pandas as pd

    vocab = Vocabulary()
    char_seqs = []

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
            chars = process_gloss_to_chars(gloss_str)
            if chars:
                char_seqs.append(chars)

    vocab.build_vocab(char_seqs)
    print(f"Char-level vocabulary size (including <blank>, <unk>): {len(vocab)}")
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

    # 训练器：使用 AdamW + weight decay
    trainer = CTCTrainer(
        model,
        device=cfg.DEVICE,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # 学习率调度器：根据验证集 CTC Loss 自适应降低学习率
    # 为兼容不同版本的 PyTorch，这里只使用位置参数，避免某些版本不支持特定关键字参数
    # ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, ...)
    scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        "min",  # mode
        cfg.LR_FACTOR,  # factor
        cfg.LR_PATIENCE,  # patience
    )

    best_val_loss = float("inf")
    best_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
    epochs_no_improve = 0
    last_ckpt_path = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint.pt")
    start_epoch = 0

    # 3.1 可选：从上一次的 checkpoint 继续训练
    if resume:
        if os.path.exists(last_ckpt_path):
            print(f"[断点续训] 加载 checkpoint: {last_ckpt_path}")
            # PyTorch 2.6 起 torch.load 默认 weights_only=True，会禁止反序列化自定义类
            # 这里的 checkpoint 完全来自本地训练，属于“可信来源”，因此显式关闭 weights_only
            ckpt = torch.load(
                last_ckpt_path,
                map_location=cfg.DEVICE,
                weights_only=False,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(
                f" 从第 {start_epoch + 1} 个 epoch 开始继续训练 "
                f"(当前 best_val_loss={best_val_loss:.4f}, "
                f"epochs_no_improve={epochs_no_improve})"
            )
        else:
            print(
                f"[断点续训] 未找到 checkpoint 文件: {last_ckpt_path}，将从头开始训练。"
            )

    # 4. 训练循环
    for epoch in range(start_epoch, cfg.EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{cfg.EPOCHS} =====")
        train_stats = trainer.train_epoch(train_loader)
        val_stats = trainer.evaluate(val_loader)

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

        # 根据验证集 loss 调整学习率
        scheduler.step(val_stats.loss)
        # 打印当前学习率（以第一个 param_group 为准）
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        print(f" 当前学习率: {current_lr:.6f}")

        # 保存最好模型
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
            print(
                f" 验证集未提升轮数: {epochs_no_improve}/{cfg.EARLY_STOP_PATIENCE}"
            )

        # 另存一份「最近一次状态」用于断点续训
        _safe_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "vocab": vocab,
                "config": _build_ckpt_config(vocab),
            },
            last_ckpt_path,
            desc="断点续训 checkpoint last_checkpoint.pt",
        )

        # 提前停止：若验证集若干轮未提升，则终止训练
        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(
                f"\n[早停] 验证集 loss 已连续 {cfg.EARLY_STOP_PATIENCE} 轮无明显提升，停止训练。"
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

