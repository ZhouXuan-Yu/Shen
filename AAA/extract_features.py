"""
阶段一：特征预提取脚本

参考 `AAA/2.md` 的两阶段方案：
- 使用 ResNet18 提取视频帧特征
- 保留视频原始帧数 (变长序列)，不压缩到固定帧数
- 每个视频保存一个 .npy 特征文件，形状为 (T, 512)

注意：
- 默认路径已适配当前项目与 CE-CSL 数据集结构
- 为了“先跑通测试一下”，提供 --max-samples 参数，只抽取少量样本
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm
import pandas as pd


class Config:
    """路径与模型配置"""

    # 数据根目录
    DATA_ROOT = r"D:\Aprogress\Shen\dataset\CE-CSL\CE-CSL"

    # 子目录
    VIDEO_DIR = os.path.join(DATA_ROOT, "video")
    LABEL_DIR = os.path.join(DATA_ROOT, "label")

    # 特征输出目录
    OUTPUT_ROOT = DATA_ROOT
    TRAIN_FEATURES = os.path.join(OUTPUT_ROOT, "train_features")
    VAL_FEATURES = os.path.join(OUTPUT_ROOT, "val_features")
    TEST_FEATURES = os.path.join(OUTPUT_ROOT, "test_features")

    # 模型配置
    RESNET_MODEL = "resnet18"
    FEATURE_DIM = 512
    IMAGE_SIZE = 224
    FRAME_SAMPLE_RATE = 1  # 每 1 帧取 1 帧（保留全部时序信息）
    # 特征提取时的帧 batch 大小（防止一次性把整段视频所有帧放进 GPU 导致 OOM）
    FEATURE_BATCH_SIZE = 16

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


class ResNetFeatureExtractor:
    """基于 ResNet 的帧级特征提取器"""

    def __init__(self, model_name: str = "resnet18"):
        if model_name == "resnet18":
            # 使用新版 weights 接口，等价于过去的 pretrained=True
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            out_dim = 512
        elif model_name == "resnet50":
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            out_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # 去掉最后的 FC 层与 avgpool，保留到全局池化前一层；后面自己做全局池化
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # (B, C, H, W)
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.to(cfg.DEVICE)

        # 特征维度（与 ResNet 输出通道数一致：resnet18 为 512）
        self.feature_dim = out_dim

        # 单次送入 backbone 的最大帧数
        self.batch_size = cfg.FEATURE_BATCH_SIZE

    @torch.no_grad()
    def extract(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Args:
            frames: list of (H, W, C) in BGR
        Returns:
            features: (T, 512)
        """
        if len(frames) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # 先在 CPU 上做预处理，得到一组 (C, H, W) 的 numpy 数组
        processed: list[np.ndarray] = []
        for frame in frames:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize (保持短边) + 中心裁剪到 IMAGE_SIZE
            frame = cv2.resize(
                frame, (cfg.IMAGE_SIZE + 32, cfg.IMAGE_SIZE + 32)
            )
            h, w = frame.shape[:2]
            sh = (h - cfg.IMAGE_SIZE) // 2
            sw = (w - cfg.IMAGE_SIZE) // 2
            frame = frame[sh : sh + cfg.IMAGE_SIZE, sw : sw + cfg.IMAGE_SIZE]
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array(
                [0.229, 0.224, 0.225]
            )
            frame = frame.transpose(2, 0, 1)  # C,H,W
            processed.append(frame)

        # 分批送入 GPU，避免一次性把整段视频的所有帧打成一个大 batch 导致显存溢出
        all_feats: list[torch.Tensor] = []
        T = len(processed)
        bs = self.batch_size

        for start in range(0, T, bs):
            end = min(start + bs, T)
            batch_np = np.stack(processed[start:end], axis=0)  # (B, C, H, W)
            batch = torch.from_numpy(batch_np).float().to(cfg.DEVICE)

            feats = self.backbone(batch)  # (B, C, H', W')
            # 全局平均池化到 (B, C, 1, 1) 再拉平成 (B, C)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))  # (B, C, 1,1)
            feats = feats.view(feats.size(0), -1)  # (B, C=self.feature_dim)

            all_feats.append(feats.cpu())

            # 对于 GPU，适当释放缓存，进一步降低 OOM 风险
            if cfg.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        feats_all = torch.cat(all_feats, dim=0)  # (T, 512)
        return feats_all.numpy()


def load_video_frames(video_path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """按采样率读取完整视频帧（变长）"""
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % cfg.FRAME_SAMPLE_RATE == 0:
            frames.append(frame)
        idx += 1
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def get_video_path(split: str, translator: str, number: str) -> str:
    """
    构造视频路径
    CSV 中的 Number 形如: train-00001 / dev-00001 / test-00001
    实际文件名与其一致: {Number}.mp4
    """
    filename = f"{number}.mp4"
    return os.path.join(cfg.VIDEO_DIR, split, translator, filename)


def extract_split_features(
    extractor: ResNetFeatureExtractor,
    split: str,
    max_samples: int | None = None,
) -> None:
    """
    提取单个数据划分 (train / dev / test) 的特征
    """
    label_file = os.path.join(cfg.LABEL_DIR, f"{split}.csv")
    if not os.path.exists(label_file):
        raise FileNotFoundError(label_file)

    # 兼容中文 CSV 编码与可能存在的多余首行 (Column1,...)
    df = None
    for enc in ("utf-8", "gbk"):
        try:
            tmp = pd.read_csv(label_file, encoding=enc)
        except UnicodeDecodeError:
            continue

        # 如果没有 Number 列但有 Column1，说明需要跳过第一行
        if "Number" not in tmp.columns and "Column1" in tmp.columns:
            tmp = pd.read_csv(label_file, encoding=enc, header=1)

        if "Number" in tmp.columns and "Translator" in tmp.columns:
            df = tmp
            break

    if df is None:
        raise RuntimeError(f"无法在 {label_file} 中找到 Number/Translator 列")

    if split == "train":
        out_dir = cfg.TRAIN_FEATURES
    elif split == "dev":
        out_dir = cfg.VAL_FEATURES
    elif split == "test":
        out_dir = cfg.TEST_FEATURES
    else:
        raise ValueError(f"Unknown split: {split}")

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n开始提取 {split} 集特征，共 {len(df)} 条标注...")

    num_processed = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        number = str(row["Number"])          # e.g. train-00001
        translator = str(row["Translator"])  # A-L

        video_path = get_video_path(split, translator, number)
        feature_name = f"{number}_{translator}.npy"
        feature_path = os.path.join(out_dir, feature_name)

        # 已存在则跳过
        if os.path.exists(feature_path):
            num_processed += 1
            if max_samples is not None and num_processed >= max_samples:
                break
            continue

        if not os.path.exists(video_path):
            print(f"[WARN] 视频不存在: {video_path}")
            continue

        frames = load_video_frames(video_path)
        if len(frames) == 0:
            print(f"[WARN] 视频为空: {video_path}")
            continue

        feats = extractor.extract(frames)  # (T, 512)
        np.save(feature_path, feats)

        num_processed += 1
        if max_samples is not None and num_processed >= max_samples:
            break

    print(f"{split} 集特征提取完成，实际处理 {num_processed} 个视频，保存至 {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="CE-CSL 特征预提取 (阶段一)")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev",
        help="需要处理的数据划分，逗号分隔: train,dev,test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="每个划分最多处理多少个视频，用于快速测试；设为 0 或负数表示处理全部",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    print(f"使用设备: {cfg.DEVICE}")
    print(f"ResNet 模型: {cfg.RESNET_MODEL}")
    print(f"数据根目录: {cfg.DATA_ROOT}")

    extractor = ResNetFeatureExtractor(cfg.RESNET_MODEL)

    for split in splits:
        extract_split_features(extractor, split, max_samples=max_samples)

    print("\n所有特征提取流程结束。")


if __name__ == "__main__":
    main()

