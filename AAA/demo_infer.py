"""
简单测试 / Demo 推理脚本

功能：
- 加载阶段二训练好的 CTC 模型 (`output/ctc_lstm/best_model.pt`)
- 复用阶段一的 ResNet18 特征提取
- 支持：
  - 直接输入一段手语视频（mp4）
  - 或者输入单张图片（视为只有一帧的“短视频”）
- 输出预测的 Gloss 序列（一个词 / 一句话是什么含义）

用法示例（在 AAA 目录下）：
------------------------------------------------
1）用视频测试：
    python demo_infer.py --video D:\\path\\to\\your_video.mp4

2）用图片测试：
    python demo_infer.py --image D:\\path\\to\\your_image.jpg

3）指定模型路径（如果不在默认位置）：
    python demo_infer.py --video xxx.mp4 --checkpoint D:\\Aprogress\\Shen\\AAA\\output\\ctc_lstm\\last_checkpoint.pt
"""

import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from ctc_model import TemporalConvBiLSTM
from ctc_dataset import Vocabulary
from extract_features import ResNetFeatureExtractor, load_video_frames, cfg as feat_cfg


AAA_ROOT = os.path.dirname(os.path.abspath(__file__))
# 默认权重放在 AAA/output/... 下。用绝对路径，避免因当前工作目录不同导致找不到文件。
# 这里默认使用训练脚本保存的最佳模型 best_model.pt
DEFAULT_CKPT = os.path.join(AAA_ROOT, "output", "ctc_lstm", "best_model.pt")


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[TemporalConvBiLSTM, Vocabulary, torch.device]:
    """加载训练好的模型与词表"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型权重文件：{checkpoint_path}\n"
            f"请先运行 train_lstm.py 完成训练，并确认 best_model.pt 已保存。"
        )

    # PyTorch 2.6 之后 torch.load 默认 weights_only=True，
    # 只能安全加载“纯权重字典”，而我们在训练时额外保存了自定义的 Vocabulary 对象，
    # 因此这里显式设置 weights_only=False 来允许反序列化自定义类。
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

    return model, vocab, device


@torch.no_grad()
def ctc_greedy_decode(
    log_probs: torch.Tensor,
    vocab: Vocabulary,
) -> List[str]:
    """
    最简单的 CTC greedy 解码（字符级）：
    - 每个时间步取概率最大的类别
    - 折叠重复的 ID
    - 去掉 blank（ID=0）

    Args:
        log_probs: (T, 1, C)  或 (T, B, C) 这里只用 B=1
    Returns:
        tokens: 解码后的字符列表（汉字 / 标点等）
    """
    # (T, 1, C) -> (T, C)
    log_probs = log_probs.squeeze(1)  # (T, C)
    best_path = log_probs.argmax(dim=-1).tolist()  # List[int]

    collapsed: List[int] = []
    prev: Optional[int] = None
    for idx in best_path:
        if idx == 0:  # blank
            prev = None
            continue
        if prev is not None and idx == prev:
            continue
        collapsed.append(idx)
        prev = idx

    tokens: List[str] = []
    for i in collapsed:
        tok = vocab.id2token.get(i, "<unk>")
        if tok not in ("<blank>",):
            tokens.append(tok)
    return tokens


@torch.no_grad()
def predict_from_features(
    features: np.ndarray,
    model: TemporalConvBiLSTM,
    vocab: Vocabulary,
    device: torch.device,
) -> Tuple[List[str], str]:
    """
    给定一段特征序列 (T, 512)，做一次前向 + 字符级 CTC 解码

    返回：
    - tokens: 字符列表，例如 ['小', '生', '命', '到', '家', '。']
    - sentence: 直接拼接后的中文句子，例如 "小生命到家。"
    """
    if features.ndim != 2:
        raise ValueError(f"features 形状应为 (T, 512)，当前为 {features.shape}")
    if features.shape[0] == 0:
        raise ValueError("输入特征为空，无法推理。")

    feats = torch.from_numpy(features).float().unsqueeze(0).to(device)  # (1, T, C)
    feat_lens = torch.tensor([features.shape[0]], dtype=torch.long).to(device)

    log_probs, out_lens = model(feats, feat_lens)  # log_probs: (T', 1, C)

    # ---- 简单调试输出：检查模型是否几乎全部输出 blank ----
    # 目的：快速判断“训练方向是否有问题”，而不改变原有推理逻辑。
    with torch.no_grad():
        lp = log_probs.squeeze(1)  # (T', C)
        probs = lp.exp()
        T, C = probs.shape
        blank_probs = probs[:, 0]
        blank_mean = blank_probs.mean().item()
        blank_high_ratio = (blank_probs > 0.9).float().mean().item()
        best_ids = probs.argmax(dim=-1)
        blank_argmax_ratio = (best_ids == 0).float().mean().item()
        print(
            f"[Debug] T'={T}, 平均blank概率={blank_mean:.3f}, "
            f"blank>0.9比例={blank_high_ratio:.3f}, "
            f"argmax为blank比例={blank_argmax_ratio:.3f}"
        )
    # ----------------------------------------------------

    tokens = ctc_greedy_decode(log_probs, vocab)
    # 字符级 CTC：直接拼成一句中文，不再用空格分隔
    sentence = "".join(tokens) if tokens else "(空预测)"
    return tokens, sentence


def build_feature_extractor() -> ResNetFeatureExtractor:
    """
    构建与阶段一一致的 ResNet 特征提取器
    """
    extractor = ResNetFeatureExtractor(feat_cfg.RESNET_MODEL)
    return extractor


def predict_from_video(
    video_path: str,
    checkpoint_path: str = DEFAULT_CKPT,
) -> Tuple[List[str], str]:
    """对一段视频做手语识别，返回 (token 列表, 句子字符串)"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    model, vocab, device = load_checkpoint(checkpoint_path)
    extractor = build_feature_extractor()

    frames = load_video_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError(f"视频无法读取或为空：{video_path}")

    feats = extractor.extract(frames)  # (T, 512)
    return predict_from_features(feats, model, vocab, device)


def predict_from_image(
    image_path: str,
    checkpoint_path: str = DEFAULT_CKPT,
) -> Tuple[List[str], str]:
    """
    对单张图片做“类似视频”的推理（视为只有一帧的短视频）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在：{image_path}")

    model, vocab, device = load_checkpoint(checkpoint_path)
    extractor = build_feature_extractor()

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    feats = extractor.extract([img])  # (1, 512)
    return predict_from_features(feats, model, vocab, device)


def main():
    parser = argparse.ArgumentParser(
        description="CE-CSL 连续手语识别 Demo 推理脚本"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="输入视频路径（mp4 等）",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="输入图片路径（jpg/png 等）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT,
        help=f"模型权重文件路径（默认：{DEFAULT_CKPT}）",
    )

    args = parser.parse_args()

    if not args.video and not args.image:
        parser.error("必须提供 --video 或 --image 其中之一。")

    if args.video and args.image:
        parser.error("请只提供 --video 或 --image 之一，而不是同时提供。")

    if args.video:
        print(f"使用视频推理: {args.video}")
        tokens, sentence = predict_from_video(
            args.video,
            checkpoint_path=args.checkpoint,
        )
    else:
        print(f"使用图片推理: {args.image}")
        tokens, sentence = predict_from_image(
            args.image,
            checkpoint_path=args.checkpoint,
        )

    print("\n=== 推理结果 ===")
    print(f"字符序列: {tokens}")
    print(f"预测句子: {sentence}")


if __name__ == "__main__":
    main()

