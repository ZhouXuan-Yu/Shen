"""
CTC 手语识别服务封装

基于 AAA 目录下已经完成的训练与推理代码：
- 复用 `demo_infer.py` 中的 `predict_from_video` / `predict_from_image`
- 使用 `Project/Back/model/last_checkpoint.pt` 作为默认权重

提供给 FastAPI 的统一调用接口：
- recognize_video_file(path) -> 适配上传视频识别
- recognize_image_file(path) -> 适配上传图片识别
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple

import cv2

# ==================== 路径与依赖 ====================

# AAA 工程根目录（已包含 CTC 模型与推理脚本）
AAA_ROOT = r"D:\Aprogress\Shen\AAA"
if AAA_ROOT not in sys.path:
    sys.path.insert(0, AAA_ROOT)

print(f"[ctc_service] 初始化: AAA_ROOT={AAA_ROOT}")

try:
    # 复用 AAA/demo_infer 中已经验证的推理流程
    from demo_infer import predict_from_video, predict_from_image  # type: ignore
    print("[ctc_service] 已成功导入 demo_infer.predict_from_video / predict_from_image")
except Exception as e:
    print("[ctc_service] 导入 demo_infer 失败，请检查 AAA 目录与 Python 解释器:")
    traceback.print_exc()
    # 继续抛出，让启动阶段就暴露问题
    raise


@dataclass
class CTCConfig:
    """CTC 推理配置"""

    # 使用 Back 下的 checkpoint，方便部署时只依赖 Back 目录
    checkpoint_path: str = r"D:\Aprogress\Shen\Project\Back\model\last_checkpoint.pt"


cfg = CTCConfig()


def _get_video_meta(video_path: str) -> Tuple[float, int]:
    """获取视频时长（秒）与帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    if fps <= 0.0:
        return 0.0, frame_count

    duration = frame_count / fps
    return float(duration), frame_count


def recognize_video_file(video_path: str) -> Dict:
    """
    对上传视频做 CTC 推理，返回统一结构：
    {
      "text": str,
      "tokens": List[str],
      "confidence": float,
      "startTime": float,
      "endTime": float,
      "videoDuration": float,
      "processedFrames": int
    }
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    print(f"[ctc_service] recognize_video_file 被调用, video_path={video_path}")
    print(f"[ctc_service] 使用 checkpoint: {cfg.checkpoint_path}")

    tokens, sentence = predict_from_video(
        video_path,
        checkpoint_path=cfg.checkpoint_path,
    )

    duration, frame_count = _get_video_meta(video_path)

    # 当前 CTC 推理脚本未输出置信度，这里先给一个固定值占位，满足前端展示需求
    confidence = 95.0

    result = {
        "text": sentence,
        "tokens": tokens,
        "confidence": confidence,
        "startTime": 0.0,
        "endTime": duration,
        "videoDuration": duration,
        "processedFrames": frame_count,
    }
    print(
        "[ctc_service] recognize_video_file 完成: "
        f"text={sentence!r}, tokens_len={len(tokens)}, "
        f"duration={duration}, frames={frame_count}"
    )
    return result


def recognize_image_file(image_path: str) -> Dict:
    """
    对上传图片做 CTC 推理，返回统一结构：
    {
      "text": str,
      "tokens": List[str],
      "confidence": float
    }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    print(f"[ctc_service] recognize_image_file 被调用, image_path={image_path}")
    print(f"[ctc_service] 使用 checkpoint: {cfg.checkpoint_path}")

    tokens, sentence = predict_from_image(
        image_path,
        checkpoint_path=cfg.checkpoint_path,
    )

    confidence = 95.0

    result = {
        "text": sentence,
        "tokens": tokens,
        "confidence": confidence,
    }
    print(
        "[ctc_service] recognize_image_file 完成: "
        f"text={sentence!r}, tokens_len={len(tokens)}"
    )
    return result


