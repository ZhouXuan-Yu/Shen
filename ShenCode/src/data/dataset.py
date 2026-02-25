"""Dataset class for sign language video recognition."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from decord import VideoReader, cpu, gpu

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class SignLanguageDataset(Dataset):
    """Dataset for sign language video recognition.

    Supports loading videos and extracting frames using decord (preferred) or OpenCV.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        indices: list[int] | None = None,
        num_frames: int = 8,
        frame_sampling: str = "uniform",
        transform: Callable | None = None,
        project_root: Path | str | None = None,
        use_gpu_decode: bool = False,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to manifest.jsonl file.
            indices: List of global_id indices to include. If None, use all.
            num_frames: Number of frames to sample from each video.
            frame_sampling: Frame sampling strategy ('uniform', 'random', 'center').
            transform: Optional transform to apply to each frame.
            project_root: Project root directory for resolving relative paths.
            use_gpu_decode: Whether to use GPU for video decoding (decord only).
        """
        self.manifest_path = Path(manifest_path)
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.transform = transform
        self.use_gpu_decode = use_gpu_decode

        if project_root is None:
            self.project_root = self.manifest_path.parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.records = self._load_manifest(indices)

        if not DECORD_AVAILABLE and not CV2_AVAILABLE:
            raise RuntimeError(
                "Neither decord nor opencv-python is available for video loading."
            )

    def _load_manifest(self, indices: list[int] | None) -> list[dict]:
        """Load manifest and filter by indices."""
        all_records = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

        if indices is None:
            return all_records

        index_set = set(indices)
        return [r for r in all_records if r.get("global_id") in index_set]

    def __len__(self) -> int:
        return len(self.records)

    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        """Sample frame indices from video."""
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            return indices

        if self.frame_sampling == "uniform":
            step = total_frames / self.num_frames
            indices = [int(i * step) for i in range(self.num_frames)]
        elif self.frame_sampling == "random":
            indices = sorted(random.sample(range(total_frames), self.num_frames))
        elif self.frame_sampling == "center":
            start = (total_frames - self.num_frames) // 2
            indices = list(range(start, start + self.num_frames))
        else:
            raise ValueError(f"Unknown frame sampling strategy: {self.frame_sampling}")

        return indices

    def _load_video_decord(self, video_path: Path) -> np.ndarray:
        """Load video frames using decord."""
        ctx = gpu(0) if self.use_gpu_decode else cpu(0)
        vr = VideoReader(str(video_path), ctx=ctx)
        total_frames = len(vr)

        indices = self._sample_frame_indices(total_frames)
        frames = vr.get_batch(indices).asnumpy()
        return frames

    def _load_video_cv2(self, video_path: Path) -> np.ndarray:
        """Load video frames using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Cannot read video: {video_path}")

        indices = self._sample_frame_indices(total_frames)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()
        return np.stack(frames, axis=0)

    def _load_video(self, video_path: Path) -> np.ndarray:
        """Load video frames."""
        if DECORD_AVAILABLE:
            return self._load_video_decord(video_path)
        elif CV2_AVAILABLE:
            return self._load_video_cv2(video_path)
        else:
            raise RuntimeError("No video loading backend available.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a video sample.

        Returns:
            frames: Tensor of shape (num_frames, C, H, W) or (C, num_frames, H, W).
            label_id: Integer label.
        """
        record = self.records[idx]
        video_path = self.project_root / record["path"]
        label_id = record["label_id"]

        frames = self._load_video(video_path)

        if self.transform is not None:
            transformed_frames = []
            for i in range(frames.shape[0]):
                frame = self.transform(frames[i])
                transformed_frames.append(frame)
            frames = torch.stack(transformed_frames, dim=0)
        else:
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        return frames, label_id


def create_dataloader(
    manifest_path: Path | str,
    indices: list[int] | None = None,
    batch_size: int = 8,
    num_frames: int = 8,
    frame_sampling: str = "uniform",
    transform: Callable | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    project_root: Path | str | None = None,
) -> DataLoader:
    """Create a DataLoader for sign language videos.

    Args:
        manifest_path: Path to manifest.jsonl.
        indices: List of global_id indices to include.
        batch_size: Batch size.
        num_frames: Number of frames per video.
        frame_sampling: Frame sampling strategy.
        transform: Transform to apply.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory.
        project_root: Project root for resolving paths.

    Returns:
        DataLoader instance.
    """
    dataset = SignLanguageDataset(
        manifest_path=manifest_path,
        indices=indices,
        num_frames=num_frames,
        frame_sampling=frame_sampling,
        transform=transform,
        project_root=project_root,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
