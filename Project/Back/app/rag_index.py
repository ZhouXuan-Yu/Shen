"""
CE-CSL 文本 → 视频 检索索引加载与查询模块

职责：
- 从 data/cecsl_index 中加载预先构建好的句向量与元数据
- 提供基于余弦相似度的检索接口
- 提供随机样本接口
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class VideoRAGIndex:
    """简单的内存向量索引"""

    def __init__(
        self,
        index_dir: str = "data/cecsl_index",
        split: str = "train",
    ) -> None:
        self.index_dir = index_dir
        self.split = split

        emb_path = os.path.join(index_dir, f"embeddings_{split}.npy")
        meta_path = os.path.join(index_dir, f"metadata_{split}.json")

        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"未找到索引文件，请先运行 scripts/build_cecsl_index.py 构建索引。\n"
                f"缺失文件: {emb_path if not os.path.exists(emb_path) else ''} "
                f"{meta_path if not os.path.exists(meta_path) else ''}"
            )

        self.embeddings: np.ndarray = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.samples: List[Dict[str, Any]] = meta.get("samples", [])
        self.model_name: str = meta.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.data_root: str = meta.get("data_root", "")
        # CE-CSL 视频通过 Nginx 暴露的基础 URL，例如: https://your-domain/cecsl/video
        # 默认使用相对路径 /cecsl/video，前端和网关可按需配置
        self.video_base_url: str = os.getenv("CECSL_VIDEO_BASE_URL", "/cecsl/video")

        if self.embeddings.shape[0] != len(self.samples):
            raise ValueError(
                f"向量数量 ({self.embeddings.shape[0]}) 与样本数量 ({len(self.samples)}) 不一致"
            )

        # 预先归一化向量，便于使用点积计算余弦相似度
        # 如果构建时已经 normalize_embeddings=True，这里只是保证万无一失
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = self.embeddings / norms

        # 延迟加载模型（第一次检索时再加载）
        self._model: SentenceTransformer | None = None

        print(
            f"[VideoRAGIndex] 已加载索引: split={split}, "
            f"samples={len(self.samples)}, dim={self.embeddings.shape[1]}, "
            f"model={self.model_name}"
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[VideoRAGIndex] 懒加载句向量模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询句子为归一化向量"""
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        emb = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return emb

    def _build_video_url(self, rel_video_path: str) -> str:
        """根据相对路径构造可直接播放的视频 URL"""
        base = (self.video_base_url or "").rstrip("/")
        rel = (rel_video_path or "").lstrip("/")
        if not base:
            return rel
        return f"{base}/{rel}"

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """基于余弦相似度的简单向量检索"""
        query_vec = self.encode_query(query)

        # 由于向量已归一化，点积即为余弦相似度
        sims = np.dot(self.embeddings, query_vec)

        top_k = max(1, min(top_k, len(self.samples)))
        idxs = np.argpartition(-sims, top_k - 1)[:top_k]
        idxs = idxs[np.argsort(-sims[idxs])]

        results: List[Dict[str, Any]] = []
        for rank, i in enumerate(idxs, start=1):
            s = self.samples[i]
            sim = float(sims[i])

            # 还原为绝对视频路径（前端可选择使用绝对/相对）
            rel_video_path = s["video_path"]
            abs_video_path = (
                os.path.join(self.data_root, "video", rel_video_path)
                if self.data_root
                else rel_video_path
            )

            results.append(
                {
                    "id": s["sample_id"],
                    "sentence": s["sentence"],
                    "split": s["split"],
                    "videoPath": rel_video_path,
                    "videoAbsPath": abs_video_path.replace("\\", "/"),
                    "videoUrl": self._build_video_url(rel_video_path),
                    "similarity": sim,
                    "rank": rank,
                }
            )

        return results

    def get_random(self, limit: int = 10) -> List[Dict[str, Any]]:
        """返回随机样本，用于首页推荐"""
        n = len(self.samples)
        if n == 0:
            return []

        limit = max(1, min(limit, n))
        idxs = np.random.choice(n, size=limit, replace=False)

        results: List[Dict[str, Any]] = []
        for rank, i in enumerate(idxs, start=1):
            s = self.samples[i]
            rel_video_path = s["video_path"]
            abs_video_path = (
                os.path.join(self.data_root, "video", rel_video_path)
                if self.data_root
                else rel_video_path
            )

            results.append(
                {
                    "id": s["sample_id"],
                    "sentence": s["sentence"],
                    "split": s["split"],
                    "videoPath": rel_video_path,
                    "videoAbsPath": abs_video_path.replace("\\", "/"),
                    "videoUrl": self._build_video_url(rel_video_path),
                    "similarity": 1.0,  # 随机推荐可认为是“满分”，前端主要用作展示
                    "rank": rank,
                }
            )

        return results


@lru_cache(maxsize=1)
def get_video_rag_index() -> VideoRAGIndex:
    """全局单例，供 FastAPI 路由调用"""
    index_dir = os.getenv("CECSL_INDEX_DIR", "data/cecsl_index")
    split = os.getenv("CECSL_INDEX_SPLIT", "train")
    return VideoRAGIndex(index_dir=index_dir, split=split)

