"""
构建 CE-CSL 数据集的句向量索引

功能：
- 从 CE-CSL 的 label CSV（train/dev/test）中读取句子与视频路径
- 使用中文句向量模型编码每条中文句子
- 将所有向量保存为 embeddings.npy
- 将 metadata（video_id / video_path / sentence / split 等）保存为 metadata.json

依赖：
- pandas
- numpy
- sentence-transformers

使用示例（在 Project/Back 下）：
    python -m scripts.build_cecsl_index ^
        --data-root "D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL" ^
        --split train ^
        --model BAAI/bge-small-zh-v1.5
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class SampleMeta:
    """单条样本的元数据"""

    sample_id: str
    split: str
    video_path: str
    sentence: str
    gloss: str = ""
    note: str = ""


def clean_sentence(text: str) -> str:
    """简单清洗中文句子"""
    if not isinstance(text, str):
        text = str(text or "")
    text = text.strip()
    # 这里可以根据需要做更多清洗，如全角转半角、统一引号等
    return text


def build_index(
    data_root: str,
    split: str,
    model_name: str,
    output_dir: str,
    batch_size: int = 64,
) -> None:
    """
    构建指定 split 的句向量索引

    data_root 例如: D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL
    默认假定：
      - 标签文件在 {data_root}/label/{split}.csv
      - 视频文件在 {data_root}/video/{split}/{Translator}/{Number}.mp4
    """
    label_path = os.path.join(data_root, "label", f"{split}.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    print(f"[build_cecsl_index] 读取标签文件: {label_path}")
    df = pd.read_csv(label_path)

    # 兼容不同版本的列名：有的使用 "Chinese"，有的使用 "Chinese Sentences"
    required_cols = {"Number", "Translator"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV 缺少必要列 {required_cols}，实际列为: {list(df.columns)}"
        )

    chinese_col_candidates = ["Chinese", "Chinese Sentences"]
    chinese_col = None
    for col in chinese_col_candidates:
        if col in df.columns:
            chinese_col = col
            break
    if chinese_col is None:
        raise ValueError(
            f"CSV 中未找到中文句子列，期望列名之一: {chinese_col_candidates}，实际列为: {list(df.columns)}"
        )

    # 兼容可选列：Gloss / Note
    gloss_col = "Gloss" if "Gloss" in df.columns else None
    note_col = "Note" if "Note" in df.columns else None

    # 准备句子与元数据
    sentences: List[str] = []
    metas: List[SampleMeta] = []

    for idx, row in df.iterrows():
        number = str(row["Number"]).strip()
        translator = str(row["Translator"]).strip()
        chinese = clean_sentence(row[chinese_col])
        gloss = str(row[gloss_col]).strip() if gloss_col else ""
        note = str(row[note_col]).strip() if note_col else ""

        if not chinese:
            continue

        # 构造视频路径
        # 目录结构：
        #   - 标签文件: {data_root}/label/{split}.csv （例如 dev.csv）
        #   - 视频文件: {data_root}/video/{split}/{Translator} （例如 video/dev/D/dev-00001.mp4）
        # 其中 Translator 列为 A-L，表示子目录名称
        video_rel = os.path.join(split, translator, f"{number}.mp4")
        video_path = os.path.join(data_root, "video", video_rel)

        if not os.path.exists(video_path):
            # 如果路径不存在，仅做警告，不中断（有些样本可能缺失视频）
            print(
                f"[build_cecsl_index] 警告: 视频不存在，sample_id={split}-{number}, path={video_path}"
            )

        # 使用 Number 作为 sample_id，便于与原始数据对应
        sample_id = number
        metas.append(
            SampleMeta(
                sample_id=sample_id,
                split=split,
                video_path=video_rel.replace("\\", "/"),
                sentence=chinese,
                gloss=gloss,
                note=note,
            )
        )
        sentences.append(chinese)

    if not sentences:
        raise RuntimeError("没有有效句子可用于构建索引")

    print(f"[build_cecsl_index] 共收集样本: {len(sentences)}")

    # 加载句向量模型
    print(f"[build_cecsl_index] 加载句向量模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 编码
    print(f"[build_cecsl_index] 开始编码句子，batch_size={batch_size} ...")
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"[build_cecsl_index] 向量形状: {embeddings.shape}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, f"embeddings_{split}.npy")
    meta_path = os.path.join(output_dir, f"metadata_{split}.json")

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "model_name": model_name,
                "data_root": data_root.replace("\\", "/"),
                "samples": [asdict(m) for m in metas],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[build_cecsl_index] 已保存向量到: {emb_path}")
    print(f"[build_cecsl_index] 已保存元数据到: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CE-CSL 句向量索引")
    parser.add_argument(
        "--data-root",
        type=str,
        default="D:/Aprogress/Shen/dataset/CE-CSL/CE-CSL",
        help="CE-CSL 数据根目录（包含 label/ 和 video/ 子目录）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="使用的数据划分（train/dev/test）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="SentenceTransformer 中文句向量模型名称",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cecsl_index",
        help="索引输出目录（相对当前工作目录）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="编码时的 batch size",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        data_root=args.data_root,
        split=args.split,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

