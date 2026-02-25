from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _is_media_file(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in IMAGE_EXTS or ext in VIDEO_EXTS


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def _scan_split(raw_dir: Path, split: str) -> list[tuple[str, Path]]:
    split_dir = raw_dir / split
    if not split_dir.exists():
        return []

    pairs: list[tuple[str, Path]] = []
    for p in sorted(split_dir.rglob("*")):
        if not p.is_file():
            continue
        if not _is_media_file(p):
            continue
        rel = p.relative_to(split_dir)
        if len(rel.parts) < 2:
            continue
        label = rel.parts[0]
        pairs.append((label, p))
    return pairs


def _scan_unsplit(raw_dir: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        if not _is_media_file(p):
            continue
        rel = p.relative_to(raw_dir)
        if len(rel.parts) < 2:
            continue
        label = rel.parts[0]
        pairs.append((label, p))
    return pairs


def _parse_sample_id(filename: str) -> str | None:
    """Extract sample_id from filename like 'signer0_sample1_color.mp4' -> 'signer0_sample1'"""
    match = re.match(r"(signer\d+_sample\d+)", filename)
    return match.group(1) if match else None


def _load_label_file(label_file: Path) -> dict[str, int]:
    """Load AUTSL-style label CSV: sample_id,label -> {sample_id: label_id}"""
    labels = {}
    with label_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                sample_id = row[0].strip()
                label_id = int(row[1].strip())
                labels[sample_id] = label_id
    return labels


def _scan_autsl_split(
    split_dir: Path, label_map: dict[str, int], color_only: bool = True
) -> list[tuple[str, int, Path]]:
    """Scan AUTSL split directory and return (sample_id, label_id, path) tuples."""
    results: list[tuple[str, int, Path]] = []
    if not split_dir.exists():
        return results

    for p in sorted(split_dir.rglob("*")):
        if not p.is_file():
            continue
        if not _is_media_file(p):
            continue
        if color_only and "_depth" in p.name:
            continue

        sample_id = _parse_sample_id(p.name)
        if sample_id is None:
            continue
        if sample_id not in label_map:
            continue

        label_id = label_map[sample_id]
        results.append((sample_id, label_id, p))

    return results


def build_manifest_autsl(
    raw_dir: Path,
    project_root: Path,
    splits_config: dict[str, tuple[Path, Path]],
    color_only: bool = True,
) -> tuple[list[dict], dict[int, str]]:
    """Build manifest for AUTSL dataset using label files.

    Args:
        raw_dir: Root raw data directory
        project_root: Project root for relative paths
        splits_config: {split_name: (video_dir, label_file)} mapping
        color_only: If True, only include *_color.mp4 files

    Returns:
        (records, id2label) tuple
    """
    records: list[dict] = []
    all_label_ids: set[int] = set()

    for split, (video_dir, label_file) in splits_config.items():
        if not label_file.exists():
            print(
                f"Warning: label file not found: {label_file}, skipping split '{split}'"
            )
            continue
        if not video_dir.exists():
            print(
                f"Warning: video dir not found: {video_dir}, skipping split '{split}'"
            )
            continue

        label_map = _load_label_file(label_file)
        all_label_ids.update(label_map.values())

        for sample_id, label_id, path in _scan_autsl_split(
            video_dir, label_map, color_only
        ):
            records.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "label_id": label_id,
                    "label": str(label_id),
                    "path": str(path.relative_to(project_root).as_posix()),
                    "media_type": _media_type(path),
                    "ext": path.suffix.lower(),
                }
            )

    for i, r in enumerate(records):
        r["global_id"] = i

    id2label = {i: str(i) for i in sorted(all_label_ids)}
    return records, id2label


def build_manifest(raw_dir: Path, project_root: Path, splits: list[str]) -> list[dict]:
    split_dirs_exist = any((raw_dir / s).exists() for s in splits)

    records: list[dict] = []
    if split_dirs_exist:
        for split in splits:
            for label, path in _scan_split(raw_dir, split):
                records.append(
                    {
                        "split": split,
                        "label": label,
                        "path": str(path.relative_to(project_root).as_posix()),
                        "media_type": _media_type(path),
                        "ext": path.suffix.lower(),
                    }
                )
    else:
        for label, path in _scan_unsplit(raw_dir):
            records.append(
                {
                    "split": "unsplit",
                    "label": label,
                    "path": str(path.relative_to(project_root).as_posix()),
                    "media_type": _media_type(path),
                    "ext": path.suffix.lower(),
                }
            )

    labels = sorted({r["label"] for r in records})
    label2id = {label: i for i, label in enumerate(labels)}

    for i, r in enumerate(records):
        r["sample_id"] = i
        r["label_id"] = label2id[r["label"]]

    return records


def compute_stats(records: list[dict]) -> dict:
    by_split = defaultdict(int)
    by_label = defaultdict(int)
    by_split_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_media_type = defaultdict(int)

    for r in records:
        split = r["split"]
        label = r["label"]
        media_type = r.get("media_type", "unknown")
        by_split[split] += 1
        by_label[label] += 1
        by_split_label[split][label] += 1
        by_media_type[media_type] += 1

    return {
        "total": len(records),
        "splits": dict(sorted(by_split.items())),
        "media_types": dict(sorted(by_media_type.items())),
        "labels": dict(sorted(by_label.items(), key=lambda x: (-x[1], x[0]))),
        "by_split_label": {
            split: dict(sorted(labels.items(), key=lambda x: (-x[1], x[0])))
            for split, labels in sorted(by_split_label.items())
        },
    }


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({r["label"] for r in records})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (output_dir / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "id2label.json").write_text(
        json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = compute_stats(records)
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dataset: scan raw data and generate manifest/stats."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="输入数据根目录（默认 data/raw）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="输出目录（默认 data/processed）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="split 名称列表（逗号分隔），默认 train,val,test",
    )
    parser.add_argument(
        "--autsl",
        action="store_true",
        help="启用 AUTSL 数据集模式：从标签文件导入（train_labels.csv, val_labels.csv, test_labels.csv）",
    )
    parser.add_argument(
        "--color-only",
        action="store_true",
        default=True,
        help="仅包含 *_color.mp4 文件，排除深度视频（默认 True）",
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="同时包含 *_depth.mp4 文件",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    raw_dir_arg = Path(args.raw_dir)
    raw_dir = raw_dir_arg if raw_dir_arg.is_absolute() else (project_root / raw_dir_arg)
    raw_dir = raw_dir.resolve()

    output_dir_arg = Path(args.output_dir)
    output_dir = (
        output_dir_arg
        if output_dir_arg.is_absolute()
        else (project_root / output_dir_arg)
    )
    output_dir = output_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    color_only = args.color_only and not args.include_depth

    if args.autsl:
        splits_config: dict[str, tuple[Path, Path]] = {
            "train": (raw_dir / "train" / "train", raw_dir / "train_labels.csv"),
            "val": (raw_dir / "val" / "val", raw_dir / "val_labels.csv"),
            "test": (raw_dir / "test" / "test", raw_dir / "test_labels.csv"),
        }
        records, id2label = build_manifest_autsl(
            raw_dir=raw_dir,
            project_root=project_root,
            splits_config=splits_config,
            color_only=color_only,
        )
        if not records:
            raise RuntimeError(
                "未扫描到任何 AUTSL 样本。请检查：\n"
                "1. 视频目录：data/raw/train/train/, data/raw/val/val/, data/raw/test/test/\n"
                "2. 标签文件：data/raw/train_labels.csv, data/raw/val_labels.csv, data/raw/test_labels.csv"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        label2id = {v: k for k, v in id2label.items()}

        manifest_path = output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        (output_dir / "label2id.json").write_text(
            json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "id2label.json").write_text(
            json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        stats = compute_stats(records)
        (output_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        records = build_manifest(
            raw_dir=raw_dir, project_root=project_root, splits=splits
        )
        if not records:
            raise RuntimeError(
                "未扫描到任何媒体文件。请检查 data/raw 目录结构是否满足："
                "data/raw/<split>/<label>/* 或 data/raw/<label>/*，并确保文件后缀在支持列表中。"
            )
        write_outputs(records=records, output_dir=output_dir)
        stats = compute_stats(records)

    print(
        "manifest written to:",
        str((output_dir / "manifest.jsonl").relative_to(project_root)),
    )
    print("total:", stats["total"])
    print("splits:", stats["splits"])
    print("media_types:", stats["media_types"])
    print("num_classes:", len(stats["labels"]))


if __name__ == "__main__":
    main()
