from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_manifest(manifest_path: Path) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_split_and_label(records: list[dict]) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        split = r.get("split", "unsplit")
        label = r["label"]
        idx = r.get("global_id", r.get("sample_id"))
        if isinstance(idx, str):
            idx = records.index(r)
        grouped[split][label].append(idx)

    for split in grouped:
        for label in grouped[split]:
            grouped[split][label].sort()

    return grouped


def sample_indices_per_class(
    by_label: dict[str, list[int]],
    n_per_class: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    picked: dict[str, list[int]] = {}
    for label, ids in sorted(by_label.items()):
        if len(ids) < n_per_class:
            raise RuntimeError(
                f"label '{label}' 样本不足：需要 {n_per_class}，但只有 {len(ids)}"
            )
        ids_copy = list(ids)
        rng.shuffle(ids_copy)
        picked[label] = sorted(ids_copy[:n_per_class])
    return picked


def flatten(picked: dict[str, list[int]]) -> list[int]:
    out: list[int] = []
    for _, ids in sorted(picked.items()):
        out.extend(ids)
    return sorted(out)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest.jsonl",
        help="manifest 路径（默认 data/processed/manifest.jsonl）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/splits",
        help="输出目录（默认 data/splits）",
    )
    parser.add_argument("--k", type=int, default=5, help="K-shot 的 K（默认 5）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=2,
        help="当官方无 val split 时，从 train 中每类抽取的 val 样本数 V（默认 2）",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="训练 split 名称（默认 train）",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="验证 split 名称（默认 val）",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="测试 split 名称（默认 test）",
    )
    parser.add_argument(
        "--force-make-val-from-train",
        action="store_true",
        help="即使存在官方 val split，也强制从 train 抽取 val（用于统一实验设置）",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (project_root / args.manifest).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    records = read_manifest(manifest_path)
    grouped = group_by_split_and_label(records)

    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split

    if train_split not in grouped:
        if "unsplit" in grouped:
            train_split = "unsplit"
        else:
            raise RuntimeError(
                f"manifest 中找不到 train split（{args.train_split}），也找不到 unsplit"
            )

    rng = random.Random(args.seed)

    has_official_val = val_split in grouped
    make_val_from_train = args.force_make_val_from_train or (not has_official_val)

    train_by_label = grouped[train_split]

    if make_val_from_train:
        val_picked = sample_indices_per_class(
            by_label=train_by_label, n_per_class=args.val_per_class, rng=rng
        )
        val_indices = flatten(val_picked)

        train_pool_by_label: dict[str, list[int]] = {}
        val_set = set(val_indices)
        for label, ids in train_by_label.items():
            remain = [i for i in ids if i not in val_set]
            train_pool_by_label[label] = remain
    else:
        val_by_label = grouped[val_split]
        val_indices = sorted([i for ids in val_by_label.values() for i in ids])
        train_pool_by_label = {
            label: list(ids) for label, ids in train_by_label.items()
        }

    fewshot_picked = sample_indices_per_class(
        by_label=train_pool_by_label, n_per_class=args.k, rng=rng
    )
    fewshot_indices = flatten(fewshot_picked)

    test_indices: list[int] = []
    if test_split in grouped:
        test_indices = sorted([i for ids in grouped[test_split].values() for i in ids])

    val_file = out_dir / f"val_seed{args.seed}.json"
    kshot_file = out_dir / f"kshot_K{args.k}_seed{args.seed}.json"

    write_json(
        val_file,
        {
            "version": 1,
            "seed": args.seed,
            "source_manifest": str(manifest_path.relative_to(project_root).as_posix()),
            "split": "val"
            if has_official_val and not make_val_from_train
            else "val_from_train",
            "val_per_class": args.val_per_class if make_val_from_train else None,
            "indices": val_indices,
        },
    )

    write_json(
        kshot_file,
        {
            "version": 1,
            "seed": args.seed,
            "k": args.k,
            "source_manifest": str(manifest_path.relative_to(project_root).as_posix()),
            "train_split": train_split,
            "val_file": str(val_file.relative_to(project_root).as_posix()),
            "train_fewshot_indices": fewshot_indices,
            "test_split": test_split if test_indices else None,
            "test_indices": test_indices,
        },
    )

    print("val indices written to:", str(val_file.relative_to(project_root)))
    print("k-shot indices written to:", str(kshot_file.relative_to(project_root)))
    print("k:", args.k)
    print("seed:", args.seed)
    print("val_count:", len(val_indices))
    print("train_fewshot_count:", len(fewshot_indices))
    if test_indices:
        print("test_count:", len(test_indices))


if __name__ == "__main__":
    main()
