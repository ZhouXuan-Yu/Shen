"""Evaluation script for sign language recognition models."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import create_dataloader, get_eval_transforms
from src.models import create_model


def load_checkpoint(
    checkpoint_path: Path, model: nn.Module, device: torch.device
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def compute_topk_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: tuple = (1, 5)
) -> list[float]:
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size).item())

    return results


@torch.no_grad()
def evaluate_protocol_a(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate using Protocol A (fixed split, full test).

    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)

        all_outputs.append(outputs.cpu())
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_targets.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    top1, top5 = compute_topk_accuracy(all_outputs, all_targets, topk=(1, 5))

    correct = all_preds.eq(all_targets)

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for i in range(len(all_targets)):
        label = all_targets[i].item()
        class_total[label] += 1
        if correct[i].item():
            class_correct[label] += 1

    class_acc = {}
    for label in range(num_classes):
        if class_total[label] > 0:
            class_acc[label] = 100.0 * class_correct[label] / class_total[label]
        else:
            class_acc[label] = 0.0

    class_acc_values = [class_acc[i] for i in range(num_classes) if class_total[i] > 0]
    macro_acc = np.mean(class_acc_values)

    sorted_acc = sorted(class_acc_values)
    worst_5 = sorted_acc[:5]
    best_5 = sorted_acc[-5:]

    percentiles = {
        "p25": np.percentile(class_acc_values, 25),
        "p50": np.percentile(class_acc_values, 50),
        "p75": np.percentile(class_acc_values, 75),
    }

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for pred, target in zip(all_preds, all_targets):
        confusion_matrix[target.item(), pred.item()] += 1

    metrics = {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_accuracy": macro_acc,
        "total_samples": len(all_targets),
        "correct_samples": correct.sum().item(),
        "class_wise": {
            "per_class_accuracy": class_acc,
            "worst_5_classes_acc": worst_5,
            "best_5_classes_acc": best_5,
            "percentiles": percentiles,
        },
        "confusion_matrix": confusion_matrix.tolist(),
    }

    return metrics


def load_split_indices(split_file: Path) -> list[int]:
    """Load indices from a split file."""
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "val_indices" in data:
        return data["val_indices"]
    elif "indices" in data:
        return data["indices"]
    else:
        raise ValueError(f"Cannot find indices in {split_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sign language recognition model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Evaluation protocol",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for val split file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for metrics JSON"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model_name = config.get("model", {}).get("name", "resnet18")
    num_frames = config.get("data", {}).get("num_frames", 8)
    input_size = config.get("data", {}).get("input_size", 224)

    label2id_path = project_root / "data/processed/label2id.json"
    with label2id_path.open("r", encoding="utf-8") as f:
        label2id = json.load(f)
    num_classes = len(label2id)

    print(f"Model: {model_name}, Classes: {num_classes}")

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        aggregation=config.get("model", {}).get("aggregation", "mean"),
        for_video=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    manifest_path = project_root / "data/processed/manifest.jsonl"

    if args.split == "val":
        split_file = project_root / f"data/splits/val_seed{args.seed}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Val split file not found: {split_file}")
        indices = load_split_indices(split_file)
    else:
        records = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("split") == "test":
                        records.append(r)
        indices = [r["global_id"] for r in records]

    print(f"Evaluating on {args.split} split: {len(indices)} samples")

    eval_transform = get_eval_transforms(input_size=input_size)

    dataloader = create_dataloader(
        manifest_path=manifest_path,
        indices=indices,
        batch_size=args.batch_size,
        num_frames=num_frames,
        frame_sampling="uniform",
        transform=eval_transform,
        shuffle=False,
        num_workers=args.num_workers,
        project_root=project_root,
    )

    if args.protocol == "a":
        print("\nRunning Protocol A evaluation...")
        metrics = evaluate_protocol_a(model, dataloader, device, num_classes)
    else:
        print("\nProtocol B not yet implemented, running Protocol A...")
        metrics = evaluate_protocol_a(model, dataloader, device, num_classes)

    metrics["evaluation_config"] = {
        "checkpoint": str(checkpoint_path),
        "protocol": args.protocol,
        "split": args.split,
        "seed": args.seed,
        "model": model_name,
        "num_classes": num_classes,
        "num_samples": len(indices),
    }

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"Macro Accuracy: {metrics['macro_accuracy']:.2f}%")
    print(f"Total Samples:  {metrics['total_samples']}")
    print(f"Correct:        {metrics['correct_samples']}")
    print("-" * 60)
    print("Class-wise Statistics:")
    print(
        f"  Worst 5 classes: {[f'{x:.1f}%' for x in metrics['class_wise']['worst_5_classes_acc']]}"
    )
    print(
        f"  Best 5 classes:  {[f'{x:.1f}%' for x in metrics['class_wise']['best_5_classes_acc']]}"
    )
    print(
        f"  P25/P50/P75:     {metrics['class_wise']['percentiles']['p25']:.1f}% / "
        f"{metrics['class_wise']['percentiles']['p50']:.1f}% / "
        f"{metrics['class_wise']['percentiles']['p75']:.1f}%"
    )
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
    else:
        run_dir = checkpoint_path.parent.parent
        output_path = run_dir / f"metrics_{args.split}_{args.protocol}.json"

    metrics_to_save = {k: v for k, v in metrics.items() if k != "confusion_matrix"}

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved to: {output_path}")

    cm_path = output_path.with_name(f"confusion_matrix_{args.split}.json")
    with cm_path.open("w", encoding="utf-8") as f:
        json.dump({"confusion_matrix": metrics["confusion_matrix"]}, f)
    print(f"Confusion matrix saved to: {cm_path}")

    # 自动生成混淆矩阵图像（用于论文）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(metrics["confusion_matrix"])
        n_classes = cm.shape[0]

        # 如果类别太多，只显示错误率最高的前20个类别
        if n_classes > 20:
            per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
            worst_idx = np.argsort(per_class_acc)[:20]
            cm_subset = cm[np.ix_(worst_idx, worst_idx)]
            title = f"Confusion Matrix (Top 20 Worst Classes) - {model_name}"
        else:
            cm_subset = cm
            title = f"Confusion Matrix - {model_name}"

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_subset, cmap="Blues", fmt="d", annot=(cm_subset.shape[0] <= 30))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()

        fig_path = output_path.with_name(f"confusion_matrix_{args.split}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix figure saved to: {fig_path}")
    except ImportError:
        print("Note: matplotlib/seaborn not available, skipping figure generation")


if __name__ == "__main__":
    main()
