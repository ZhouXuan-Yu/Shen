"""Training script for few-shot sign language recognition."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import create_dataloader, get_train_transforms, get_eval_transforms
from src.models import create_model


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split_indices(split_file: Path) -> list[int]:
    """Load indices from a split file."""
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "train_fewshot_indices" in data:
        return data["train_fewshot_indices"]
    elif "indices" in data:
        return data["indices"]
    else:
        raise ValueError(f"Cannot find indices in {split_file}")


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    train_cfg = config["training"]

    if train_cfg["optimizer"] == "adam":
        return Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "sgd":
        return SGD(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=0.9,
            weight_decay=train_cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler from config."""
    train_cfg = config["training"]
    scheduler_type = train_cfg.get("scheduler", "none")

    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    writer: SummaryWriter | None = None,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)

    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train few-shot sign language recognition model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/default.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (overrides config)"
    )
    parser.add_argument("--k", type=int, default=None, help="K-shot (overrides config)")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config
    config = load_config(config_path)

    if args.model:
        config["model"]["name"] = args.model
    if args.k:
        config["fewshot"]["k"] = args.k
    if args.seed is not None:
        config["fewshot"]["seed"] = args.seed
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    k = config["fewshot"]["k"]
    seed = config["fewshot"]["seed"]
    model_name = config["model"]["name"]

    set_seed(config.get("seed", 42), config.get("deterministic", True))

    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_K{k}_seed{seed}_{timestamp}"

    run_dir = project_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = project_root / config["data"]["manifest"]

    kshot_file = project_root / f"data/splits/kshot_K{k}_seed{seed}.json"
    val_file = project_root / f"data/splits/val_seed{seed}.json"

    if not kshot_file.exists():
        raise FileNotFoundError(f"K-shot split file not found: {kshot_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Val split file not found: {val_file}")

    train_indices = load_split_indices(kshot_file)
    val_indices = load_split_indices(val_file)

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    input_size = config["data"].get("input_size", 224)
    train_transform = get_train_transforms(input_size=input_size)
    eval_transform = get_eval_transforms(input_size=input_size)

    train_loader = create_dataloader(
        manifest_path=manifest_path,
        indices=train_indices,
        batch_size=config["training"]["batch_size"],
        num_frames=config["data"]["num_frames"],
        frame_sampling=config["data"]["frame_sampling"],
        transform=train_transform,
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        project_root=project_root,
    )

    val_loader = create_dataloader(
        manifest_path=manifest_path,
        indices=val_indices,
        batch_size=config["training"]["batch_size"],
        num_frames=config["data"]["num_frames"],
        frame_sampling="uniform",
        transform=eval_transform,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        project_root=project_root,
    )

    label2id_path = project_root / "data/processed/label2id.json"
    with label2id_path.open("r", encoding="utf-8") as f:
        label2id = json.load(f)
    num_classes = len(label2id)
    print(f"Number of classes: {num_classes}")

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        aggregation=config["model"]["aggregation"],
        for_video=True,
    )
    model = model.to(device)

    label_smoothing = config["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    writer = None
    if config["logging"].get("tensorboard", True):
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc: {best_val_acc:.2f}%")

    epochs = config["training"]["epochs"]
    log_interval = config["logging"].get("log_interval", 10)
    save_interval = config["logging"].get("save_interval", 5)

    print(f"\nStarting training: {model_name}, K={k}, seed={seed}")
    print(
        f"Epochs: {epochs}, Batch size: {config['training']['batch_size']}, LR: {config['training']['learning_rate']}"
    )
    print("-" * 60)

    # 用于保存训练曲线数据（供绘图使用）
    metrics_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            log_interval,
            writer,
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, f"Epoch {epoch} [Val]"
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        # 记录训练曲线数据
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history["train_loss"].append(train_loss)
        metrics_history["train_acc"].append(train_acc)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_acc"].append(val_acc)
        metrics_history["learning_rate"].append(current_lr)

        if writer is not None:
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/Accuracy", train_acc, epoch)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Accuracy", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "config": config,
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_data["best_val_acc"] = best_val_acc
            torch.save(checkpoint_data, checkpoint_dir / "best.pt")
            print(f"  -> New best model saved (val_acc: {val_acc:.2f}%)")

        if (epoch + 1) % save_interval == 0:
            torch.save(checkpoint_data, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        torch.save(checkpoint_data, checkpoint_dir / "last.pt")

    if writer is not None:
        writer.close()

    # 保存训练曲线数据（供绘图使用）
    metrics_file = run_dir / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Training metrics saved to: {metrics_file}")

    print("-" * 60)
    print(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    results = {
        "model": model_name,
        "k": k,
        "seed": seed,
        "epochs": epochs,
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
    }

    with (run_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
