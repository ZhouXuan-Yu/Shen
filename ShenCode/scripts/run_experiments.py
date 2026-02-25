"""自动化实验流程脚本 - Python版本，更灵活可配置"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_CONFIGS = {
    "full": {
        "description": "完整实验矩阵（144次训练）：8模型 x 6K x 3seeds",
        "models": [
            "resnet18",
            "resnet34",
            "resnet50",
            "mobilenetv2",
            "mobilenetv3_small",
            "mobilenetv3_large",
            "shufflenetv2",
            "efficientnet_b0",
        ],
        "k_values": [1, 2, 5, 10, 20, 50],
        "seeds": [0, 1, 2],
    },
    "quick": {
        "description": "快速验证（8次训练）：8模型 x 1K x 1seed",
        "models": [
            "resnet18",
            "resnet34",
            "resnet50",
            "mobilenetv2",
            "mobilenetv3_small",
            "mobilenetv3_large",
            "shufflenetv2",
            "efficientnet_b0",
        ],
        "k_values": [5],
        "seeds": [0],
    },
    "single": {
        "description": "单次训练测试",
        "models": ["resnet18"],
        "k_values": [5],
        "seeds": [0],
    },
    "resnet_only": {
        "description": "仅 ResNet 系列（54次训练）",
        "models": ["resnet18", "resnet34", "resnet50"],
        "k_values": [1, 2, 5, 10, 20, 50],
        "seeds": [0, 1, 2],
    },
    "mobile_only": {
        "description": "仅 MobileNet 系列（54次训练）",
        "models": ["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"],
        "k_values": [1, 2, 5, 10, 20, 50],
        "seeds": [0, 1, 2],
    },
    "lightweight": {
        "description": "轻量模型对比（36次训练）",
        "models": ["mobilenetv2", "shufflenetv2", "efficientnet_b0"],
        "k_values": [1, 2, 5, 10, 20, 50],
        "seeds": [0, 1],
    },
    "core": {
        "description": "核心实验（72次训练）：4模型 x 6K x 3seeds",
        "models": ["resnet18", "resnet34", "mobilenetv2", "efficientnet_b0"],
        "k_values": [1, 2, 5, 10, 20, 50],
        "seeds": [0, 1, 2],
    },
}


def run_command(cmd: list[str], desc: str = "") -> bool:
    """运行命令并返回是否成功"""
    print(f"\n{'=' * 60}")
    print(f"执行: {desc or ' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_training(model: str, k: int, seed: int, epochs: int = 50) -> bool:
    """运行单次训练"""
    cmd = [
        sys.executable,
        "tools/train.py",
        "--model",
        model,
        "--k",
        str(k),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
    ]
    return run_command(cmd, f"训练 {model} K={k} seed={seed}")


def run_evaluation(checkpoint_path: Path, seed: int) -> bool:
    """运行评估"""
    cmd = [
        sys.executable,
        "tools/eval.py",
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "val",
        "--seed",
        str(seed),
    ]
    return run_command(cmd, f"评估 {checkpoint_path.parent.parent.name}")


def find_latest_checkpoint(model: str, k: int, seed: int) -> Path | None:
    """找到最新的 checkpoint"""
    runs_dir = PROJECT_ROOT / "runs"
    pattern = f"{model}_K{k}_seed{seed}_*"

    matching_dirs = sorted(runs_dir.glob(pattern), reverse=True)
    for run_dir in matching_dirs:
        best_pt = run_dir / "checkpoints" / "best.pt"
        if best_pt.exists():
            return best_pt
    return None


def list_experiments(config_name: str):
    """列出实验配置中的所有实验"""
    if config_name not in EXPERIMENT_CONFIGS:
        print(f"未知配置: {config_name}")
        return

    config = EXPERIMENT_CONFIGS[config_name]
    experiments = []
    for model in config["models"]:
        for k in config["k_values"]:
            for seed in config["seeds"]:
                experiments.append((model, k, seed))

    print(f"\n实验配置: {config_name}")
    print(f"描述: {config['description']}")
    print(f"总计: {len(experiments)} 个实验\n")
    print(f"{'编号':<6} {'模型':<20} {'K值':<6} {'Seed':<6}")
    print("-" * 40)
    for i, (model, k, seed) in enumerate(experiments, 1):
        print(f"{i:<6} {model:<20} {k:<6} {seed:<6}")


def run_experiment_matrix(
    config_name: str,
    epochs: int = 50,
    skip_existing: bool = True,
    start: int = 1,
    end: int = None,
):
    """运行实验矩阵

    Args:
        config_name: 实验配置名称
        epochs: 训练轮数
        skip_existing: 是否跳过已存在的实验
        start: 从第几个实验开始（1-indexed）
        end: 到第几个实验结束（包含，1-indexed）
    """
    if config_name not in EXPERIMENT_CONFIGS:
        print(f"未知配置: {config_name}")
        print(f"可用配置: {list(EXPERIMENT_CONFIGS.keys())}")
        return

    config = EXPERIMENT_CONFIGS[config_name]

    # 生成完整实验列表
    experiments = []
    for model in config["models"]:
        for k in config["k_values"]:
            for seed in config["seeds"]:
                experiments.append((model, k, seed))

    total = len(experiments)

    # 处理范围参数
    if end is None:
        end = total
    start = max(1, min(start, total))
    end = max(start, min(end, total))

    print(f"\n{'#' * 60}")
    print(f"# 实验配置: {config_name}")
    print(f"# {config['description']}")
    print(f"# 模型: {config['models']}")
    print(f"# K值: {config['k_values']}")
    print(f"# Seeds: {config['seeds']}")
    print(f"# 实验范围: {start} ~ {end} (共 {end - start + 1} 个，总计 {total} 个)")
    print(f"{'#' * 60}")

    for i in range(start - 1, end):
        model, k, seed = experiments[i]
        current = i + 1
        print(f"\n[{current}/{total}] {model} K={k} seed={seed}")

        existing_ckpt = find_latest_checkpoint(model, k, seed)
        if skip_existing and existing_ckpt:
            print("  已存在 checkpoint，跳过训练")
        else:
            run_training(model, k, seed, epochs)

        ckpt = find_latest_checkpoint(model, k, seed)
        if ckpt:
            run_evaluation(ckpt, seed)
        else:
            print("  警告: 未找到 checkpoint")


def main():
    parser = argparse.ArgumentParser(description="自动化实验流程")
    parser.add_argument(
        "--config",
        type=str,
        default="p0_quick",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="实验配置",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已存在的实验")
    parser.add_argument(
        "--start", type=int, default=1, help="从第几个实验开始（1-indexed）"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="到第几个实验结束（包含）"
    )
    parser.add_argument("--benchmark-only", action="store_true", help="仅运行资源评测")
    parser.add_argument("--summarize-only", action="store_true", help="仅运行结果汇总")
    parser.add_argument("--list", action="store_true", help="仅列出实验列表，不运行")

    args = parser.parse_args()

    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.list:
        list_experiments(args.config)
        return

    if args.benchmark_only:
        run_command([sys.executable, "tools/benchmark.py"], "资源评测")
        return

    if args.summarize_only:
        run_command([sys.executable, "tools/summarize_results.py"], "结果汇总")
        return

    run_experiment_matrix(
        args.config, args.epochs, not args.no_skip, args.start, args.end
    )

    run_command([sys.executable, "tools/benchmark.py"], "资源评测")

    run_command([sys.executable, "tools/summarize_results.py"], "结果汇总")

    print(f"\n{'=' * 60}")
    print(f"实验完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果文件: results/tables/table_main.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
