"""生成论文用图表的脚本"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 设置中文字体和风格
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(run_dir: Path, save_path: Path = None):
    """
    绘制单次训练的学习曲线

    生成内容：
    - 训练/验证损失曲线
    - 训练/验证准确率曲线
    - 学习率变化曲线
    """
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"未找到 metrics.json: {run_dir}")
        return

    with open(metrics_file) as f:
        metrics = json.load(f)

    epochs = list(range(len(metrics.get("train_loss", []))))
    if not epochs:
        print(f"无训练数据: {run_dir}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, metrics["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, metrics["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2 = axes[1]
    ax2.plot(epochs, metrics["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(epochs, metrics["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 学习率曲线
    ax3 = axes[2]
    if "learning_rate" in metrics:
        ax3.plot(epochs, metrics["learning_rate"], linewidth=2, color="green")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"training_curve_{run_dir.name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存训练曲线: {save_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list = None,
    save_path: Path = None,
    title: str = "Confusion Matrix",
    top_n: int = 20,
):
    """
    绘制混淆矩阵热力图

    参数:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
        title: 图表标题
        top_n: 仅显示前N个类别（避免图表过大）
    """
    n_classes = confusion_matrix.shape[0]

    if n_classes > top_n:
        # 选择错误率最高的类别
        per_class_acc = np.diag(confusion_matrix) / (
            confusion_matrix.sum(axis=1) + 1e-10
        )
        worst_classes = np.argsort(per_class_acc)[:top_n]
        confusion_matrix = confusion_matrix[np.ix_(worst_classes, worst_classes)]
        if class_names:
            class_names = [class_names[i] for i in worst_classes]
        title += f" (Top {top_n} Worst Classes)"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion_matrix,
        annot=n_classes <= 30,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存混淆矩阵: {save_path}")
    plt.close()


def plot_model_comparison(results_csv: Path, save_dir: Path = None):
    """
    绘制模型对比图表

    生成内容：
    1. 不同K值下各模型准确率对比（柱状图）
    2. 准确率 vs 参数量 散点图
    3. 准确率 vs 推理延迟 散点图
    4. 模型性能雷达图
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # 1. 不同K值下各模型准确率对比
    fig, ax = plt.subplots(figsize=(12, 6))
    models = df["model"].unique()
    k_values = sorted(df["k"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(k_values)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))

    for i, k in enumerate(k_values):
        k_data = df[df["k"] == k].groupby("model")["val_acc"].mean()
        k_std = df[df["k"] == k].groupby("model")["val_acc"].std()
        bars = ax.bar(
            x + i * width - width * (len(k_values) - 1) / 2,
            [k_data.get(m, 0) for m in models],
            width,
            label=f"K={k}",
            color=colors[i],
            yerr=[k_std.get(m, 0) for m in models],
            capsize=3,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Model Performance Comparison Across Different K Values")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="K-shot")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison_by_k.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存模型对比图: {save_dir / 'model_comparison_by_k.png'}")

    # 2. 准确率 vs 参数量 散点图
    if "params_m" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for k in k_values:
            k_df = (
                df[df["k"] == k]
                .groupby("model")
                .agg({"val_acc": "mean", "params_m": "first"})
                .reset_index()
            )
            ax.scatter(
                k_df["params_m"], k_df["val_acc"], label=f"K={k}", s=100, alpha=0.7
            )
            for _, row in k_df.iterrows():
                ax.annotate(
                    row["model"],
                    (row["params_m"], row["val_acc"]),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Accuracy vs Model Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "accuracy_vs_params.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"保存准确率-参数量图: {save_dir / 'accuracy_vs_params.png'}")

    # 3. 准确率 vs 推理延迟 散点图
    if "latency_ms" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for k in k_values:
            k_df = (
                df[df["k"] == k]
                .groupby("model")
                .agg({"val_acc": "mean", "latency_ms": "first"})
                .reset_index()
            )
            ax.scatter(
                k_df["latency_ms"], k_df["val_acc"], label=f"K={k}", s=100, alpha=0.7
            )
            for _, row in k_df.iterrows():
                ax.annotate(
                    row["model"],
                    (row["latency_ms"], row["val_acc"]),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("Inference Latency (ms)")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Accuracy vs Inference Speed")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "accuracy_vs_latency.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"保存准确率-延迟图: {save_dir / 'accuracy_vs_latency.png'}")


def plot_k_sensitivity(results_csv: Path, save_dir: Path = None):
    """
    绘制 K 值敏感性分析图

    展示不同模型对 K 值变化的敏感程度
    """
    if save_dir is None:
        save_dir = FIGURES_DIR

    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(10, 6))

    models = df["model"].unique()
    k_values = sorted(df["k"].unique())

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_df = (
            df[df["model"] == model]
            .groupby("k")
            .agg({"val_acc": ["mean", "std"]})
            .reset_index()
        )
        model_df.columns = ["k", "mean", "std"]

        ax.errorbar(
            model_df["k"],
            model_df["mean"],
            yerr=model_df["std"],
            marker=markers[i % len(markers)],
            label=model,
            linewidth=2,
            markersize=8,
            capsize=4,
            color=colors[i],
        )

    ax.set_xlabel("K (samples per class)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("K-shot Sensitivity Analysis")
    ax.set_xticks(k_values)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "k_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存K值敏感性图: {save_dir / 'k_sensitivity.png'}")


def plot_per_class_accuracy(eval_result: dict, save_path: Path = None, top_n: int = 30):
    """
    绘制每类准确率柱状图

    参数:
        eval_result: 评估结果字典（包含 per_class_accuracy）
        save_path: 保存路径
        top_n: 显示准确率最低的前N个类别
    """
    if "per_class_accuracy" not in eval_result:
        print("评估结果中无 per_class_accuracy")
        return

    per_class = eval_result["per_class_accuracy"]
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])

    # 取最差的 top_n 个类别
    worst_classes = sorted_classes[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    classes = [c[0] for c in worst_classes]
    accs = [c[1] for c in worst_classes]

    colors = plt.cm.RdYlGn(np.array(accs) / 100)

    bars = ax.barh(classes, accs, color=colors)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Class")
    ax.set_title(f"Per-Class Accuracy (Bottom {top_n} Classes)")
    ax.set_xlim(0, 100)

    # 添加数值标签
    for bar, acc in zip(bars, accs):
        ax.text(
            acc + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存每类准确率图: {save_path}")
    plt.close()


def plot_resource_comparison(benchmark_json: Path, save_dir: Path = None):
    """
    绘制资源消耗对比图

    生成内容：
    1. 参数量对比柱状图
    2. 推理速度对比柱状图
    3. 显存消耗对比柱状图
    4. 综合雷达图
    """
    if save_dir is None:
        save_dir = FIGURES_DIR

    with open(benchmark_json) as f:
        data = json.load(f)

    models = list(data.keys())
    params = [data[m].get("params_m", 0) for m in models]
    latency = [data[m].get("latency_ms", 0) for m in models]
    throughput = [data[m].get("throughput", 0) for m in models]
    vram = [data[m].get("peak_vram_mb", 0) for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 参数量
    ax1 = axes[0, 0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(models)))
    bars = ax1.bar(models, params, color=colors)
    ax1.set_ylabel("Parameters (M)")
    ax1.set_title("Model Parameters")
    ax1.set_xticklabels(models, rotation=45, ha="right")
    for bar, p in zip(bars, params):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p:.2f}M",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 推理延迟
    ax2 = axes[0, 1]
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(models)))
    bars = ax2.bar(models, latency, color=colors)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Inference Latency")
    ax2.set_xticklabels(models, rotation=45, ha="right")
    for bar, l in zip(bars, latency):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{l:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 吞吐量
    ax3 = axes[1, 0]
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(models)))
    bars = ax3.bar(models, throughput, color=colors)
    ax3.set_ylabel("Throughput (samples/s)")
    ax3.set_title("Inference Throughput")
    ax3.set_xticklabels(models, rotation=45, ha="right")
    for bar, t in zip(bars, throughput):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}/s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 显存
    ax4 = axes[1, 1]
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(models)))
    bars = ax4.bar(models, vram, color=colors)
    ax4.set_ylabel("Peak VRAM (MB)")
    ax4.set_title("Peak VRAM Usage")
    ax4.set_xticklabels(models, rotation=45, ha="right")
    for bar, v in zip(bars, vram):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.0f}MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_dir / "resource_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存资源对比图: {save_dir / 'resource_comparison.png'}")


def plot_all_training_curves(runs_dir: Path = None, save_dir: Path = None):
    """
    批量绘制所有训练曲线
    """
    if runs_dir is None:
        runs_dir = PROJECT_ROOT / "runs"
    if save_dir is None:
        save_dir = FIGURES_DIR / "training_curves"
    save_dir.mkdir(parents=True, exist_ok=True)

    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            plot_training_curves(run_path, save_dir / f"{run_path.name}.png")


def main():
    parser = argparse.ArgumentParser(description="生成论文用图表")
    parser.add_argument(
        "--action",
        type=str,
        choices=["all", "training", "comparison", "sensitivity", "resource"],
        default="all",
        help="生成图表类型",
    )
    parser.add_argument("--results-csv", type=str, help="结果 CSV 文件路径")
    parser.add_argument("--benchmark-json", type=str, help="资源评测 JSON 文件路径")
    parser.add_argument("--run-dir", type=str, help="单个训练目录")

    args = parser.parse_args()

    results_csv = (
        Path(args.results_csv)
        if args.results_csv
        else PROJECT_ROOT / "results" / "tables" / "table_main.csv"
    )
    benchmark_json = (
        Path(args.benchmark_json)
        if args.benchmark_json
        else PROJECT_ROOT / "results" / "benchmark.json"
    )

    if args.action in ["all", "training"]:
        print("\n=== 绘制训练曲线 ===")
        if args.run_dir:
            plot_training_curves(Path(args.run_dir))
        else:
            plot_all_training_curves()

    if args.action in ["all", "comparison"]:
        print("\n=== 绘制模型对比图 ===")
        if results_csv.exists():
            plot_model_comparison(results_csv)
        else:
            print(f"未找到结果文件: {results_csv}")

    if args.action in ["all", "sensitivity"]:
        print("\n=== 绘制 K 值敏感性图 ===")
        if results_csv.exists():
            plot_k_sensitivity(results_csv)
        else:
            print(f"未找到结果文件: {results_csv}")

    if args.action in ["all", "resource"]:
        print("\n=== 绘制资源对比图 ===")
        if benchmark_json.exists():
            plot_resource_comparison(benchmark_json)
        else:
            print(f"未找到资源评测文件: {benchmark_json}")

    print(f"\n图表保存目录: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
