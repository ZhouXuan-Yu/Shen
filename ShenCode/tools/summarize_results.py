"""Summarize results from multiple runs into tables and figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_run_results(runs_dir: Path) -> list[dict]:
    """Load results from all runs."""
    results = []

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        with results_file.open("r", encoding="utf-8") as f:
            result = json.load(f)

        result["run_id"] = run_dir.name
        result["run_dir"] = str(run_dir)

        metrics_file = run_dir / "metrics_val_a.json"
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            result["top1_accuracy"] = metrics.get("top1_accuracy")
            result["top5_accuracy"] = metrics.get("top5_accuracy")
            result["macro_accuracy"] = metrics.get("macro_accuracy")

        results.append(result)

    return results


def group_results_by_config(results: list[dict]) -> dict:
    """Group results by model, K, and aggregate across seeds."""
    grouped = defaultdict(list)

    for r in results:
        key = (r.get("model", "unknown"), r.get("k", 0))
        grouped[key].append(r)

    return grouped


def compute_statistics(values: list[float]) -> dict:
    """Compute mean and std."""
    if not values:
        return {"mean": None, "std": None, "n": 0}

    import numpy as np

    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(values),
    }


def generate_summary_table(
    results: list[dict], benchmark_path: Path | None = None
) -> pd.DataFrame:
    """Generate summary table with mean±std across seeds."""
    grouped = group_results_by_config(results)

    benchmark_data = {}
    if benchmark_path and benchmark_path.exists():
        with benchmark_path.open("r", encoding="utf-8") as f:
            benchmark = json.load(f)
        for model_name, data in benchmark.get("models", {}).items():
            if "parameters" in data:
                benchmark_data[model_name] = {
                    "params_M": data["parameters"]["total_params_M"],
                    "latency_ms": data.get("batch_results", {})
                    .get("batch_1", {})
                    .get("latency", {})
                    .get("mean_ms"),
                    "throughput": data.get("batch_results", {})
                    .get("batch_16", {})
                    .get("throughput", {})
                    .get("samples_per_sec"),
                }

    rows = []
    for (model, k), run_list in sorted(grouped.items()):
        val_accs = [
            r.get("best_val_acc") for r in run_list if r.get("best_val_acc") is not None
        ]
        top1_accs = [
            r.get("top1_accuracy")
            for r in run_list
            if r.get("top1_accuracy") is not None
        ]
        top5_accs = [
            r.get("top5_accuracy")
            for r in run_list
            if r.get("top5_accuracy") is not None
        ]

        val_stats = compute_statistics(val_accs)
        top1_stats = compute_statistics(top1_accs)
        top5_stats = compute_statistics(top5_accs)

        row = {
            "Model": model,
            "K": k,
            "Seeds": val_stats["n"],
            "Val Acc (mean)": f"{val_stats['mean']:.2f}"
            if val_stats["mean"]
            else "N/A",
            "Val Acc (std)": f"{val_stats['std']:.2f}" if val_stats["std"] else "N/A",
            "Top-1 (mean)": f"{top1_stats['mean']:.2f}"
            if top1_stats["mean"]
            else "N/A",
            "Top-5 (mean)": f"{top5_stats['mean']:.2f}"
            if top5_stats["mean"]
            else "N/A",
        }

        if model in benchmark_data:
            bm = benchmark_data[model]
            row["Params(M)"] = f"{bm['params_M']:.2f}" if bm.get("params_M") else "N/A"
            row["Latency(ms)"] = (
                f"{bm['latency_ms']:.2f}" if bm.get("latency_ms") else "N/A"
            )
            row["Throughput"] = (
                f"{bm['throughput']:.1f}" if bm.get("throughput") else "N/A"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize results from multiple runs")
    parser.add_argument(
        "--runs-dir", type=str, default="runs", help="Directory containing run outputs"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="results/benchmark.json",
        help="Benchmark results file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables",
        help="Output directory for tables",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / args.runs_dir
    benchmark_path = project_root / args.benchmark
    output_dir = project_root / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {runs_dir}")
    results = load_run_results(runs_dir)
    print(f"Found {len(results)} runs")

    if not results:
        print("No results found. Run training first.")
        return

    print("\nGenerating summary table...")
    df = generate_summary_table(results, benchmark_path)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    csv_path = output_dir / "table_main.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTable saved to: {csv_path}")

    json_path = output_dir / "summary.json"
    summary_data = {
        "num_runs": len(results),
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
