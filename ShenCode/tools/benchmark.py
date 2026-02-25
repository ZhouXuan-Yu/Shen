"""Benchmark script for measuring model resource metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import create_model

try:
    from fvcore.nn import FlopCountAnalysis

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

try:
    from thop import profile as thop_profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": total / 1e6,
        "trainable_params_M": trainable / 1e6,
    }


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> dict | None:
    """Count FLOPs using fvcore or thop."""
    flops = None
    method = None

    if FVCORE_AVAILABLE:
        try:
            flop_counter = FlopCountAnalysis(model, input_tensor)
            flops = flop_counter.total()
            method = "fvcore"
        except Exception as e:
            print(f"fvcore failed: {e}")

    if flops is None and THOP_AVAILABLE:
        try:
            macs, _ = thop_profile(model, inputs=(input_tensor,), verbose=False)
            flops = macs * 2
            method = "thop"
        except Exception as e:
            print(f"thop failed: {e}")

    if flops is None:
        return None

    return {
        "flops": flops,
        "flops_G": flops / 1e9,
        "method": method,
    }


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 50,
    iterations: int = 200,
) -> dict:
    """Measure inference latency."""
    model.eval()
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies_sorted = sorted(latencies)

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "std_ms": (
            sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies)
            / len(latencies)
        )
        ** 0.5,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": latencies_sorted[len(latencies) // 2],
        "p95_ms": latencies_sorted[int(len(latencies) * 0.95)],
        "p99_ms": latencies_sorted[int(len(latencies) * 0.99)],
    }


def measure_throughput(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 20,
    duration_sec: float = 5.0,
) -> dict:
    """Measure inference throughput."""
    model.eval()
    input_tensor = input_tensor.to(device)
    batch_size = input_tensor.shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    count = 0
    start = time.perf_counter()

    with torch.no_grad():
        while time.perf_counter() - start < duration_sec:
            _ = model(input_tensor)
            count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed = end - start

    return {
        "samples_per_sec": (count * batch_size) / elapsed,
        "batches_per_sec": count / elapsed,
        "total_samples": count * batch_size,
        "duration_sec": elapsed,
    }


def measure_peak_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> dict | None:
    """Measure peak GPU memory usage."""
    if device.type != "cuda":
        return None

    model.eval()
    input_tensor = input_tensor.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        _ = model(input_tensor)

    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated(device)

    return {
        "peak_memory_bytes": peak_memory,
        "peak_memory_MB": peak_memory / (1024**2),
    }


def get_environment_info() -> dict:
    """Get environment information."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info


def benchmark_model(
    model_name: str,
    num_classes: int,
    input_size: int = 224,
    num_frames: int = 8,
    batch_sizes: list[int] = [1, 16],
    device: torch.device = None,
) -> dict:
    """Run full benchmark for a model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'=' * 60}")

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        for_video=True,
    )
    model = model.to(device)
    model.eval()

    params = count_parameters(model)
    print(f"Parameters: {params['total_params_M']:.2f}M")

    results = {
        "model_name": model_name,
        "num_classes": num_classes,
        "input_size": input_size,
        "num_frames": num_frames,
        "parameters": params,
        "batch_results": {},
    }

    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")

        input_tensor = torch.randn(
            batch_size, num_frames, 3, input_size, input_size
        ).to(device)

        batch_result = {}

        flops = count_flops(model, input_tensor)
        if flops:
            batch_result["flops"] = flops
            print(f"FLOPs: {flops['flops_G']:.2f}G ({flops['method']})")
        else:
            print("FLOPs: N/A (install fvcore or thop)")

        print("Measuring latency...")
        latency = measure_latency(model, input_tensor, device)
        batch_result["latency"] = latency
        print(
            f"Latency: {latency['mean_ms']:.2f}ms ± {latency['std_ms']:.2f}ms (P95: {latency['p95_ms']:.2f}ms)"
        )

        print("Measuring throughput...")
        throughput = measure_throughput(model, input_tensor, device)
        batch_result["throughput"] = throughput
        print(f"Throughput: {throughput['samples_per_sec']:.1f} samples/sec")

        memory = measure_peak_memory(model, input_tensor, device)
        if memory:
            batch_result["peak_memory"] = memory
            print(f"Peak Memory: {memory['peak_memory_MB']:.1f} MB")

        results["batch_results"][f"batch_{batch_size}"] = batch_result

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model resource metrics")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to benchmark (default: all P0 models)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=226, help="Number of classes"
    )
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames")
    parser.add_argument(
        "--batch-sizes", type=str, default="1,16", help="Batch sizes (comma-separated)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_info = get_environment_info()
    print(f"PyTorch: {env_info['torch_version']}")
    if env_info.get("gpu_name"):
        print(f"GPU: {env_info['gpu_name']}")

    if args.model:
        models_to_benchmark = [args.model]
    else:
        models_to_benchmark = ["resnet18", "resnet34", "mobilenetv2"]

    all_results = {
        "environment": env_info,
        "config": {
            "num_classes": args.num_classes,
            "input_size": args.input_size,
            "num_frames": args.num_frames,
            "batch_sizes": batch_sizes,
        },
        "models": {},
    }

    for model_name in models_to_benchmark:
        try:
            result = benchmark_model(
                model_name=model_name,
                num_classes=args.num_classes,
                input_size=args.input_size,
                num_frames=args.num_frames,
                batch_sizes=batch_sizes,
                device=device,
            )
            all_results["models"][model_name] = result
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            all_results["models"][model_name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Params(M)':<12} {'Latency(ms)':<15} {'Throughput':<15}")
    print("-" * 60)

    for model_name, result in all_results["models"].items():
        if "error" in result:
            print(f"{model_name:<15} ERROR: {result['error']}")
            continue

        params = result["parameters"]["total_params_M"]
        latency = (
            result["batch_results"]
            .get("batch_1", {})
            .get("latency", {})
            .get("mean_ms", "N/A")
        )
        throughput = (
            result["batch_results"]
            .get("batch_16", {})
            .get("throughput", {})
            .get("samples_per_sec", "N/A")
        )

        if isinstance(latency, float):
            latency = f"{latency:.2f}"
        if isinstance(throughput, float):
            throughput = f"{throughput:.1f}/s"

        print(f"{model_name:<15} {params:<12.2f} {latency:<15} {throughput:<15}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "results" / "benchmark.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
