from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    shape = (batch_size, sequence_length, head_dim)
    return tuple(torch.randn(shape, device=device, requires_grad=True) for _ in range(3))


def benchmark_attention_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    from basics.model import scaled_dot_product_attention

    if getattr(benchmark_attention_once, "_compiled", False):
        attention = torch.compile(scaled_dot_product_attention)
    else:
        attention = scaled_dot_product_attention

    mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device))
    for _ in range(5):
        out = attention(q, k, v, mask=mask)
        if q.is_cuda:
            torch.cuda.synchronize()
        out.sum().backward()
        for tensor in (q, k, v):
            tensor.grad = None
        if q.is_cuda:
            torch.cuda.synchronize()

    forward_timings: list[float] = []
    with torch.no_grad():
        for _ in range(100):
            start = timeit.default_timer()
            attention(q, k, v, mask=mask)
            if q.is_cuda:
                torch.cuda.synchronize()
            forward_timings.append(timeit.default_timer() - start)

    memory_before_backward = torch.cuda.memory_allocated(q.device) if q.is_cuda else 0
    backward_timings: list[float] = []
    for _ in range(100):
        for tensor in (q, k, v):
            tensor.grad = None
        out = attention(q, k, v, mask=mask)
        loss = out.sum()
        if q.is_cuda:
            torch.cuda.synchronize()
        start = timeit.default_timer()
        loss.backward()
        if q.is_cuda:
            torch.cuda.synchronize()
        backward_timings.append(timeit.default_timer() - start)

    return {
        "forward_mean_seconds": statistics.fmean(forward_timings),
        "forward_std_seconds": statistics.stdev(forward_timings),
        "backward_mean_seconds": statistics.fmean(backward_timings),
        "backward_std_seconds": statistics.stdev(backward_timings),
        "memory_before_backward_bytes": float(memory_before_backward),
    }


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(benchmark_attention_once, "_compiled", config.compile_attention)
    results: list[dict[str, float | int | str]] = []
    for head_dim, sequence_length in iter_benchmark_shapes(config):
        try:
            q, k, v = make_qkv(config.batch_size, sequence_length, head_dim, device)
            metrics = benchmark_attention_once(q, k, v)
            row: dict[str, float | int | str] = {
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "status": "ok",
                **metrics,
            }
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row = {
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "status": f"oom: {exc}",
            }
        print(row)
        results.append(row)
    return results


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
