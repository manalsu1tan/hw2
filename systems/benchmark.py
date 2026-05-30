from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    from basics.model import BasicsTransformerLM

    spec = MODEL_SPECS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10_000.0,
    )
    if config.compile_model:
        model = torch.compile(model)
    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        dtype=torch.long,
        device=device,
    )


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    optimizer = getattr(model, "_benchmark_optimizer", None)
    if mode == "train-step" and optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        setattr(model, "_benchmark_optimizer", optimizer)

    if mode in {"forward-backward", "train-step"}:
        model.zero_grad(set_to_none=True)

    with autocast_context:
        outputs = model(batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]), batch[:, 1:].contiguous().view(-1))

    if mode in {"forward-backward", "train-step"}:
        loss.backward()
    if mode == "train-step":
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if batch.is_cuda:
        torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.train(config.mode != "forward")
    batch = make_random_batch(config, device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    autocast_context = make_autocast_context(config.use_bf16 and device.type == "cuda")
    for _ in range(config.warmup_steps):
        run_single_step(model, batch, config.mode, autocast_context)

    maybe_start_memory_history(config.use_memory_profiler and device.type == "cuda")
    timings: list[float] = []
    for _ in range(config.measure_steps):
        start = timeit.default_timer()
        run_single_step(model, batch, config.mode, autocast_context)
        timings.append(timeit.default_timer() - start)
    maybe_dump_memory_snapshot(
        config.use_memory_profiler and device.type == "cuda",
        config.output_dir / f"{config.model_size}_{config.context_length}_{config.mode}_memory.pickle",
    )

    result = {
        "mean_seconds": statistics.fmean(timings),
        "std_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }
    print(result)
    return result


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    import torch.cuda.nvtx as nvtx
    from einops import einsum
    from basics.nn_utils import softmax

    q = kwargs.get("Q", args[0] if args else None)
    k = kwargs.get("K", args[1] if len(args) > 1 else None)
    v = kwargs.get("V", args[2] if len(args) > 2 else None)
    mask = kwargs.get("mask", args[3] if len(args) > 3 else None)
    if q is None or k is None or v is None:
        raise TypeError("expected Q, K, and V tensors")

    with nvtx.range("scaled dot product attention"):
        with nvtx.range("computing attention scores"):
            attention_scores = einsum(q, k, "... query d_k, ... key d_k -> ... query key") / math.sqrt(k.shape[-1])
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with nvtx.range("final matmul"):
            return einsum(attention_weights, v, "... query key, ... key d_v -> ... query d_v")


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
