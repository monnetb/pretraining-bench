"""
Main entry point for the transformer pretraining GPU benchmark.

Orchestrates model creation, dataset loading, training, and reporting.
Supports single-GPU and multi-GPU (FSDP2) modes, tensor parallelism,
multiple precision modes (BF16, FP8, MXFP8, NVFP4), and sweeps across
configurations.

Usage:
    # Single GPU, single config
    python -m src.benchmark --model-size small-gpt2 --precision bf16

    # All precisions for one model
    python -m src.benchmark --model-size small-llama --precision auto

    # Full sweep
    python -m src.benchmark --sweep

    # Multi-GPU with FSDP2
    torchrun --nproc_per_node=8 -m src.benchmark --model-size large-llama --precision auto

    # Multi-GPU with Tensor Parallelism (TP=8, no DP)
    torchrun --nproc_per_node=8 -m src.benchmark --model-size 70b-llama --tp-size 8

    # 2D parallelism: TP=4, DP=2
    torchrun --nproc_per_node=8 -m src.benchmark --model-size 70b-llama --tp-size 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Set up logging before heavy imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transformer-bench")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU benchmark for transformer pretraining with TransformerEngine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-size",
        type=str,
        default="small-gpt2",
        help=(
            "Model config name from model_configs.yaml. "
            "Use 'all' to run all models, or specify comma-separated names. "
            "Available: small-gpt2, small-llama, medium-gpt2, medium-llama, "
            "large-gpt2, large-llama, xlarge-gpt2, xlarge-llama, 8b-llama, "
            "30b-llama, 70b-llama"
        ),
    )

    # Precision
    prec_group = parser.add_argument_group("Precision")
    prec_group.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help=(
            "Precision mode: bf16, fp8-delayed, fp8-current, fp8-block, "
            "mxfp8, nvfp4, auto (best available), all (run all available)"
        ),
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "tiny-text"],
        help="Dataset type",
    )
    data_group.add_argument(
        "--text-path", type=str, default=None,
        help="Path to text file for tiny-text dataset",
    )
    data_group.add_argument(
        "--seq-length", type=int, default=2048,
        help="Sequence length",
    )
    data_group.add_argument(
        "--batch-size", type=int, default=8,
        help="Per-GPU micro batch size",
    )
    data_group.add_argument(
        "--num-samples", type=int, default=10000,
        help="Number of samples in synthetic dataset",
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num-steps", type=int, default=100,
        help="Total training steps",
    )
    train_group.add_argument(
        "--warmup-steps", type=int, default=10,
        help="Warmup steps excluded from measurements",
    )
    train_group.add_argument(
        "--lr", type=float, default=3e-4,
        help="Peak learning rate",
    )
    train_group.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Gradient clipping max norm (0 to disable)",
    )

    # Parallelism
    par_group = parser.add_argument_group("Parallelism")
    par_group.add_argument(
        "--tp-size", type=int, default=1,
        help=(
            "Tensor parallelism size. Must divide world_size evenly. "
            "dp_size = world_size / tp_size. Use tp_size > 1 for models "
            "that don't fit with FSDP-only (e.g., 70b-llama)."
        ),
    )

    # Optimization
    opt_group = parser.add_argument_group("Optimizations")
    opt_group.add_argument(
        "--activation-checkpointing", action="store_true",
        help="Enable activation checkpointing to save memory",
    )
    opt_group.add_argument(
        "--use-compile", action="store_true",
        help="Apply torch.compile to the model",
    )

    # Sweep
    sweep_group = parser.add_argument_group("Sweep")
    sweep_group.add_argument(
        "--sweep", action="store_true",
        help="Run all model×precision combinations",
    )
    sweep_group.add_argument(
        "--sweep-models", type=str, default=None,
        help="Comma-separated model names for sweep (default: all)",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for results output",
    )
    out_group.add_argument(
        "--json-filename", type=str, default=None,
        help="Override JSON output filename",
    )

    # Misc
    parser.add_argument(
        "--config-path", type=str, default=None,
        help="Path to model_configs.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def find_config_path(override: Optional[str] = None) -> str:
    """Locate the model_configs.yaml file."""
    if override and os.path.exists(override):
        return override

    # Search relative to this file's location
    candidates = [
        Path(__file__).parent.parent / "configs" / "model_configs.yaml",
        Path.cwd() / "configs" / "model_configs.yaml",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Cannot find model_configs.yaml. Use --config-path to specify location."
    )


def run_single_benchmark(
    model_name: str,
    precision_mode_str: str,
    args: argparse.Namespace,
    config_data: dict,
    gpu_caps,
    rank: int,
    world_size: int,
    parallel_groups=None,
):
    """
    Run a single benchmark configuration.

    Returns BenchmarkResult on rank 0, None on other ranks.
    """
    import torch

    from .model import load_model_config, build_model, compute_flops_per_token
    from .data import get_dataloader
    from .metrics import BenchmarkMetrics, BenchmarkResult, reset_memory_stats
    from .precision import PrecisionMode, get_available_precisions
    from .trainer import Trainer
    from .distributed import apply_fsdp2, apply_activation_checkpointing, is_main_process
    from .report import print_single_result

    precision_mode = PrecisionMode(precision_mode_str)
    tp_size = args.tp_size if parallel_groups is not None else 1
    dp_size = world_size // tp_size

    # Check if precision mode is available
    available = get_available_precisions(gpu_caps)
    if precision_mode not in available:
        if is_main_process():
            logger.warning(
                f"Precision {precision_mode} not available on {gpu_caps.gpu_name}. "
                f"Available: {[str(m) for m in available]}. Skipping."
            )
        return None

    if is_main_process():
        logger.info(f"\n{'=' * 60}")
        tp_info = f" | TP={tp_size}" if tp_size > 1 else ""
        logger.info(f"  Benchmark: {model_name} | {precision_mode}{tp_info}")
        logger.info(f"{'=' * 60}")

    # Load model config
    model_config = load_model_config(
        config_data["models"], model_name,
        vocab_size=config_data.get("vocab_size", 50257),
    )
    # Override seq_length from CLI if specified
    model_config.max_seq_length = args.seq_length

    # Build model with TP support
    device = torch.device("cuda")
    tp_group = parallel_groups.tp_group if parallel_groups else None
    model = build_model(model_config, device=device, tp_group=tp_group, tp_size=tp_size)

    # Get parameter counts — use config-level count for total (architecture-level, independent of TP)
    num_params = model_config.num_params_approx
    flops_per_token = compute_flops_per_token(model_config)

    # Apply FSDP2 for data parallelism (when dp_size > 1)
    if dp_size > 1:
        dp_group = parallel_groups.dp_group if parallel_groups else None
        model = apply_fsdp2(model, dp_group=dp_group)

    # Activation checkpointing
    if args.activation_checkpointing:
        apply_activation_checkpointing(model)

    # Create dataloader — distributed sampler uses DP group, not full world
    dp_rank = parallel_groups.dp_rank if parallel_groups else None
    dataloader = get_dataloader(
        dataset_type=args.dataset,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        text_path=args.text_path,
        distributed=(dp_size > 1),
        dp_rank=dp_rank if dp_size > 1 else None,
        dp_size=dp_size if dp_size > 1 else None,
    )

    # Create metrics collector
    metrics = BenchmarkMetrics(
        model_name=model_name,
        precision_mode=precision_mode,
        gpu_caps=gpu_caps,
        num_params=num_params,
        flops_per_token=flops_per_token,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        warmup_steps=args.warmup_steps,
        num_gpus=world_size,
        arch_style=model_config.arch_style,
        activation_checkpointing=args.activation_checkpointing,
        torch_compile=args.use_compile,
        tp_size=tp_size,
    )

    # Create trainer and run
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        precision_mode=precision_mode,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        use_compile=args.use_compile,
    )

    # Reset memory tracking
    reset_memory_stats()

    # Run training
    trainer.train()

    # Collect results
    result = metrics.get_result()

    if is_main_process():
        print_single_result(result)

    # Clean up model to free GPU memory for next benchmark
    del trainer, model, dataloader
    torch.cuda.empty_cache()

    return result if is_main_process() else None


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Lazy imports — allows --help to work without torch/TE installed
    import torch
    import yaml

    # --- Distributed setup ---
    from .distributed import (
        setup_distributed, cleanup_distributed, is_main_process, barrier,
        setup_parallel_groups,
    )

    rank, world_size, local_rank = setup_distributed()

    if is_main_process():
        logger.info("=" * 60)
        logger.info("  Transformer Pretraining GPU Benchmark")
        logger.info("=" * 60)

    try:
        # --- Validate TP size ---
        tp_size = args.tp_size
        if tp_size > 1:
            if world_size % tp_size != 0:
                logger.error(
                    f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
                )
                sys.exit(1)
            if world_size < tp_size:
                logger.error(
                    f"tp_size ({tp_size}) cannot exceed world_size ({world_size})"
                )
                sys.exit(1)

        # --- Setup parallel groups (TP + DP) ---
        parallel_groups = None
        if world_size > 1 and tp_size > 1:
            parallel_groups = setup_parallel_groups(world_size, tp_size)
            if is_main_process():
                dp_size = world_size // tp_size
                logger.info(f"Parallelism: TP={tp_size} × DP={dp_size}")
        elif world_size > 1:
            # Pure FSDP mode — create trivial parallel groups for consistency
            parallel_groups = setup_parallel_groups(world_size, tp_size=1)

        # --- GPU detection ---
        from .precision import (
            detect_gpu_capabilities,
            get_available_precisions,
            precision_summary,
            PrecisionMode,
        )

        gpu_caps = detect_gpu_capabilities()

        if is_main_process():
            logger.info(f"\n{precision_summary(gpu_caps)}\n")

        # --- Load model configs ---
        config_path = find_config_path(args.config_path)
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        all_model_names = list(config_data["models"].keys())

        # --- Determine which models to benchmark ---
        if args.sweep:
            model_names = (
                args.sweep_models.split(",") if args.sweep_models
                else all_model_names
            )
        elif args.model_size == "all":
            model_names = all_model_names
        else:
            model_names = args.model_size.split(",")

        # Validate model names
        for name in model_names:
            if name not in config_data["models"]:
                logger.error(
                    f"Unknown model '{name}'. Available: {all_model_names}"
                )
                sys.exit(1)

        # --- Determine which precisions to test ---
        available_precisions = get_available_precisions(gpu_caps)

        if args.sweep or args.precision == "all":
            precision_modes = [str(p) for p in available_precisions]
        elif args.precision == "auto":
            # Use the best available precision
            precision_modes = [str(available_precisions[-1])]
        else:
            precision_modes = args.precision.split(",")

        total_runs = len(model_names) * len(precision_modes)

        if is_main_process():
            logger.info(f"Models:     {model_names}")
            logger.info(f"Precisions: {precision_modes}")
            if tp_size > 1:
                logger.info(f"TP size:    {tp_size}")
            logger.info(f"Total runs: {total_runs}")

        # --- Run benchmarks ---
        from .report import print_summary_table, save_json, save_csv

        all_results = []
        run_idx = 0

        for model_name in model_names:
            for prec in precision_modes:
                run_idx += 1
                if is_main_process():
                    logger.info(
                        f"\n>>> Run {run_idx}/{total_runs}: "
                        f"{model_name} @ {prec}"
                    )

                barrier()

                try:
                    result = run_single_benchmark(
                        model_name=model_name,
                        precision_mode_str=prec,
                        args=args,
                        config_data=config_data,
                        gpu_caps=gpu_caps,
                        rank=rank,
                        world_size=world_size,
                        parallel_groups=parallel_groups,
                    )
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    if is_main_process():
                        logger.error(
                            f"Benchmark failed for {model_name}@{prec}: {e}",
                            exc_info=True,
                        )
                    # Continue with next configuration
                    torch.cuda.empty_cache()
                    continue

        # --- Final report ---
        if is_main_process() and all_results:
            print_summary_table(all_results, title="Benchmark Summary")

            # Save results
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                args.output_dir,
            )
            json_path = save_json(all_results, output_dir, args.json_filename)
            csv_path = save_csv(all_results, output_dir)

            logger.info(f"JSON results: {json_path}")
            logger.info(f"CSV results:  {csv_path}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
