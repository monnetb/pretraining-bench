"""
Results formatting and export.

Generates:
  - Console summary table (using tabulate)
  - JSON results file with full metadata
  - CSV export for spreadsheet comparison
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .metrics import BenchmarkResult

logger = logging.getLogger(__name__)


# CSV columns for export
CSV_COLUMNS = [
    "model_name",
    "arch_style",
    "precision_mode",
    "gpu_name",
    "num_gpus",
    "batch_size",
    "seq_length",
    "num_params",
    "tokens_per_second",
    "achieved_tflops",
    "mfu",
    "peak_memory_gb",
    "step_time_mean_ms",
    "step_time_min_ms",
    "step_time_max_ms",
    "step_time_std_ms",
    "num_measured_steps",
    "final_loss",
    "activation_checkpointing",
    "torch_compile",
]


def print_summary_table(
    results: list[BenchmarkResult],
    title: str = "Benchmark Results",
) -> None:
    """
    Print a formatted summary table to the console.

    Args:
        results: List of benchmark results
        title: Table title
    """
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback to simple formatting
        _print_simple_table(results, title)
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    if not results:
        print("  No results to display.")
        return

    # Print GPU info from first result
    r0 = results[0]
    print(f"  GPU: {r0.gpu_name} × {r0.num_gpus}")
    print(f"{'─' * 80}")

    headers = [
        "Model",
        "Precision",
        "Tok/s",
        "TFLOPS",
        "MFU",
        "Mem (GB)",
        "Step (ms)",
        "Loss",
    ]

    rows = []
    for r in results:
        rows.append([
            r.model_name,
            r.precision_mode,
            f"{r.tokens_per_second:,.0f}",
            f"{r.achieved_tflops:.1f}",
            f"{r.mfu:.1%}",
            f"{r.peak_memory_gb:.1f}",
            f"{r.step_time_mean_ms:.1f}±{r.step_time_std_ms:.1f}",
            f"{r.final_loss:.3f}" if r.final_loss is not None else "n/a",
        ])

    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))
    print(f"{'=' * 80}\n")


def _print_simple_table(results: list[BenchmarkResult], title: str) -> None:
    """Fallback table printer without tabulate dependency."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    header = (
        f"{'Model':<20} {'Precision':<14} {'Tok/s':>10} {'TFLOPS':>8} "
        f"{'MFU':>7} {'Mem(GB)':>8} {'Step(ms)':>12} {'Loss':>8}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        line = (
            f"{r.model_name:<20} {r.precision_mode:<14} "
            f"{r.tokens_per_second:>10,.0f} {r.achieved_tflops:>8.1f} "
            f"{r.mfu:>6.1%} {r.peak_memory_gb:>8.1f} "
            f"{r.step_time_mean_ms:>5.1f}±{r.step_time_std_ms:<5.1f} "
            f"{'n/a' if r.final_loss is None else f'{r.final_loss:.3f}':>8}"
        )
        print(line)

    print(f"{'=' * 100}\n")


def save_json(
    results: list[BenchmarkResult],
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save full results to a JSON file.

    Args:
        results: List of benchmark results
        output_dir: Directory to save to
        filename: Optional filename override

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu = results[0].gpu_name.replace(" ", "_") if results else "unknown"
        filename = f"benchmark_{gpu}_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "results": [r.to_dict() for r in results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to: {filepath}")
    return filepath


def save_csv(
    results: list[BenchmarkResult],
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save results as CSV for spreadsheet analysis.

    Args:
        results: List of benchmark results
        output_dir: Directory to save to
        filename: Optional filename override

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu = results[0].gpu_name.replace(" ", "_") if results else "unknown"
        filename = f"benchmark_{gpu}_{timestamp}.csv"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    logger.info(f"CSV saved to: {filepath}")
    return filepath


def print_single_result(result: BenchmarkResult) -> None:
    """Print a detailed summary of a single benchmark result."""
    print(f"\n{'─' * 60}")
    print(f"  {result.model_name} | {result.precision_mode}")
    print(f"{'─' * 60}")
    print(f"  GPU:            {result.gpu_name} × {result.num_gpus}")
    print(f"  Parameters:     {result.num_params:,}")
    print(f"  Batch × Seq:    {result.batch_size} × {result.seq_length}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Throughput:     {result.tokens_per_second:,.0f} tok/s")
    print(f"  TFLOPS:         {result.achieved_tflops:.1f}")
    print(f"  MFU:            {result.mfu:.1%}")
    print(f"  Peak Memory:    {result.peak_memory_gb:.1f} GB")
    print(f"  Step Time:      {result.step_time_mean_ms:.1f} ± {result.step_time_std_ms:.1f} ms")
    print(f"  Step Range:     [{result.step_time_min_ms:.1f}, {result.step_time_max_ms:.1f}] ms")
    if result.final_loss is not None:
        print(f"  Final Loss:     {result.final_loss:.4f}")
    print(f"{'─' * 60}\n")
