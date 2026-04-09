"""
Performance metrics collection for GPU benchmarking.

Tracks per-step timing, throughput, memory, and computes MFU
(Model FLOPS Utilization) against theoretical GPU peak.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .precision import GPUCapabilities, PrecisionMode, get_peak_tflops

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Metrics for a single training step."""
    step: int
    wall_time_ms: float
    tokens: int
    loss: float
    is_warmup: bool


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    # Identification
    model_name: str
    precision_mode: str
    gpu_name: str
    num_gpus: int
    batch_size: int
    seq_length: int
    num_params: int

    # Performance
    tokens_per_second: float
    achieved_tflops: float
    mfu: float                           # model flops utilization [0, 1]
    peak_memory_gb: float

    # Step timing (excluding warmup)
    step_time_mean_ms: float
    step_time_min_ms: float
    step_time_max_ms: float
    step_time_std_ms: float
    num_measured_steps: int

    # Optional: loss curve (for real data)
    final_loss: Optional[float] = None
    loss_values: list[float] = field(default_factory=list)

    # Config metadata
    arch_style: str = ""
    activation_checkpointing: bool = False
    torch_compile: bool = False
    compile_mode: str = ""
    fused_attn: bool = True
    sequence_parallel: bool = False
    gradient_accumulation_steps: int = 1
    total_steps: int = 0
    warmup_steps: int = 0
    flops_per_token: int = 0
    tp_size: int = 1

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "precision_mode": self.precision_mode,
            "gpu_name": self.gpu_name,
            "num_gpus": self.num_gpus,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "num_params": self.num_params,
            "arch_style": self.arch_style,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "achieved_tflops": round(self.achieved_tflops, 2),
            "mfu": round(self.mfu, 4),
            "peak_memory_gb": round(self.peak_memory_gb, 2),
            "step_time_mean_ms": round(self.step_time_mean_ms, 2),
            "step_time_min_ms": round(self.step_time_min_ms, 2),
            "step_time_max_ms": round(self.step_time_max_ms, 2),
            "step_time_std_ms": round(self.step_time_std_ms, 2),
            "num_measured_steps": self.num_measured_steps,
            "final_loss": round(self.final_loss, 4) if self.final_loss is not None else None,
            "activation_checkpointing": self.activation_checkpointing,
            "torch_compile": self.torch_compile,
            "compile_mode": self.compile_mode,
            "fused_attn": self.fused_attn,
            "sequence_parallel": self.sequence_parallel,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "flops_per_token": self.flops_per_token,
            "tp_size": self.tp_size,
        }


class BenchmarkMetrics:
    """
    Collects and aggregates training step metrics.

    Usage:
        metrics = BenchmarkMetrics(...)
        for step in range(total_steps):
            metrics.start_step(step)
            ... training step ...
            metrics.end_step(step, loss, tokens)
        result = metrics.get_result()
    """

    def __init__(
        self,
        model_name: str,
        precision_mode: PrecisionMode,
        gpu_caps: GPUCapabilities,
        num_params: int,
        flops_per_token: int,
        batch_size: int,
        seq_length: int,
        warmup_steps: int = 10,
        num_gpus: int = 1,
        arch_style: str = "",
        activation_checkpointing: bool = False,
        torch_compile: bool = False,
        compile_mode: str = "",
        fused_attn: bool = True,
        sequence_parallel: bool = False,
        gradient_accumulation_steps: int = 1,
        tp_size: int = 1,
    ):
        self.model_name = model_name
        self.precision_mode = precision_mode
        self.gpu_caps = gpu_caps
        self.num_params = num_params
        self.flops_per_token = flops_per_token
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.warmup_steps = warmup_steps
        self.num_gpus = num_gpus
        self.arch_style = arch_style
        self.activation_checkpointing = activation_checkpointing
        self.torch_compile = torch_compile
        self.compile_mode = compile_mode
        self.fused_attn = fused_attn
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.tp_size = tp_size
        self.log_interval = 50  # log every N measured steps

        self._records: list[StepRecord] = []
        self._step_start_events: dict[int, torch.cuda.Event] = {}
        self._step_end_events: dict[int, torch.cuda.Event] = {}
        self._step_start_time: dict[int, float] = {}

    def start_step(self, step: int) -> None:
        """Record the start of a training step using CUDA events for accurate timing."""
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._step_start_events[step] = start_event
        self._step_start_time[step] = time.perf_counter()

    def end_step(self, step: int, loss: float, tokens: int) -> None:
        """
        Record the end of a training step.

        Args:
            step: Step number (0-indexed)
            loss: Training loss value
            tokens: Number of tokens processed in this step
        """
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._step_end_events[step] = end_event

        # We'll compute elapsed time later after synchronization
        is_warmup = step < self.warmup_steps

        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        elapsed_ms = self._step_start_events[step].elapsed_time(end_event)

        record = StepRecord(
            step=step,
            wall_time_ms=elapsed_ms,
            tokens=tokens,
            loss=loss,
            is_warmup=is_warmup,
        )
        self._records.append(record)

        # Log progress (only at intervals to reduce verbosity)
        if is_warmup:
            if step == 0 or step == self.warmup_steps - 1:
                logger.info(f"  Step {step:4d} | loss={loss:.4f} | WARMUP")
        else:
            measured_step = step - self.warmup_steps
            is_first = (measured_step == 0)
            is_interval = (measured_step % self.log_interval == 0)
            if is_first or is_interval:
                tokens_per_sec = tokens / (elapsed_ms / 1000.0)
                logger.info(
                    f"  Step {step:4d} | loss={loss:.4f} | "
                    f"time={elapsed_ms:.1f}ms | "
                    f"tok/s={tokens_per_sec:,.0f}"
                )

    def get_result(self) -> BenchmarkResult:
        """
        Compute and return aggregated benchmark results.

        Excludes warmup steps from all performance calculations.
        """
        measured = [r for r in self._records if not r.is_warmup]

        if not measured:
            raise RuntimeError("No measured (non-warmup) steps recorded")

        step_times = [r.wall_time_ms for r in measured]
        total_tokens = sum(r.tokens for r in measured)
        total_time_s = sum(r.wall_time_ms for r in measured) / 1000.0

        # Throughput (aggregate across all data-parallel ranks)
        # With TP, all TP ranks process the same tokens — don't multiply by tp_size
        dp_size = self.num_gpus // self.tp_size
        tokens_per_second = (total_tokens / total_time_s) * dp_size

        # TFLOPS achieved (aggregate)
        # flops_per_token already includes the 3x for training (fwd + bwd)
        achieved_flops_per_sec = self.flops_per_token * tokens_per_second
        achieved_tflops = achieved_flops_per_sec / 1e12

        # MFU: fraction of theoretical peak (aggregate achieved / aggregate peak)
        peak_tflops = get_peak_tflops(self.gpu_caps, self.precision_mode)
        total_peak_tflops = peak_tflops * self.num_gpus
        mfu = achieved_tflops / total_peak_tflops if total_peak_tflops > 0 else 0.0

        # Peak memory
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / (1024**3)

        # Loss curve
        loss_values = [r.loss for r in measured]
        final_loss = loss_values[-1] if loss_values else None

        mean_time = statistics.mean(step_times)
        std_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

        return BenchmarkResult(
            model_name=self.model_name,
            precision_mode=str(self.precision_mode),
            gpu_name=self.gpu_caps.gpu_name,
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            num_params=self.num_params,
            tokens_per_second=tokens_per_second,
            achieved_tflops=achieved_tflops,
            mfu=mfu,
            peak_memory_gb=peak_memory_gb,
            step_time_mean_ms=mean_time,
            step_time_min_ms=min(step_times),
            step_time_max_ms=max(step_times),
            step_time_std_ms=std_time,
            num_measured_steps=len(measured),
            final_loss=final_loss,
            loss_values=loss_values,
            arch_style=self.arch_style,
            activation_checkpointing=self.activation_checkpointing,
            torch_compile=self.torch_compile,
            compile_mode=self.compile_mode,
            fused_attn=self.fused_attn,
            sequence_parallel=self.sequence_parallel,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            total_steps=len(self._records),
            warmup_steps=self.warmup_steps,
            flops_per_token=self.flops_per_token,
            tp_size=self.tp_size,
        )


def reset_memory_stats() -> None:
    """Reset CUDA memory tracking for fresh measurement."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
        torch.cuda.reset_accumulated_memory_stats()
