"""
GPU capability detection and TransformerEngine precision recipe configuration.

Handles auto-detection of available precision modes (BF16, FP8, MXFP8, NVFP4)
and creates the appropriate TE recipe objects for each mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Precision mode enumeration
# ============================================================================

class PrecisionMode(str, Enum):
    """Supported precision modes for benchmarking."""
    BF16 = "bf16"
    FP8_DELAYED = "fp8-delayed"
    FP8_CURRENT = "fp8-current"
    FP8_BLOCK = "fp8-block"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# GPU specification database (per-GPU theoretical peak FLOPS)
# ============================================================================
# Values are in TFLOPS (teraflops) for a single GPU.
# Sources: NVIDIA datasheets, product briefs.

@dataclass
class GPUSpec:
    """Theoretical peak performance for a single GPU."""
    name: str
    bf16_tflops: float
    fp8_tflops: float
    fp4_tflops: float        # 0 if not supported
    memory_gb: float
    nvlink_bw_gbps: float    # per-GPU NVLink bandwidth (GB/s)

# Per-GPU specs (single GPU, not system-level)
GPU_SPECS: dict[str, GPUSpec] = {
    # Blackwell
    "NVIDIA B300": GPUSpec("B300 SXM", bf16_tflops=4500, fp8_tflops=9000, fp4_tflops=18000, memory_gb=288, nvlink_bw_gbps=225),
    "NVIDIA B200": GPUSpec("B200 SXM", bf16_tflops=4500, fp8_tflops=9000, fp4_tflops=13500, memory_gb=192, nvlink_bw_gbps=225),
    # Hopper
    "NVIDIA H200": GPUSpec("H200 SXM", bf16_tflops=1979, fp8_tflops=3958, fp4_tflops=0, memory_gb=141, nvlink_bw_gbps=112.5),
    "NVIDIA H100": GPUSpec("H100 SXM", bf16_tflops=1979, fp8_tflops=3958, fp4_tflops=0, memory_gb=80, nvlink_bw_gbps=112.5),
    "NVIDIA H100 80GB HBM3": GPUSpec("H100 SXM", bf16_tflops=1979, fp8_tflops=3958, fp4_tflops=0, memory_gb=80, nvlink_bw_gbps=112.5),
    # Ada Lovelace (PCIe, no NVLink typically)
    "NVIDIA L40S": GPUSpec("L40S", bf16_tflops=366, fp8_tflops=733, fp4_tflops=0, memory_gb=48, nvlink_bw_gbps=0),
    # Ampere (no FP8)
    "NVIDIA A100": GPUSpec("A100 SXM", bf16_tflops=312, fp8_tflops=0, fp4_tflops=0, memory_gb=80, nvlink_bw_gbps=75),
    "NVIDIA A100 80GB PCIe": GPUSpec("A100 PCIe", bf16_tflops=312, fp8_tflops=0, fp4_tflops=0, memory_gb=80, nvlink_bw_gbps=0),
}


@dataclass
class GPUCapabilities:
    """Detected capabilities for the current GPU."""
    gpu_name: str
    compute_capability: tuple[int, int]
    total_memory_gb: float
    num_gpus: int
    has_fp8: bool = False
    has_mxfp8: bool = False
    has_nvfp4: bool = False
    has_fp8_block_scaling: bool = False
    spec: Optional[GPUSpec] = None


def detect_gpu_capabilities() -> GPUCapabilities:
    """
    Detect the current GPU and its precision capabilities.

    Returns:
        GPUCapabilities with all detected features.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires an NVIDIA GPU.")

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    cc = torch.cuda.get_device_capability(device)
    mem_bytes = torch.cuda.get_device_properties(device).total_memory
    num_gpus = torch.cuda.device_count()

    caps = GPUCapabilities(
        gpu_name=gpu_name,
        compute_capability=cc,
        total_memory_gb=mem_bytes / (1024**3),
        num_gpus=num_gpus,
    )

    # Probe TransformerEngine precision support
    try:
        import transformer_engine.pytorch as te_pytorch
        _te_available = True
    except (ImportError, OSError) as e:
        logger.warning(f"TransformerEngine not available ({e}). FP8/MXFP8/NVFP4 disabled.")
        _te_available = False

    if _te_available:
        try:
            caps.has_fp8 = te_pytorch.is_fp8_available()
        except Exception:
            caps.has_fp8 = False

        try:
            caps.has_mxfp8 = te_pytorch.is_mxfp8_available()
        except (AttributeError, Exception):
            caps.has_mxfp8 = False

        try:
            caps.has_nvfp4 = te_pytorch.is_nvfp4_available()
        except (AttributeError, Exception):
            caps.has_nvfp4 = False

        try:
            caps.has_fp8_block_scaling = te_pytorch.is_fp8_block_scaling_available()
        except (AttributeError, Exception):
            caps.has_fp8_block_scaling = False

    # Match GPU spec from database
    for key, spec in GPU_SPECS.items():
        if key in gpu_name:
            caps.spec = spec
            break

    if caps.spec is None:
        logger.warning(
            f"GPU '{gpu_name}' not in spec database. MFU calculation will use "
            f"estimated values based on compute capability {cc}."
        )
        # Estimate based on compute capability
        if cc >= (9, 0):   # Blackwell
            caps.spec = GPUSpec(gpu_name, bf16_tflops=4500, fp8_tflops=9000, fp4_tflops=18000, memory_gb=caps.total_memory_gb, nvlink_bw_gbps=225)
        elif cc >= (8, 9):  # Ada
            caps.spec = GPUSpec(gpu_name, bf16_tflops=366, fp8_tflops=733, fp4_tflops=0, memory_gb=caps.total_memory_gb, nvlink_bw_gbps=0)
        elif cc >= (8, 0):  # Hopper
            caps.spec = GPUSpec(gpu_name, bf16_tflops=1979, fp8_tflops=3958, fp4_tflops=0, memory_gb=caps.total_memory_gb, nvlink_bw_gbps=112.5)
        else:               # Ampere or older
            caps.spec = GPUSpec(gpu_name, bf16_tflops=312, fp8_tflops=0, fp4_tflops=0, memory_gb=caps.total_memory_gb, nvlink_bw_gbps=0)

    logger.info(f"Detected GPU: {gpu_name} (CC {cc[0]}.{cc[1]})")
    logger.info(f"  Memory: {caps.total_memory_gb:.1f} GB | FP8: {caps.has_fp8} | MXFP8: {caps.has_mxfp8} | NVFP4: {caps.has_nvfp4}")

    return caps


def get_available_precisions(caps: GPUCapabilities) -> list[PrecisionMode]:
    """Return list of precision modes available on the detected GPU."""
    modes = [PrecisionMode.BF16]

    if caps.has_fp8:
        modes.append(PrecisionMode.FP8_DELAYED)
        modes.append(PrecisionMode.FP8_CURRENT)

    if caps.has_fp8_block_scaling:
        modes.append(PrecisionMode.FP8_BLOCK)

    if caps.has_mxfp8:
        modes.append(PrecisionMode.MXFP8)

    if caps.has_nvfp4:
        modes.append(PrecisionMode.NVFP4)

    return modes


def create_recipe(mode: PrecisionMode) -> Any:
    """
    Create a TransformerEngine precision recipe for the given mode.

    Returns:
        A TE recipe object, or None for BF16 (no recipe needed).
    """
    if mode == PrecisionMode.BF16:
        return None

    from transformer_engine.common.recipe import DelayedScaling, Format

    if mode == PrecisionMode.FP8_DELAYED:
        return DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=1024,
            amax_compute_algo="max",
        )

    if mode == PrecisionMode.FP8_CURRENT:
        from transformer_engine.common.recipe import Float8CurrentScaling
        return Float8CurrentScaling()

    if mode == PrecisionMode.FP8_BLOCK:
        from transformer_engine.common.recipe import Float8BlockScaling
        return Float8BlockScaling()

    if mode == PrecisionMode.MXFP8:
        from transformer_engine.common.recipe import MXFP8BlockScaling
        return MXFP8BlockScaling()

    if mode == PrecisionMode.NVFP4:
        from transformer_engine.common.recipe import NVFP4BlockScaling
        return NVFP4BlockScaling()

    raise ValueError(f"Unknown precision mode: {mode}")


def get_peak_tflops(caps: GPUCapabilities, mode: PrecisionMode) -> float:
    """
    Get the theoretical peak TFLOPS for a given GPU and precision mode.

    Used for MFU (Model FLOPS Utilization) calculation.
    """
    spec = caps.spec
    if spec is None:
        raise RuntimeError("No GPU spec available for MFU calculation")

    if mode == PrecisionMode.BF16:
        return spec.bf16_tflops
    elif mode in (PrecisionMode.FP8_DELAYED, PrecisionMode.FP8_CURRENT,
                  PrecisionMode.FP8_BLOCK, PrecisionMode.MXFP8):
        return spec.fp8_tflops if spec.fp8_tflops > 0 else spec.bf16_tflops
    elif mode == PrecisionMode.NVFP4:
        return spec.fp4_tflops if spec.fp4_tflops > 0 else spec.fp8_tflops
    else:
        return spec.bf16_tflops


def precision_summary(caps: GPUCapabilities) -> str:
    """Return a human-readable summary of GPU precision capabilities."""
    modes = get_available_precisions(caps)
    lines = [
        f"GPU: {caps.gpu_name}",
        f"Compute Capability: {caps.compute_capability[0]}.{caps.compute_capability[1]}",
        f"Memory: {caps.total_memory_gb:.1f} GB",
        f"Available precisions: {', '.join(str(m) for m in modes)}",
    ]
    if caps.spec:
        lines.append(f"Peak BF16: {caps.spec.bf16_tflops} TFLOPS")
        if caps.spec.fp8_tflops > 0:
            lines.append(f"Peak FP8:  {caps.spec.fp8_tflops} TFLOPS")
        if caps.spec.fp4_tflops > 0:
            lines.append(f"Peak FP4:  {caps.spec.fp4_tflops} TFLOPS")
    return "\n".join(lines)
