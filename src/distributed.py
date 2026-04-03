"""
Distributed training setup with FSDP2.

Handles:
  - Process group initialization (NCCL backend)
  - FSDP2 wrapping of TransformerEngine models
  - Cleanup
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize distributed process group.

    Expects torchrun environment variables (RANK, WORLD_SIZE, LOCAL_RANK).

    Returns:
        (rank, world_size, local_rank)
    """
    if not dist.is_initialized():
        # torchrun sets these environment variables
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if world_size > 1:
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
            torch.cuda.set_device(local_rank)
            logger.info(f"Distributed: rank={rank}/{world_size}, local_rank={local_rank}")
        else:
            logger.info("Single GPU mode (no distributed)")
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    return rank, world_size, local_rank


def apply_fsdp2(
    model: torch.nn.Module,
    *,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """
    Apply FSDP2 (fully_shard) to a GPTModel.

    Wraps each TransformerLayer individually, then wraps the entire model.
    This provides per-layer sharding for memory efficiency.

    Args:
        model: GPTModel instance (already on CUDA)
        mixed_precision_dtype: Compute dtype for FSDP mixed precision

    Returns:
        FSDP2-wrapped model
    """
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    import transformer_engine.pytorch as te

    mp_policy = MixedPrecisionPolicy(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=torch.float32,
    )

    # Shard each TransformerLayer individually
    for i, layer in enumerate(model.layers):
        fully_shard(layer, mp_policy=mp_policy)
        logger.debug(f"  FSDP2 wrapped layer {i}")

    # Shard the entire model (top-level)
    fully_shard(model, mp_policy=mp_policy)

    logger.info(f"FSDP2 applied: {len(model.layers)} layers individually sharded")
    return model


def apply_activation_checkpointing(model: torch.nn.Module) -> None:
    """
    Apply activation checkpointing to transformer layers.

    This trades compute for memory by recomputing activations during backward.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
    )
    import transformer_engine.pytorch as te

    count = 0
    for i, layer in enumerate(model.layers):
        # Wrap each TE TransformerLayer with activation checkpointing
        model.layers[i] = checkpoint_wrapper(
            layer,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        count += 1

    logger.info(f"Activation checkpointing applied to {count} layers")


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    """Get the world size (number of processes)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
