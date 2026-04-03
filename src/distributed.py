"""
Distributed training setup with FSDP2 and Tensor Parallelism.

Handles:
  - Process group initialization (NCCL backend)
  - Tensor parallel / data parallel process group creation
  - FSDP2 wrapping of TransformerEngine models
  - Vocab-parallel embedding and LM head for TP
  - Cleanup
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
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


# ============================================================================
# Tensor Parallelism process groups
# ============================================================================

@dataclass
class ParallelGroups:
    """Process groups for hybrid TP + DP parallelism."""
    tp_group: Optional[dist.ProcessGroup]
    dp_group: Optional[dist.ProcessGroup]
    tp_size: int
    dp_size: int
    tp_rank: int   # rank within TP group
    dp_rank: int   # rank within DP group


def setup_parallel_groups(world_size: int, tp_size: int) -> ParallelGroups:
    """
    Create tensor-parallel and data-parallel process groups.

    With 8 GPUs and tp_size=4:
      TP groups: [0,1,2,3], [4,5,6,7]
      DP groups: [0,4], [1,5], [2,6], [3,7]

    Args:
        world_size: Total number of GPUs
        tp_size: Number of GPUs per tensor-parallel group

    Returns:
        ParallelGroups with TP and DP groups for this rank
    """
    assert world_size % tp_size == 0, (
        f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    )
    dp_size = world_size // tp_size
    rank = dist.get_rank()

    # No parallelism needed
    if tp_size == 1:
        return ParallelGroups(
            tp_group=None,
            dp_group=None,
            tp_size=1,
            dp_size=dp_size,
            tp_rank=0,
            dp_rank=rank,
        )

    # Create TP groups: contiguous ranks
    tp_group = None
    for i in range(0, world_size, tp_size):
        tp_ranks = list(range(i, i + tp_size))
        group = dist.new_group(tp_ranks)
        if rank in tp_ranks:
            tp_group = group
            tp_rank = tp_ranks.index(rank)

    # Create DP groups: strided ranks
    dp_group = None
    for i in range(tp_size):
        dp_ranks = list(range(i, world_size, tp_size))
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = group
            dp_rank = dp_ranks.index(rank)

    logger.info(
        f"Parallel groups: tp_size={tp_size}, dp_size={dp_size}, "
        f"tp_rank={tp_rank}, dp_rank={dp_rank}"
    )

    return ParallelGroups(
        tp_group=tp_group,
        dp_group=dp_group,
        tp_size=tp_size,
        dp_size=dp_size,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
    )


# ============================================================================
# Vocab-parallel modules for tensor parallelism
# ============================================================================

class _AllReduceFunc(torch.autograd.Function):
    """Autograd-compatible all-reduce for embedding forward (sum partial embeddings)."""
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        output = input_.clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # In backward, gradients are already correct per-rank for the embedding
        # (each rank only has non-zero grad for its own shard).
        # No all-reduce needed — just pass through.
        return grad_output, None


class _AllGatherFunc(torch.autograd.Function):
    """Autograd-compatible all-gather along last dimension."""
    @staticmethod
    def forward(ctx, input_, group, world_size):
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = dist.get_rank(group)
        gathered = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(gathered, input_, group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        # Split grad along last dim, keep only this rank's shard
        chunks = grad_output.chunk(ctx.world_size, dim=-1)
        return chunks[ctx.rank].contiguous(), None, None


class VocabParallelEmbedding(nn.Module):
    """
    Embedding table split across TP ranks along the vocabulary dimension.

    Each rank holds a shard of vocab_size // tp_size rows.
    On forward, tokens outside the local shard produce zeros;
    an all-reduce across the TP group sums them to get the correct embeddings.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_group: dist.ProcessGroup,
        tp_size: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank(tp_group)

        # Compute local shard bounds
        # Pad vocab to be divisible by tp_size
        self.padded_vocab_size = ((num_embeddings + tp_size - 1) // tp_size) * tp_size
        self.local_vocab_size = self.padded_vocab_size // tp_size
        self.vocab_start = self.tp_rank * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size

        self.embedding = nn.Embedding(self.local_vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Determine which tokens belong to this rank's shard
        input_mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)

        # Remap to local indices; out-of-shard tokens get index 0 (will be zeroed)
        masked_input = input_ids.clone() - self.vocab_start
        masked_input[input_mask] = 0

        # Lookup embeddings — out-of-shard tokens hit row 0 but will be zeroed
        output = self.embedding(masked_input)

        # Zero out embeddings for tokens not in this shard (in-place, Megatron-LM style).
        # This is safe because F.embedding backward only needs input indices, not output values.
        output[input_mask, :] = 0.0

        # All-reduce across TP group to sum partial embeddings (autograd-compatible)
        output = _AllReduceFunc.apply(output, self.tp_group)
        return output


class ParallelLMHead(nn.Module):
    """
    Language model head split across TP ranks (column-parallel).

    Each rank computes logits for vocab_size // tp_size output classes.
    An all-gather reconstructs the full logits tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tp_group: dist.ProcessGroup,
        tp_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank(tp_group)

        # Pad vocab to be divisible by tp_size
        self.padded_vocab_size = ((vocab_size + tp_size - 1) // tp_size) * tp_size
        self.local_vocab_size = self.padded_vocab_size // tp_size

        self.linear = nn.Linear(hidden_size, self.local_vocab_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul: (B, S, H) @ (local_vocab, H)^T -> (B, S, local_vocab)
        local_logits = self.linear(x)
        # All-gather across TP group to get full logits (autograd-compatible)
        full_logits = _AllGatherFunc.apply(local_logits, self.tp_group, self.tp_size)
        # Trim padding if vocab wasn't evenly divisible
        if self.padded_vocab_size != self.vocab_size:
            full_logits = full_logits[..., :self.vocab_size]
        return full_logits


# ============================================================================
# FSDP2 wrapping
# ============================================================================

def apply_fsdp2(
    model: torch.nn.Module,
    *,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> torch.nn.Module:
    """
    Apply FSDP2 (fully_shard) to a GPTModel.

    Wraps each TransformerLayer individually, then wraps the entire model.
    This provides per-layer sharding for memory efficiency.

    Args:
        model: GPTModel instance (already on CUDA)
        mixed_precision_dtype: Compute dtype for FSDP mixed precision
        dp_group: Data-parallel process group (None = WORLD).
                  When using TP, this should be the DP sub-group.

    Returns:
        FSDP2-wrapped model
    """
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    import transformer_engine.pytorch as te

    mp_policy = MixedPrecisionPolicy(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=torch.float32,
    )

    # Build kwargs for fully_shard
    fsdp_kwargs = dict(mp_policy=mp_policy)
    if dp_group is not None:
        # Create a 2D DeviceMesh (dp, tp) and extract the DP submesh.
        # All ranks must participate in init_device_mesh (it's collective).
        # Layout: rows = DP groups, columns = TP groups
        # e.g., 8 GPUs, TP=2: [[0,1],[2,3],[4,5],[6,7]]
        # TP groups (cols): [0,1], [2,3], [4,5], [6,7]
        # DP groups (rows): [0,2,4,6], [1,3,5,7]
        from torch.distributed.device_mesh import init_device_mesh
        world_size = dist.get_world_size()
        tp_size = world_size // dist.get_world_size(dp_group)
        dp_size = dist.get_world_size(dp_group)
        mesh_2d = init_device_mesh(
            "cuda",
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh = mesh_2d["dp"]
        fsdp_kwargs["mesh"] = dp_mesh

    # Shard each TransformerLayer individually
    for i, layer in enumerate(model.layers):
        fully_shard(layer, **fsdp_kwargs)
        logger.debug(f"  FSDP2 wrapped layer {i}")

    # Shard the entire model (top-level)
    fully_shard(model, **fsdp_kwargs)

    logger.info(f"FSDP2 applied: {len(model.layers)} layers individually sharded")
    return model


def apply_activation_checkpointing(model: torch.nn.Module) -> None:
    """
    Apply activation checkpointing to transformer layers.

    This trades compute for memory by recomputing activations during backward.
    Sets a flag on the model so the forward pass uses TE's own checkpoint
    function, which is compatible with TE's FP8 autograd functions.
    """
    model._use_activation_checkpointing = True
    logger.info(f"Activation checkpointing enabled ({len(model.layers)} layers)")


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
