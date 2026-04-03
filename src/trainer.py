"""
Training loop for GPU benchmarking.

Implements the core training loop with:
  - TransformerEngine autocast for FP8/FP4 precision
  - Proper warmup and measurement phases
  - CUDA-synchronized timing
  - Gradient clipping
  - Cosine LR scheduling
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

import transformer_engine.pytorch as te

from .metrics import BenchmarkMetrics, reset_memory_stats
from .precision import PrecisionMode, create_recipe

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    use_fused: bool = True,
) -> torch.optim.Optimizer:
    """
    Create optimizer with weight-decay separation.

    Uses FusedAdam from TransformerEngine when available for better performance,
    falls back to torch.optim.AdamW otherwise.

    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        use_fused: Whether to try TE FusedAdam first

    Returns:
        Configured optimizer
    """
    # Separate parameters: decay vs no-decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases, norms, or embeddings
        if param.ndim == 1 or "embedding" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if use_fused:
        try:
            from transformer_engine.pytorch.optimizers import FusedAdam
            optimizer = FusedAdam(param_groups, lr=lr, betas=betas, master_weights=True)
            logger.info("Using TE FusedAdam optimizer")
            return optimizer
        except (ImportError, Exception) as e:
            logger.warning(f"FusedAdam not available ({e}), falling back to torch AdamW")

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=True)
    logger.info("Using torch.optim.AdamW (fused=True)")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    warmup_steps: int = 10,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer
        num_steps: Total number of training steps
        warmup_steps: Number of warmup steps with linear ramp
        min_lr_ratio: Minimum LR as fraction of peak LR

    Returns:
        LR scheduler
    """
    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Training loop with precision-aware autocast and metrics collection.

    Handles the complete training lifecycle:
      1. Warmup steps (excluded from measurements)
      2. Measured steps (metrics recorded)
      3. Results aggregation
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        metrics: BenchmarkMetrics,
        precision_mode: PrecisionMode,
        num_steps: int = 100,
        warmup_steps: int = 10,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        use_compile: bool = False,
    ):
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.precision_mode = precision_mode
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.use_compile = use_compile

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(model, lr=lr)
        self.scheduler = create_scheduler(
            self.optimizer, num_steps=num_steps, warmup_steps=warmup_steps
        )

        # Create precision recipe
        self.recipe = create_recipe(precision_mode)
        self.use_autocast = precision_mode != PrecisionMode.BF16

        # amax_reduction_group is only needed for DelayedScaling (global amax allreduce).
        # MXFP8 and NVFP4 use local block scaling — no distributed amax needed.
        self.amax_reduction_group = None
        if (self.use_autocast
                and precision_mode == PrecisionMode.FP8_DELAYED
                and dist.is_initialized()):
            self.amax_reduction_group = dist.group.WORLD

        # Optionally compile the model
        if use_compile:
            logger.info("Applying torch.compile to model...")
            self.model = torch.compile(self.model)

        logger.info(
            f"Trainer initialized: {num_steps} steps "
            f"({warmup_steps} warmup + {num_steps - warmup_steps} measured), "
            f"precision={precision_mode}, compile={use_compile}"
        )

    def train(self) -> None:
        """
        Execute the training loop.

        Runs warmup + measured steps, collecting metrics along the way.
        """
        self.model.train()
        device = next(self.model.parameters()).device

        # Reset memory stats before measurement
        reset_memory_stats()

        # Create infinite data iterator (cycle through dataset)
        data_iter = iter(self._infinite_loader())

        total_steps = self.num_steps

        logger.info(f"Starting training ({total_steps} steps)...")

        for step in range(total_steps):
            # Get batch
            input_ids, targets = next(data_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_tokens = input_ids.numel()

            # --- Forward + Backward + Optimizer Step ---
            self.metrics.start_step(step)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional FP8/FP4 autocast
            if self.use_autocast:
                with te.autocast(
                    enabled=True,
                    recipe=self.recipe,
                    amax_reduction_group=self.amax_reduction_group,
                ):
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                    )
            else:
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )

            # Backward
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Record metrics
            self.metrics.end_step(step, loss.item(), batch_tokens)

        logger.info("Training complete.")

    def _infinite_loader(self):
        """Yield batches infinitely by cycling through the dataloader."""
        while True:
            for batch in self.dataloader:
                yield batch
