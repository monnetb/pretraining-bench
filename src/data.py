"""
Dataset and DataLoader factories for GPU benchmarking.

Provides two dataset modes:
  - Synthetic: Random token IDs (zero I/O overhead, pure compute benchmark)
  - Tiny-text: Small text corpus for loss-curve validation
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)

DEFAULT_VOCAB_SIZE = 50257  # GPT-2 tokenizer vocabulary


class SyntheticDataset(Dataset):
    """
    Pre-generated random token dataset for pure GPU benchmarking.

    Each sample is a contiguous block of random token IDs.
    Input = tokens[:-1], target = tokens[1:] (standard causal LM).
    Data is generated once and cached in CPU memory.
    """

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        logger.info(
            f"Generating synthetic dataset: {num_samples} samples, "
            f"seq_length={seq_length}, vocab_size={vocab_size}"
        )
        gen = torch.Generator()
        gen.manual_seed(seed)
        # Generate seq_length + 1 tokens so we can create (input, target) pairs
        self.data = torch.randint(
            0, vocab_size, (num_samples, seq_length + 1), generator=gen
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


class TinyTextDataset(Dataset):
    """
    Small text dataset for loss validation.

    Reads a text file, converts to character-level token IDs, and creates
    fixed-length chunks. Uses a simple byte-level encoding (ord(c) % vocab_size)
    so no external tokenizer is needed.
    """

    def __init__(
        self,
        text_path: str,
        seq_length: int,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        logger.info(f"Loading text dataset from: {text_path}")
        with open(text_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Simple byte-level encoding: map each character to a token ID
        self.tokens = torch.tensor(
            [ord(c) % vocab_size for c in text], dtype=torch.long
        )

        # Number of complete (input, target) chunks we can extract
        self.num_chunks = (len(self.tokens) - 1) // seq_length
        if self.num_chunks == 0:
            raise ValueError(
                f"Text file too short ({len(self.tokens)} chars) "
                f"for seq_length={seq_length}"
            )

        logger.info(
            f"Text dataset: {len(text)} chars → {self.num_chunks} chunks "
            f"of {seq_length} tokens"
        )

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


def get_dataloader(
    dataset_type: str = "synthetic",
    seq_length: int = 2048,
    batch_size: int = 8,
    num_samples: int = 10000,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    text_path: Optional[str] = None,
    num_workers: int = 2,
    distributed: bool = False,
    seed: int = 42,
    dp_rank: Optional[int] = None,
    dp_size: Optional[int] = None,
) -> DataLoader:
    """
    Factory function to create the appropriate DataLoader.

    Args:
        dataset_type: "synthetic" or "tiny-text"
        seq_length: Sequence length per sample
        batch_size: Per-GPU batch size
        num_samples: Number of samples for synthetic dataset
        vocab_size: Vocabulary size
        text_path: Path to text file (required for tiny-text)
        num_workers: DataLoader workers
        distributed: Whether to use DistributedSampler
        seed: Random seed for synthetic data
        dp_rank: Data-parallel rank (for TP+DP; ensures TP ranks in the
                 same group see the same data)
        dp_size: Data-parallel world size (for TP+DP)

    Returns:
        Configured DataLoader
    """
    if dataset_type == "synthetic":
        dataset = SyntheticDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size,
            seed=seed,
        )
    elif dataset_type == "tiny-text":
        if text_path is None:
            raise ValueError("--text-path required for tiny-text dataset")
        dataset = TinyTextDataset(
            text_path=text_path,
            seq_length=seq_length,
            vocab_size=vocab_size,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    sampler = None
    shuffle = True
    if distributed:
        # When using TP+DP, the sampler must shard data across DP groups only
        # (not the full world), so that all TP ranks within a group see the same data.
        sampler_kwargs = dict(shuffle=True, seed=seed)
        if dp_rank is not None and dp_size is not None:
            sampler_kwargs["num_replicas"] = dp_size
            sampler_kwargs["rank"] = dp_rank
        sampler = DistributedSampler(dataset, **sampler_kwargs)
        shuffle = False  # DistributedSampler handles shuffling

    # Use a seeded generator for deterministic shuffle order.
    # This is critical for tensor parallelism: all TP ranks must see
    # the same data in the same order, so the DataLoader shuffle must
    # be identical across processes in the same TP group.
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )

    logger.info(
        f"DataLoader: {dataset_type}, batch_size={batch_size}, "
        f"samples={len(dataset)}, batches={len(loader)}, "
        f"distributed={distributed}"
    )

    return loader
