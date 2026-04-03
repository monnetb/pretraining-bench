"""
GPT-style transformer model built from TransformerEngine layers.

Supports two architecture families:
  - GPT-2: LayerNorm, GELU, full multi-head attention, learned positional embeddings
  - LLaMA: RMSNorm, SwiGLU, grouped-query attention, rotary positional embeddings
             (rotary embeddings are handled internally by TransformerEngine)

The model is a standard decoder-only causal language model:
  Token Embedding → [Position Embedding] → N × TransformerLayer → Final Norm → LM Head
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a GPT model."""
    name: str
    arch_style: str            # "gpt2" or "llama"
    hidden_size: int
    num_attention_heads: int
    num_layers: int
    ffn_hidden_size: int
    vocab_size: int = 50257
    max_seq_length: int = 2048
    num_gqa_groups: Optional[int] = None  # None = full MHA
    normalization: str = "LayerNorm"       # "LayerNorm" or "RMSNorm"
    activation: str = "gelu"               # "gelu" or "swiglu"
    bias: bool = True
    dropout: float = 0.0                   # 0 for benchmarking

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_params_approx(self) -> int:
        """Approximate parameter count (millions)."""
        # Embedding
        emb = self.vocab_size * self.hidden_size
        # Per-layer: attention (QKV + out proj) + FFN + norms
        if self.activation == "swiglu":
            # SwiGLU has 3 weight matrices in FFN (gate, up, down)
            ffn = 3 * self.hidden_size * self.ffn_hidden_size
        else:
            ffn = 2 * self.hidden_size * self.ffn_hidden_size
        attn = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
        if self.num_gqa_groups is not None:
            # GQA: K,V are smaller
            kv_size = self.num_gqa_groups * self.head_dim
            attn = (self.hidden_size * self.hidden_size  # Q
                    + 2 * self.hidden_size * kv_size       # K, V
                    + self.hidden_size * self.hidden_size)  # O
        per_layer = attn + ffn
        total = emb + self.num_layers * per_layer + emb  # +emb for LM head (tied)
        return total


def _count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops_per_token(config: ModelConfig) -> int:
    """
    Compute the number of floating-point operations per token for a training step.

    Formula (from "Scaling Laws" and Megatron-LM):
      Forward pass per token ≈ 2 * P + 2 * L * S * H
        where P = num parameters (excluding embeddings for pure transformer flops)
              L = num layers
              S = sequence length
              H = hidden size
        The 2*L*S*H term accounts for the attention score computation (QK^T and attn@V).

      Training step = 3 × forward pass (forward + backward ≈ 3× forward)

    This gives a rough but standard estimate used industry-wide for MFU calculation.
    """
    # Parameter-based flops (matrix multiplications in transformer layers)
    P = config.num_layers * (
        # Attention: QKV projection + output projection
        4 * config.hidden_size * config.hidden_size
        # FFN
        + (3 if config.activation == "swiglu" else 2) * config.hidden_size * config.ffn_hidden_size
    )

    # Attention score computation: QK^T and softmax(QK^T)V
    attn_flops = 2 * config.num_layers * config.max_seq_length * config.hidden_size

    # Forward pass flops per token
    forward_flops = 2 * P + attn_flops

    # Training = forward + backward ≈ 3 × forward
    training_flops = 3 * forward_flops

    return training_flops


class GPTModel(nn.Module):
    """
    Decoder-only transformer language model using TransformerEngine layers.

    Architecture:
      - Token embedding (+ learned position embedding for GPT-2)
      - Stack of te.TransformerLayer blocks
      - Final normalization
      - Linear language model head (weight-tied with token embedding)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.use_learned_pos_emb = config.arch_style == "gpt2"
        if self.use_learned_pos_emb:
            self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # --- Transformer Layers ---
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_kwargs = dict(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.ffn_hidden_size,
                num_attention_heads=config.num_attention_heads,
                self_attn_mask_type="causal",
                fuse_qkv_params=True,
                normalization=config.normalization,
                activation=config.activation,
                bias=config.bias,
                params_dtype=torch.bfloat16,
                seq_length=config.max_seq_length,
                layer_number=i + 1,
                hidden_dropout=config.dropout,
                attention_dropout=config.dropout,
                attn_input_format="bshd",  # use (batch, seq, hidden) to avoid transposes
            )
            # GQA support
            if config.num_gqa_groups is not None:
                layer_kwargs["num_gqa_groups"] = config.num_gqa_groups

            self.layers.append(te.TransformerLayer(**layer_kwargs))

        # --- Final Norm ---
        if config.normalization == "RMSNorm":
            self.final_norm = te.RMSNorm(config.hidden_size)
        else:
            self.final_norm = te.LayerNorm(config.hidden_size)

        # --- LM Head (tied with token embedding) ---
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        num_params = _count_parameters(self)
        logger.info(
            f"Model '{config.name}': {num_params:,} parameters "
            f"({num_params / 1e6:.1f}M)"
        )

    def _init_weights(self):
        """Initialize embeddings with standard normal scaled by hidden_size."""
        std = 0.02
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        if self.use_learned_pos_emb:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_length) token IDs
            attention_mask: optional attention mask (not typically needed for causal)

        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        B, S = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # (B, S, H)

        # Position embeddings (GPT-2 style)
        if self.use_learned_pos_emb:
            positions = torch.arange(0, S, dtype=torch.long, device=input_ids.device)
            x = x + self.position_embedding(positions).unsqueeze(0)

        x = self.embedding_dropout(x)

        # Cast to BF16 for transformer layers
        x = x.to(dtype=torch.bfloat16)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Final norm
        x = self.final_norm(x)

        # LM head
        logits = self.lm_head(x.float())  # cast back to FP32 for loss computation

        return logits


def build_model(config: ModelConfig, device: torch.device = torch.device("cuda")) -> GPTModel:
    """
    Build and return a GPTModel on the specified device.

    Args:
        config: Model configuration
        device: Target device

    Returns:
        GPTModel instance on device
    """
    model = GPTModel(config)
    model = model.to(device)
    return model


def load_model_config(
    configs_dict: dict,
    model_name: str,
    vocab_size: int = 50257,
) -> ModelConfig:
    """
    Load a model config from the parsed YAML dictionary.

    Args:
        configs_dict: Parsed YAML config (the 'models' sub-dict)
        model_name: Key in the models dict (e.g., "small-gpt2")
        vocab_size: Override vocabulary size

    Returns:
        ModelConfig dataclass
    """
    cfg = configs_dict[model_name]
    return ModelConfig(
        name=model_name,
        arch_style=cfg["arch_style"],
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_layers=cfg["num_layers"],
        ffn_hidden_size=cfg["ffn_hidden_size"],
        vocab_size=vocab_size,
        max_seq_length=cfg.get("max_seq_length", 2048),
        num_gqa_groups=cfg.get("num_gqa_groups"),
        normalization=cfg.get("normalization", "LayerNorm"),
        activation=cfg.get("activation", "gelu"),
        bias=cfg.get("bias", True),
    )
