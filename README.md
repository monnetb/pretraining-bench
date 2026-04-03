# Transformer Pretraining GPU Benchmark

A comprehensive GPU benchmark for transformer pretraining from scratch using
[NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine).

Designed to compare performance across GPU architectures (B300, B200, H200, H100, etc.)
with multiple precision modes and model configurations.

## Features

- **Multi-architecture models**: GPT-2 style (LayerNorm/GELU/MHA) and LLaMA style (RMSNorm/SwiGLU/GQA)
- **Multiple model sizes**: 125M, 350M, 760M, 1.3B, 8B, 30B, 70B parameters
- **Auto-detected precision modes**: BF16, FP8 (Delayed/Current/Block scaling), MXFP8, NVFP4
- **Performance metrics**: Tokens/s, TFLOPS, MFU, peak memory, step time statistics
- **Multi-GPU support**: FSDP2 with torchrun, tensor parallelism (TP) for large models, hybrid TP×DP 2D parallelism
- **Sweep mode**: Automatically benchmarks all model × precision combinations
- **Structured output**: JSON and CSV results for cross-GPU comparison

## Quick Start

```bash
# 1. Create the virtual environment and install dependencies
./scripts/setup_venv.sh          # or: ./scripts/setup_venv.sh --dev  (includes ruff, pytest)

# 2. Run benchmarks (scripts auto-activate the venv)

# Single GPU, small model, BF16 baseline
./scripts/run_single_gpu.sh

# Single GPU, auto-detect best precision
./scripts/run_single_gpu.sh small-llama auto

# Full sweep — all models × all available precisions
./scripts/run_full_sweep.sh

# Multi-GPU with FSDP2 (8 GPUs)
./scripts/run_multi_gpu.sh 8 large-llama auto

# 70B model with tensor parallelism (TP=8)
./scripts/run_multi_gpu.sh 8 70b-llama auto 1 2048 20 5 8

# 8B model with 2D parallelism (TP=2, DP=4)
./scripts/run_multi_gpu.sh 8 8b-llama auto 8 2048 100 10 2
```

## Direct Python Usage

```bash
# Single config
python -m src.benchmark --model-size small-gpt2 --precision bf16

# All precisions for one model
python -m src.benchmark --model-size medium-llama --precision all

# Full sweep
python -m src.benchmark --sweep

# Multi-GPU
torchrun --nproc_per_node=8 -m src.benchmark --model-size large-llama --precision auto

# 70B with tensor parallelism (TP=8, no FSDP)
torchrun --nproc_per_node=8 -m src.benchmark --model-size 70b-llama --precision bf16 --tp-size 8

# 2D parallelism: TP=4 × DP=2
torchrun --nproc_per_node=8 -m src.benchmark --model-size 70b-llama --precision bf16 --tp-size 4
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-size` | `small-gpt2` | Model name(s), comma-separated, or `all` |
| `--precision` | `bf16` | `bf16`, `fp8-delayed`, `fp8-current`, `fp8-block`, `mxfp8`, `nvfp4`, `auto`, `all` |
| `--batch-size` | `8` | Per-GPU micro batch size |
| `--seq-length` | `2048` | Sequence length |
| `--num-steps` | `100` | Total training steps |
| `--warmup-steps` | `10` | Warmup steps (excluded from metrics) |
| `--dataset` | `synthetic` | `synthetic` or `tiny-text` |
| `--text-path` | - | Text file path for `tiny-text` dataset |
| `--activation-checkpointing` | off | Enable gradient checkpointing |
| `--use-compile` | off | Apply `torch.compile` |
| `--tp-size` | `1` | Tensor parallel size (must divide world_size; dp_size = world_size / tp_size) |
| `--sweep` | off | Run all model × precision combinations |
| `--output-dir` | `results` | Results output directory |

## Model Configurations

| Name | Params | Hidden | Heads | Layers | FFN | Style |
|------|--------|--------|-------|--------|-----|-------|
| `small-gpt2` | ~125M | 768 | 12 | 12 | 3072 | GPT-2 |
| `small-llama` | ~125M | 768 | 12 | 12 | 3072 | LLaMA |
| `medium-gpt2` | ~350M | 1024 | 16 | 24 | 4096 | GPT-2 |
| `medium-llama` | ~350M | 1024 | 16 | 24 | 4096 | LLaMA |
| `large-gpt2` | ~760M | 1536 | 16 | 24 | 6144 | GPT-2 |
| `large-llama` | ~760M | 1536 | 16 | 24 | 6144 | LLaMA |
| `xlarge-gpt2` | ~1.3B | 2048 | 32 | 24 | 8192 | GPT-2 |
| `xlarge-llama` | ~1.3B | 2048 | 32 | 24 | 8192 | LLaMA |
| `8b-llama` | ~8B | 4096 | 32 | 32 | 14336 | LLaMA 3 |
| `30b-llama` | ~30B | 6144 | 48 | 60 | 21504 | LLaMA |
| `70b-llama` | ~70B | 8192 | 64 | 80 | 28672 | LLaMA 3 |

## Precision Modes

| Mode | GPU Requirement | Description |
|------|----------------|-------------|
| `bf16` | Any CUDA GPU | BF16 baseline — always available |
| `fp8-delayed` | Hopper+ (SM90) | FP8 with delayed scaling (amax history) |
| `fp8-current` | Hopper+ (SM90) | FP8 with current-tensor scaling |
| `fp8-block` | Hopper+ (SM90) | FP8 with per-block scaling |
| `mxfp8` | Blackwell (SM100) | Microscaling FP8 (MX format) |
| `nvfp4` | Blackwell (SM100) | FP4 with NVIDIA block scaling |

## Tensor Parallelism

Large models (8B, 30B, 70B) require tensor parallelism (TP) to fit across
multiple GPUs. TP shards the model weights (attention QKV/output projections,
FFN, embeddings, and LM head) across GPUs within a TP group, while FSDP2
handles data parallelism (DP) across TP groups.

| Layout | `--tp-size` | DP size | Example |
|--------|-------------|---------|---------|
| Pure FSDP | 1 (default) | world_size | `torchrun --nproc_per_node=8 -m src.benchmark --model-size xlarge-llama` |
| Pure TP | world_size | 1 | `torchrun --nproc_per_node=8 -m src.benchmark --model-size 70b-llama --tp-size 8` |
| 2D (TP×DP) | 2–N | world_size/tp_size | `torchrun --nproc_per_node=8 -m src.benchmark --model-size 8b-llama --tp-size 2` |

With 8 GPUs and `--tp-size 4`:
- **TP groups** (share model weights): [0,1,2,3], [4,5,6,7]
- **DP groups** (share data batches): [0,4], [1,5], [2,6], [3,7]

**Implementation details:**
- `VocabParallelEmbedding`: shards the embedding table along the vocabulary dimension across TP ranks; uses all-reduce to combine partial lookups
- `ParallelLMHead`: column-parallel linear layer with all-gather to reconstruct full logits
- TransformerEngine layers use native `tp_group`/`tp_size`/`set_parallel_mode` for column- and row-parallel sharding of attention and FFN weights
- FSDP2 uses a `DeviceMesh` restricted to the DP sub-group so it only shards across data-parallel ranks
- `DataLoader` seeding is deterministic per TP group to ensure all TP ranks see the same data

**Verified configurations (8×H200):**
- 70B with TP=8: fits in ~131 GB/GPU
- 8B with TP=2 + DP=4: correct 2D parallelism
- All existing FSDP-only configs (≤1.3B) remain unaffected

## MFU Calculation

Model FLOPS Utilization measures how much of the GPU's theoretical peak compute
is being used for useful model computation:

```
Forward FLOPS/token = 2P + 2LSH
  where P = parameter flops (all matmul weights in transformer layers)
        L = num layers, S = seq length, H = hidden size

Training FLOPS/token = 3 × Forward FLOPS/token

MFU = (training_flops_per_token × tokens_per_second) / GPU_peak_FLOPS
```

## Output

Results are saved to the `results/` directory:
- **JSON**: Full results with all metadata
- **CSV**: One row per benchmark run, spreadsheet-friendly

Example console output:
```
================================================================================
  Benchmark Summary
================================================================================
  GPU: NVIDIA H100 80GB HBM3 × 1
────────────────────────────────────────────────────────────────────────────────
       Model    Precision      Tok/s  TFLOPS    MFU  Mem (GB)     Step (ms)   Loss
  small-gpt2         bf16    380,000   102.3  5.2%      12.4    43.1±1.2    8.234
  small-gpt2   fp8-delayed   520,000   139.8  3.5%      10.2    31.5±0.8    8.241
================================================================================
```

## Project Structure

```
transformer-bench/
├── pyproject.toml              # Dependencies
├── configs/
│   └── model_configs.yaml      # Model size presets
├── src/
│   ├── __init__.py
│   ├── benchmark.py            # Main entry point + CLI
│   ├── model.py                # GPT model from TE layers
│   ├── data.py                 # Synthetic + text datasets
│   ├── trainer.py              # Training loop
│   ├── metrics.py              # Throughput, MFU, memory
│   ├── precision.py            # GPU detection + recipes
│   ├── distributed.py          # FSDP2 + tensor parallelism setup
│   └── report.py               # Console table, JSON, CSV
├── scripts/
│   ├── setup_venv.sh              # Create venv and install deps
│   ├── run_single_gpu.sh       # Single GPU launcher
│   ├── run_multi_gpu.sh        # Multi-GPU (torchrun) launcher
│   └── run_full_sweep.sh       # Full matrix sweep
└── results/                    # Output directory
```
