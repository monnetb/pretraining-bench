# Transformer Pretraining GPU Benchmark

A comprehensive GPU benchmark for transformer pretraining from scratch using
[NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine).

Designed to compare performance across GPU architectures (B300, B200, H200, H100, etc.)
with multiple precision modes and model configurations.

## Features

- **Multi-architecture models**: GPT-2 style (LayerNorm/GELU/MHA) and LLaMA style (RMSNorm/SwiGLU/GQA)
- **Multiple model sizes**: 125M, 350M, 760M, 1.3B parameters
- **Auto-detected precision modes**: BF16, FP8 (Delayed/Current/Block scaling), MXFP8, NVFP4
- **Performance metrics**: Tokens/s, TFLOPS, MFU, peak memory, step time statistics
- **Multi-GPU support**: FSDP2 with torchrun
- **Sweep mode**: Automatically benchmarks all model × precision combinations
- **Structured output**: JSON and CSV results for cross-GPU comparison

## Quick Start

```bash
# Single GPU, small model, BF16 baseline
./scripts/run_single_gpu.sh

# Single GPU, auto-detect best precision
./scripts/run_single_gpu.sh small-llama auto

# Full sweep — all models × all available precisions
./scripts/run_full_sweep.sh

# Multi-GPU with FSDP2 (8 GPUs)
./scripts/run_multi_gpu.sh 8 large-llama auto
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

## Precision Modes

| Mode | GPU Requirement | Description |
|------|----------------|-------------|
| `bf16` | Any CUDA GPU | BF16 baseline — always available |
| `fp8-delayed` | Hopper+ (SM90) | FP8 with delayed scaling (amax history) |
| `fp8-current` | Hopper+ (SM90) | FP8 with current-tensor scaling |
| `fp8-block` | Hopper+ (SM90) | FP8 with per-block scaling |
| `mxfp8` | Blackwell (SM100) | Microscaling FP8 (MX format) |
| `nvfp4` | Blackwell (SM100) | FP4 with NVIDIA block scaling |

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
│   ├── distributed.py          # FSDP2 setup
│   └── report.py               # Console table, JSON, CSV
├── scripts/
│   ├── run_single_gpu.sh       # Single GPU launcher
│   ├── run_multi_gpu.sh        # Multi-GPU (torchrun) launcher
│   └── run_full_sweep.sh       # Full matrix sweep
└── results/                    # Output directory
```
