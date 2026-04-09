# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transformer pretraining GPU benchmark using NVIDIA TransformerEngine. Compares GPU performance (B300, B200, H200, H100, etc.) across precision modes (BF16, FP8, MXFP8, NVFP4) and model sizes (125M to 70B). Not a training framework -- purely a benchmarking tool that measures tokens/s, TFLOPS, MFU, and peak memory.

## Common Commands

### Setup
```bash
./scripts/setup_venv.sh          # create .venv, install deps
./scripts/setup_venv.sh --dev    # includes ruff and pytest
```

### Running Benchmarks
```bash
# Single GPU
python -m src.benchmark --model-size small-gpt2 --precision bf16
python -m src.benchmark --model-size small-llama --precision auto

# Multi-GPU with FSDP2
torchrun --nproc_per_node=8 -m src.benchmark --model-size large-llama --precision auto

# Tensor parallelism for large models (70B needs TP)
torchrun --nproc_per_node=8 -m src.benchmark --model-size 70b-llama --tp-size 8

# Full sweep (all models x all precisions)
python -m src.benchmark --sweep
```

### Shell Script Launchers
```bash
./scripts/run_single_gpu.sh [model] [precision] [batch] [seq_len] [steps] [warmup]
./scripts/run_multi_gpu.sh [ngpus] [model] [precision] [batch] [seq_len] [steps] [warmup] [tp_size]
./scripts/run_full_sweep.sh [steps] [size_filter] [batch] [seq_len] [warmup]
```

### Linting
```bash
ruff check src/     # requires dev install
```

## Architecture

The project is a Python package (`src/`) run as `python -m src.benchmark`. Entry point is `src/benchmark.py:main()`.

### Data Flow
`benchmark.py` (CLI + orchestration) -> builds model via `model.py` -> wraps with FSDP2/TP via `distributed.py` -> creates dataloader via `data.py` -> runs training loop via `trainer.py` (which uses `precision.py` for TE autocast recipes) -> collects metrics via `metrics.py` -> outputs results via `report.py`

### Key Modules

- **`model.py`**: `GPTModel` class -- decoder-only transformer built from `te.TransformerLayer` blocks. Supports two arch families: GPT-2 (LayerNorm/GELU/MHA) and LLaMA (RMSNorm/SwiGLU/GQA). `ModelConfig` dataclass holds all hyperparameters. TP support uses `VocabParallelEmbedding` and `ParallelLMHead` from `distributed.py`.

- **`precision.py`**: GPU capability detection (`detect_gpu_capabilities`) and TE recipe creation (`create_recipe`). `PrecisionMode` enum maps CLI strings to TE recipe types. `GPUCapabilities` probes TE for FP8/MXFP8/NVFP4 support. `GPU_SPECS` dict holds theoretical peak TFLOPS for MFU calculation.

- **`distributed.py`**: `setup_parallel_groups()` creates TP and DP process groups (contiguous ranks for TP, strided for DP). `apply_fsdp2()` wraps each `TransformerLayer` individually then the whole model. `VocabParallelEmbedding` shards embedding along vocab dim with all-reduce; `ParallelLMHead` uses column-parallel linear with all-gather.

- **`trainer.py`**: `Trainer` class runs the training loop with `te.autocast` for non-BF16 precision. Uses TE `FusedAdam` when available. Cosine LR schedule with linear warmup. `amax_reduction_group` is only set for `FP8_DELAYED` mode.

- **`metrics.py`**: `BenchmarkMetrics` records per-step CUDA-event-timed durations. `get_result()` computes aggregate throughput, TFLOPS, and MFU. MFU is computed as achieved TFLOPS / (peak_tflops * num_gpus). Throughput accounts for DP size (not TP, since TP ranks process the same tokens).

- **`data.py`**: `SyntheticDataset` (random tokens, zero I/O overhead) and `TinyTextDataset` (byte-level encoding from text file). `DistributedSampler` shards by DP rank when using TP+DP to ensure TP ranks see identical data.

- **`report.py`**: Console table (via `tabulate`), JSON, and CSV export to `results/`.

### Model Configs
Defined in `configs/model_configs.yaml`. Sizes: small/medium/large/xlarge (GPT-2 and LLaMA variants) plus 8B, 30B, 70B LLaMA. Vocab size is fixed at 50257.

### Parallelism
- **Single GPU**: default, no distributed init
- **FSDP2 (pure DP)**: `--tp-size 1` (default), wraps each layer then full model
- **Pure TP**: `--tp-size` = world_size, shards weights across all GPUs
- **2D (TP x DP)**: `--tp-size` < world_size, e.g. TP=2 DP=4 on 8 GPUs

### Environment Notes
- Scripts auto-activate `.venv` and add pip-installed NVIDIA libs to `LD_LIBRARY_PATH`
- `NVTE_FUSED_ATTN=0` is set by scripts to work around cuDNN sublibrary loading issues
- Precision modes are auto-detected from GPU compute capability (SM90 for Hopper FP8, SM100 for Blackwell MXFP8/NVFP4)
- Results (JSON/CSV) go to `results/` and are gitignored
