# Transformer-Bench: GPU Pretraining Benchmark Report

**Date:** April 9, 2026
**Project:** transformer-bench — GPU pretraining benchmark using NVIDIA TransformerEngine
**Objective:** Measure and compare single-GPU training throughput (tokens/s, TFLOPS, MFU) across five GPU platforms spanning two architecture generations (Hopper SM90, Blackwell SM100/SM103).

---

## 1. Executive Summary

Five GPU platforms were benchmarked with a decoder-only LLaMA-architecture transformer model in BF16 precision. All Hopper (SM90) and standard Blackwell (SM100) platforms achieved **>50% Model FLOPS Utilization (MFU)**, with the GH200 reaching **60.2%**. The NVIDIA B300 SXM6 AC (SM103) encountered a TransformerEngine kernel compatibility issue requiring a pure-PyTorch fallback, achieving **44.3% MFU** and **997 TFLOPS** — nearly 1 PFLOP/s from a single GPU.

| Rank | GPU | TFLOPS | MFU | Status |
|------|-----|--------|-----|--------|
| 1 | GB300 (Sapling) | 1,230.6 | 54.7% | TE optimized |
| 2 | B300 SXM6 AC (Vader) | 996.6 | 44.3% | Pure PyTorch fallback |
| 3 | GH200 (Grenoble) | 595.0 | 60.2% | TE optimized |
| 4 | H200 (Vader) | 558.5 | 56.5% | TE optimized |
| 5 | H100 80GB (Grenoble) | 495.0 | 50.1% | TE optimized |

---

## 2. Test Infrastructure

### 2.1 Clusters and GPUs

| Cluster | Partition / Node | Architecture | GPU | CC | GPU Memory | Interconnect |
|---------|-----------------|--------------|-----|-----|-----------|-------------|
| Grenoble | jakku (8×GPU/node) | x86_64 | NVIDIA H100 80GB HBM3 | SM90 (9.0) | 80 GB HBM3 | NVLink |
| Grenoble | aarch64 (1×GPU/node) | aarch64 | NVIDIA GH200 144G HBM3e | SM90 (9.0) | 144 GB HBM3e | NVLink C2C |
| Vader | H200rhel94 / sith nodes | x86_64 | NVIDIA H200 | SM90 (9.0) | 141 GB HBM3e | NVLink |
| Vader | B300rhel94 / leia nodes | x86_64 | NVIDIA B300 SXM6 AC | SM103 (10.3) | 275 GB HBM3e | NVLink |
| Sapling | gpu-001–018 (4×GPU/node) | aarch64 | NVIDIA GB300 | SM100 (10.0) | 288 GB HBM3e | NVLink |

### 2.2 Software Stack

| Component | Grenoble (H100) | Grenoble (GH200) | Vader (H200) | Vader (B300) | Sapling (GB300) |
|-----------|----------------|-------------------|--------------|--------------|-----------------|
| Python | 3.12 | 3.12 | 3.12 | 3.12 | 3.12 |
| PyTorch | 2.11.0+cu130 | 2.11.0 (cu13) | 2.11.0+cu126 | 2.11.0+cu130 | 2.11.0 (cu13) |
| TransformerEngine | 2.13.0 cu13 | 2.13.0 cu13 | 2.13.0 cu12 | 2.13.0 cu13 | 2.13.0 cu13 |
| CUDA Runtime | 13.0 | 13.0 | 12.6 | 13.0 | 13.0 |
| CUDA Driver | — | — | 570.x | 595.58 (CUDA 13.2) | — |
| nvidia-cublas | 13.3.0.5 | 13.3.0.5 | 12.6.4.1 (cu12) | 13.3.0.5 | 13.3.0.5 |
| Model Backend | TE layers | TE layers | TE layers (cu12) | **Pure PyTorch** | TE layers |

**Key version note:** The H200 (SM90) uses the `cu12` TE wheel because the `cu13` TE pip wheels contain SASS compiled only for SM100+, causing kernel crashes on SM90 Hopper GPUs. The `cu12` wheel supports both SM90 and SM100. This required a separate venv (`.venv-h200-cu12`) with `torch cu126` for ABI compatibility.

### 2.3 NCCL Configuration (aarch64 platforms)

For aarch64 systems (GH200, GB300) which use NVLink instead of InfiniBand:
```bash
NCCL_IB_DISABLE=1
NCCL_MNNVL_ENABLE=1
```

---

## 3. Benchmark Configuration

### 3.1 Common Parameters

| Parameter | Value |
|-----------|-------|
| Precision | BF16 |
| Sequence Length | 2048 |
| Batch Size | 6 (per GPU) |
| Training Steps | 50 (10 warmup + 40 measured) |
| Dataset | Synthetic (random tokens, zero I/O overhead) |
| Optimizer | FusedAdam (TE) or AdamW fused (PyTorch) |
| Learning Rate | 3e-4, cosine schedule with linear warmup |
| Gradient Clipping | max_norm = 1.0 |
| Loss Chunking | 512 tokens (avoids materializing full logit tensor) |
| Fused Attention | Enabled (except B300 which uses `--no-te`) |
| GPUs | 1 (single-GPU benchmark) |

### 3.2 Model Configurations

| Platform | Model | Parameters | Hidden | Heads | Layers | FFN | GQA Groups |
|----------|-------|-----------|--------|-------|--------|-----|-----------|
| H100 | 3b-llama | 3.79B | 3072 | 24 | 28 | 10752 | 8 |
| GH200, H200, GB300, B300 | 8b-llama | 7.39B | 4096 | 32 | 32 | 14336 | 8 |

The H100 (80 GB) uses the smaller 3b-llama model because the 8b-llama with batch=6 exceeds its memory capacity. All other GPUs (141–288 GB) run the 8b-llama model.

### 3.3 Theoretical Peak Performance (BF16 Dense)

| GPU | Peak BF16 TFLOPS | Source |
|-----|-----------------|--------|
| H100 80GB SXM | 989 | NVIDIA spec sheet |
| GH200 144G HBM3e | 989 | Same H100 compute die |
| H200 SXM | 989 | Same H100 compute die, more memory |
| B300 SXM6 AC | 2,250 | NVIDIA Blackwell spec |
| GB300 NVL72 | 2,250 | NVIDIA Blackwell spec |

### 3.4 MFU Calculation

```
MFU = achieved_tflops / (peak_bf16_tflops × num_gpus)

where:
  achieved_tflops = (flops_per_token × tokens_per_second) / 1e12
  flops_per_token = 3 × (2P + 2LSH)    # forward + backward ≈ 3× forward
  P = per-layer params (attn + FFN)
  L = num_layers, S = seq_length, H = hidden_size
```

---

## 4. Results

### 4.1 Summary Table

| Platform | GPU | Model | Tok/s | TFLOPS | MFU | Peak Mem (GB) | Step Time (ms) |
|----------|-----|-------|------:|-------:|----:|-------------:|--------------:|
| Sapling | GB300 288GB | 8b-llama | 25,469 | 1,230.6 | **54.7%** | 122.6 | 482.5 |
| Vader | B300 SXM6 AC 275GB | 8b-llama | 20,625 | 996.6 | **44.3%** | 111.4 | 595.8 |
| Grenoble | GH200 144GB HBM3e | 8b-llama | 12,315 | 595.0 | **60.2%** | 128.0 | 997.8 |
| Vader | H200 141GB | 8b-llama | 11,560 | 558.5 | **56.5%** | 122.6 | 1,063.0 |
| Grenoble | H100 80GB HBM3 | 3b-llama | 20,587 | 495.0 | **50.1%** | 71.9 | 596.9 |

### 4.2 Performance Analysis

**Absolute throughput ranking** (TFLOPS):
1. **GB300**: 1,230.6 TFLOPS — 2.5× faster than H100, highest absolute performance
2. **B300**: 996.6 TFLOPS — still nearly 1 PFLOP/s despite running without TE optimizations
3. **GH200**: 595.0 TFLOPS — highest MFU (60.2%), excellent efficiency
4. **H200**: 558.5 TFLOPS — solid Hopper performance, 13% faster than H100
5. **H100**: 495.0 TFLOPS — baseline Hopper performance

**Efficiency ranking** (MFU):
1. **GH200**: 60.2% — best MFU, likely benefits from unified CPU-GPU memory architecture
2. **H200**: 56.5% — excellent for Hopper
3. **GB300**: 54.7% — strong Blackwell efficiency with TE fused kernels
4. **H100**: 50.1% — meets the >50% target with smaller model
5. **B300**: 44.3% — limited by pure-PyTorch fallback (no TE fused kernels)

### 4.3 Blackwell Generation Scaling

Comparing the two Blackwell GPUs against Hopper:

| Metric | H200 (Hopper) | GB300 (Blackwell) | Speedup |
|--------|--------------|-------------------|---------|
| TFLOPS | 558.5 | 1,230.6 | **2.20×** |
| Tokens/s | 11,560 | 25,469 | **2.20×** |
| Peak BF16 | 989 | 2,250 | 2.28× |
| Actual/Peak | 56.5% | 54.7% | ~parity |

Blackwell delivers **2.2× the throughput** of Hopper in BF16 training, closely tracking the 2.28× theoretical peak ratio.

---

## 5. Optimization Techniques Applied

The following optimizations were implemented and validated during the benchmarking campaign:

### 5.1 Fused Attention (NVTE_FUSED_ATTN=1)
- **Impact:** +15–40% throughput, -20–40% peak memory
- **Mechanism:** TransformerEngine's fused attention combines QKV projection, attention score computation, and softmax into a single kernel via cuDNN
- **Applied to:** All platforms except B300 (uses PyTorch's `F.scaled_dot_product_attention` instead)

### 5.2 Chunked Cross-Entropy Loss (--loss-chunk-size 512)
- **Impact:** Saves ~1–2.7 GB per batch element
- **Mechanism:** Computes cross-entropy in chunks along the sequence dimension, avoiding materializing the full `(B×S, vocab_size)` logit tensor
- **Applied to:** All platforms

### 5.3 FusedAdam without Master Weights
- **Impact:** Saves ~14–28 GB (no FP32 parameter copy)
- **Mechanism:** TE's FusedAdam with `master_weights=False` maintains optimizer states in BF16 only
- **Applied to:** All TE platforms; B300 uses PyTorch's fused AdamW

### 5.4 Memory Allocator Tuning
- **Impact:** Reduces CUDA memory fragmentation
- **Setting:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Applied to:** All platforms

### 5.5 Architecture-Specific Venvs
- **Problem:** Different GPU architectures require different TE wheel variants (cu12 vs cu13)
- **Solution:** `.venv-h200-cu12` for Hopper on x86_64 (cu13 kernels only support SM100+), `.venv-x86_64` for Blackwell, `.venv-aarch64` for ARM platforms

---

## 6. Platform-Specific Issues and Workarounds

### 6.1 TE cu13 Kernel Incompatibility on SM90 (H200)

**Problem:** `transformer_engine_cu13` pip wheels (versions 2.9–2.13) contain SASS compiled only for SM100 (Blackwell). Running on SM90 (Hopper H200) produces:
```
RuntimeError: CUDA Error: invalid argument
```
in RMSNorm/LayerNorm kernels.

**Root Cause:** The cu13 pip wheels target CUDA 13.0+ which defaults to SM100 architecture. SM90 support requires the cu12 wheel variant.

**Fix:** Created a separate venv (`.venv-h200-cu12`) with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126  # torch 2.11.0+cu126
pip install transformer_engine_cu12 transformer_engine_torch transformer_engine
```

### 6.2 TransformerEngine Kernel Bug on SM103 (B300 SXM6 AC)

**Problem:** All TE normalization kernels (RMSNorm, LayerNorm) crash with `CUDA Error: invalid argument` on the B300 SXM6 AC (compute capability 10.3 / sm_103).

**Scope:** Affects every tested configuration:
- TE 2.9.0 through 2.13.0 (pip cu13 wheels)
- TE 2.13.0 cu12 pip wheels (no SM100+ SASS at all)
- TE built from source with CUDA 13.2 toolkit (which includes sm_103a support)
- All hidden sizes (64 through 4096)
- Both "tuned" and "general" kernel paths
- Both RMSNorm and LayerNorm
- Both BF16 and FP32 input

**Key observation:** The sapling GB300 at CC 10.0 (sm_100) works perfectly with the same TE version. Pure PyTorch operations (matmul, `nn.RMSNorm`, `nn.LayerNorm`, `F.scaled_dot_product_attention`) all work correctly on the B300.

**Analysis:** The TE normalization kernels in `rmsnorm_fwd_cuda_kernel.cu` use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to compute grid dimensions and employ `cudaLaunchCooperativeKernel` for multi-CTA-per-row configurations. The kernel launch parameters that are valid on sm_100 hardware appear to be invalid on the sm_103 sub-architecture variant. The B300 reports 148 SMs and a default `sharedMemPerBlock` of 8192 bytes.

**Workaround:** Implemented `--no-te` flag that builds a pure-PyTorch model:
- Uses `nn.RMSNorm` / `nn.LayerNorm` (PyTorch native, works on sm_103)
- Uses `F.scaled_dot_product_attention` for flash attention
- Standard `nn.Linear` for all projections
- BF16 precision only (no FP8 support)
- ~10% lower MFU than TE-optimized runs due to lack of fused kernels

**Recommended action:** File a bug report with NVIDIA TransformerEngine team. Minimal reproducer:
```python
import transformer_engine.pytorch as te
import torch
norm = te.RMSNorm(64).cuda().bfloat16()
x = torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16)
y = norm(x)  # CUDA Error: invalid argument on B300 SXM6 AC (CC 10.3)
```

### 6.3 cuBLAS Version Pinning

**Problem:** TE 2.13.0 requires `cublasLtGroupedMatrixLayoutInit_internal` which is only available in `nvidia-cublas >= 13.3`. The pip dependency resolver sometimes installs 13.1.

**Fix:** Explicit upgrade after venv creation:
```bash
pip install "nvidia-cublas>=13.3"
```

### 6.4 nvidia-smi SIGPIPE on Multi-GPU Nodes

**Problem:** `nvidia-smi --query-gpu=... | head -1` under `set -o pipefail` causes exit code 141 (SIGPIPE) on multi-GPU nodes because `head` closes the pipe early.

**Fix:** Use `nvidia-smi --id=0` to query only GPU 0.

---

## 7. Code Changes Made

### 7.1 New Features

| File | Change | Purpose |
|------|--------|---------|
| `src/model.py` | Added `PureTorchGPTModel` class | Pure-PyTorch fallback for GPUs without TE kernel support |
| `src/model.py` | Added `_SwiGLU`, `_GELUMLP`, `_PureTorchTransformerBlock` | Building blocks for pure-PyTorch model |
| `src/model.py` | Added `use_te` parameter to `build_model()` | Selects between TE and pure-PyTorch model |
| `src/benchmark.py` | Added `--no-te` CLI flag | Enables pure-PyTorch mode |
| `src/benchmark.py` | Forces BF16 and disables fused attention when `--no-te` | Ensures compatibility |
| `src/trainer.py` | Added `use_te` parameter to `Trainer` | Skips TE autocast and FusedAdam when not using TE |
| `src/precision.py` | Fixed CC fallback ordering | Correct MFU calculation for unknown GPUs |
| `src/precision.py` | Added GH200, GB300 to `GPU_SPECS` | Correct peak TFLOPS for these GPUs |
| `scripts/lib/env_setup.sh` | New shared environment setup | Platform-aware venv activation, NCCL config, LD_LIBRARY_PATH |

### 7.2 Bug Fixes

| File | Fix | Impact |
|------|-----|--------|
| `src/model.py` | Fixed `forward_with_loss` else branch returning `(None, logits)` | Was crashing when `loss_chunk_size=0` |
| `src/precision.py` | Reordered compute-capability branches (CC≥10.0→Blackwell, CC≥9.0→Hopper) | Was assigning Blackwell TFLOPS to Hopper GPUs, inflating MFU by ~2.3× |

---

## 8. Reproducing the Results

### Grenoble H100 (jakku)
```bash
ssh grenoble-login
salloc -p jakku --exclusive -N1
cd /nfs/bruno/APPLICATIONS/CCC/transformer-bench
source scripts/lib/env_setup.sh
python -m src.benchmark --model-size 3b-llama --precision bf16 --batch-size 6 \
    --num-steps 50 --warmup-steps 10 --loss-chunk-size 512
```

### Grenoble GH200 (aarch64)
```bash
ssh grenoble-login
salloc -p aarch64 --exclusive -N1
cd /nfs/bruno/APPLICATIONS/CCC/transformer-bench
source scripts/lib/env_setup.sh
python -m src.benchmark --model-size 8b-llama --precision bf16 --batch-size 6 \
    --num-steps 50 --warmup-steps 10 --loss-chunk-size 512
```

### Vader H200
```bash
ssh monnetb@vader-login1.hpcrb.rdlabs.ext.hpe.com
salloc -p H200rhel94_lowpri --account=ai_users -t 4-00:00:00 -N1 -n8 --exclusive
cd /home/users/monnetb/bench-AI
source .venv-h200-cu12/bin/activate  # cu12 venv for SM90
# (set LD_LIBRARY_PATH, CUDNN_PATH, NVTE_FUSED_ATTN=1)
python -m src.benchmark --model-size 8b-llama --precision bf16 --batch-size 6 \
    --num-steps 50 --warmup-steps 10 --loss-chunk-size 512
```

### Vader B300
```bash
ssh monnetb@vader-login1.hpcrb.rdlabs.ext.hpe.com
salloc -p B300rhel94 --account=ai_users -t 4-00:00:00 -N1 -n8 --exclusive
cd /home/users/monnetb/bench-AI
source .venv-x86_64/bin/activate
# (set LD_LIBRARY_PATH, CUDNN_PATH)
python -m src.benchmark --model-size 8b-llama --precision bf16 --batch-size 6 \
    --num-steps 50 --warmup-steps 10 --loss-chunk-size 512 --no-te
```

### Sapling GB300
```bash
ssh hpebench@sapling.hpcrb.rdlabs.ext.hpe.com
salloc -N1 -n4 --exclusive
cd /home/hpebench/bruno/bench-AI
source .venv-aarch64/bin/activate
export NCCL_IB_DISABLE=1 NCCL_MNNVL_ENABLE=1
# (set LD_LIBRARY_PATH, CUDNN_PATH, NVTE_FUSED_ATTN=1)
python -m src.benchmark --model-size 8b-llama --precision bf16 --batch-size 6 \
    --num-steps 50 --warmup-steps 10 --loss-chunk-size 512
```

---

## 9. Conclusions

1. **All SM90 (Hopper) platforms and SM100 (Blackwell) achieve >50% MFU** with TransformerEngine fused kernels enabled. The best efficiency was obtained on the GH200 at 60.2% MFU.

2. **Blackwell delivers ~2.2× the absolute throughput of Hopper** in BF16 training, closely tracking the 2.28× theoretical peak ratio. The GB300 achieves 1,230 TFLOPS (1.23 PFLOP/s) per GPU.

3. **The B300 SXM6 AC (SM103) has a TransformerEngine kernel bug** that prevents TE's custom CUDA normalization kernels from running. A pure-PyTorch fallback achieves 44.3% MFU (997 TFLOPS). Once NVIDIA fixes TE for CC 10.3, the B300 should match the GB300's 54.7% MFU, which would put it at ~1,230 TFLOPS.

4. **Platform-specific venvs are essential** — cu13 TE wheels only support SM100+, requiring cu12 wheels for Hopper GPUs. The `scripts/lib/env_setup.sh` shared environment script handles architecture detection automatically.

5. **Key optimizations that pushed MFU above 50%:** fused attention (NVTE_FUSED_ATTN=1), chunked cross-entropy loss (--loss-chunk-size 512), FusedAdam without master weights, and expandable memory segments. Without these, typical MFU was 35–45%.

---

## 10. Open Items

| Item | Priority | Description |
|------|----------|-------------|
| TE bug report for SM103 | High | File GitHub issue with NVIDIA for B300 SXM6 AC kernel crashes |
| FP8 benchmarks | Medium | Run FP8-delayed and MXFP8 precision modes on all platforms |
| Multi-GPU scaling | Medium | Benchmark 8-GPU FSDP and TP configurations on vader/sapling |
| torch.compile | Low | Test `--use-compile --compile-mode max-autotune` for additional gains |
| B300 re-benchmark with TE | Blocked | Re-run B300 with TE once NVIDIA releases a fix for SM103 |
