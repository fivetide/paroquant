# ParoQuant

**Pairwise Rotation Quantization for Efficient Reasoning LLM Inference**

<p>
  <a href="https://arxiv.org/abs/2511.10645"><img src="https://img.shields.io/badge/arXiv-2511.10645-b31b1b.svg" alt="Paper"></a>
  <a href="https://paroquant.z-lab.ai"><img src="https://img.shields.io/badge/Blog-ParoQuant-blue" alt="Blog"></a>
  <a href="https://huggingface.co/collections/z-lab/paroquant"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow" alt="Models"></a>
  <a href="https://pypi.org/project/paroquant/"><img src="https://img.shields.io/pypi/v/paroquant" alt="PyPI"></a>
</p>

## Shisa fork changes

This branch is our working integration branch on top of upstream ParoQuant `v0.1.15`.  The fork-specific changes we intend to upstream/cherry-pick are:

- **ROCm runtime support:** HIP-aware rotation extension loading plus ROCm fallbacks for ParoQuant/AWQ inference.
- **ROCm AWQ kernels:** direct GEMV and dequantize+GEMM paths for W4A16 AWQ-style tensors, with small-batch GEMV defaults tuned for MoE expert inference.
- **Qwen3.5/3.6 MoE real-export support:** real ParoQuant export preserves standard fp16 fallback weights when needed, saves exact PARO safetensors keys, and can load quantized Qwen MoE expert tensors in the Transformers backend.
- **Packed/export workflow support:** packed checkpoints can remove duplicate fp16 fallback tensors while preserving the quantized PARO tensors used by packed-aware runtimes.
- **Optimizer/checkpoint fixes:** saved quantizer tensors are restored directly during resume instead of recalibrating from full rotated weights, avoiding multi-GB temporary tensors on large MoE models.
- **Calibration improvements:** JSONL calibration sources are supported, including text rows, chat/message rows, prompt/completion rows, and chosen/preference-style data.
- **Build compatibility:** CUDA extension builds can select an older host compiler via `PAROQUANT_CUDAHOSTCXX`/`CUDAHOSTCXX` for CUDA 13.x toolchains on newer Linux distributions.

### Qwen3.6-35B-A3B quality/size snapshot

Canonical tx4/quality3 evaluation compares each candidate directly against the original BF16 HF model on the same scored token positions.

| Model | Format | Size GiB ↓ | BPW ↓ | PPL ↓ | ΔNLL ↓ | KL nats ↓ | Top-1 % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Original BF16 HF | HF safetensors | 66.966 | 16.435 | 6.5590 | +0.000000 | 0.000000 | 100.000 |
| PARO full4096-e5 packed | packed safetensors | 19.068 | 4.680 | 6.6216 | +0.009506 | 0.034684 | 92.000 |
| PARO full4096-e5 unpacked/original-format | legacy safetensors | 21.686 | 5.322 | 6.6216 | +0.009506 | 0.034684 | 92.000 |
| GGUF UD-Q4_K_S | GGUF | 19.458 | 4.776 | 6.5783 | +0.003842 | 0.012800 | 94.999 |
| GGUF UD-Q4_K_M | GGUF | 20.614 | 5.059 | 6.5643 | +0.001718 | 0.010849 | 95.354 |

See [`docs/QUANTIZATION-QUALITY.md`](docs/QUANTIZATION-QUALITY.md) for metric definitions and the canonical evaluation protocol. Size GiB is active weight artifact bytes divided by `2^30`; BPW is active weight artifact bytes × 8 / `35,000,000,000`.

Local runbooks and utilities for the Qwen3.6 PARO work are included under:

- [`docs/PAROQUANT-COMPRESSION.md`](docs/PAROQUANT-COMPRESSION.md) — Qwen3.6 compression/optimization run notes.
- [`docs/QUANTIZATION-QUALITY.md`](docs/QUANTIZATION-QUALITY.md) — canonical PPL/ΔNLL/KLD/top-1 evaluation protocol and interpretation notes.
- [`scripts/`](scripts/) — calibration mix builders, quantization-quality eval scripts, safetensors patch/strip helpers, and comparison utilities used for the local Qwen3.6 experiments.

State-of-the-art INT4 quantization for LLMs. ParoQuant uses learned pairwise rotations to suppress weight outliers, closing the accuracy gap with FP16 while running at near-AWQ speed. Supports NVIDIA GPUs (vLLM, Transformers) and Apple Silicon (MLX).

## Quick Start

### Installation

```bash
# NVIDIA GPU (CUDA 12.9)
pip install "paroquant[vllm]"

# NVIDIA GPU (CUDA 13.0)
pip install "paroquant[vllm]" "vllm==0.19.1" \
  --extra-index-url https://wheels.vllm.ai/0.19.1/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130

# Apple Silicon
pip install "paroquant[mlx]"
```

Pick a model from our [Hugging Face collection](https://huggingface.co/collections/z-lab/paroquant):

```bash
export MODEL=z-lab/Qwen3.5-4B-PARO
```

### Interactive Chat

```bash
python -m paroquant.cli.chat --model $MODEL
```

### OpenAI-Compatible API Server

For vLLM, you can directly use `vllm serve` to serve ParoQuant models:

```bash
vllm serve $MODEL --port 8000
```

For other frameworks:

```bash
python -m paroquant.cli.serve --model $MODEL --port 8000
```

For MLX, add `--vlm` if you wish to load the VLM components and use the model's multimodal features. For vLLM, VLM components are loaded by default and can be skipped with the server argument `--language-model-only`.

### Docker (NVIDIA GPU)

> [!NOTE]
> The following commands map the local cache directory to the container in order to persist kernel cache across runs. Remove `-v ...` to disable this behaviour.

```bash
# Interactive chat
docker run --pull=always --rm -it --gpus all --ipc=host \
  -v $HOME/.cache/paroquant:/root/.cache/paroquant \
  ghcr.io/z-lab/paroquant:chat --model $MODEL

# API server (port 8000)
docker run --pull=always --rm -it --gpus all --ipc=host -p 8000:8000 \
  -v $HOME/.cache/paroquant:/root/.cache/paroquant \
  ghcr.io/z-lab/paroquant:serve --model $MODEL
```

## Models

All models are available on [Hugging Face](https://huggingface.co/collections/z-lab/paroquant). Swap the model name in the commands above to try any of them.

**Gemma 4**

| Model              | Checkpoint                                                                              |
| ------------------ | --------------------------------------------------------------------------------------- |
| gemma-4-31B-it     | [`z-lab/gemma-4-31B-it-PARO`](https://huggingface.co/z-lab/gemma-4-31B-it-PARO)         |
| gemma-4-26B-A4B-it | [`z-lab/gemma-4-26B-A4B-it-PARO`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-PARO) |
| gemma-4-E4B-it     | [`z-lab/gemma-4-E4B-it-PARO`](https://huggingface.co/z-lab/gemma-4-E4B-it-PARO)         |
| gemma-4-E2B-it     | [`z-lab/gemma-4-E2B-it-PARO`](https://huggingface.co/z-lab/gemma-4-E2B-it-PARO)         |

**Qwen3.6**

| Model           | Checkpoint                                                                        |
| --------------- | --------------------------------------------------------------------------------- |
| Qwen3.6-27B     | [`z-lab/Qwen3.6-27B-PARO`](https://huggingface.co/z-lab/Qwen3.6-27B-PARO)         |
| Qwen3.6-35B-A3B | [`z-lab/Qwen3.6-35B-A3B-PARO`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-PARO) |

**Qwen3.5**

| Model           | Checkpoint                                                                        |
| --------------- | --------------------------------------------------------------------------------- |
| Qwen3.5-0.8B    | [`z-lab/Qwen3.5-0.8B-PARO`](https://huggingface.co/z-lab/Qwen3.5-0.8B-PARO)       |
| Qwen3.5-2B      | [`z-lab/Qwen3.5-2B-PARO`](https://huggingface.co/z-lab/Qwen3.5-2B-PARO)           |
| Qwen3.5-4B      | [`z-lab/Qwen3.5-4B-PARO`](https://huggingface.co/z-lab/Qwen3.5-4B-PARO)           |
| Qwen3.5-9B      | [`z-lab/Qwen3.5-9B-PARO`](https://huggingface.co/z-lab/Qwen3.5-9B-PARO)           |
| Qwen3.5-27B     | [`z-lab/Qwen3.5-27B-PARO`](https://huggingface.co/z-lab/Qwen3.5-27B-PARO)         |
| Qwen3.5-35B-A3B | [`z-lab/Qwen3.5-35B-A3B-PARO`](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-PARO) |

**Qwen3**

| Model      | Checkpoint                                                              |
| ---------- | ----------------------------------------------------------------------- |
| Qwen3-0.6B | [`z-lab/Qwen3-0.6B-PARO`](https://huggingface.co/z-lab/Qwen3-0.6B-PARO) |
| Qwen3-1.7B | [`z-lab/Qwen3-1.7B-PARO`](https://huggingface.co/z-lab/Qwen3-1.7B-PARO) |
| Qwen3-4B   | [`z-lab/Qwen3-4B-PARO`](https://huggingface.co/z-lab/Qwen3-4B-PARO)     |
| Qwen3-8B   | [`z-lab/Qwen3-8B-PARO`](https://huggingface.co/z-lab/Qwen3-8B-PARO)     |
| Qwen3-14B  | [`z-lab/Qwen3-14B-PARO`](https://huggingface.co/z-lab/Qwen3-14B-PARO)   |

**Llama**

| Model                 | Checkpoint                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Llama-2-7B            | [`z-lab/Llama-2-7b-hf-PARO`](https://huggingface.co/z-lab/Llama-2-7b-hf-PARO)                 |
| Llama-3-8B            | [`z-lab/Meta-Llama-3-8B-PARO`](https://huggingface.co/z-lab/Meta-Llama-3-8B-PARO)             |
| Llama-3.1-8B-Instruct | [`z-lab/Llama-3.1-8B-Instruct-PARO`](https://huggingface.co/z-lab/Llama-3.1-8B-Instruct-PARO) |

Want a model that's not listed? [Open an issue](https://github.com/z-lab/paroquant/issues/new) and let us know.

## Reproduction

> [!NOTE]
> The main branch of this repository is under active development, and reproducibility is not guaranteed.
> Please use the [`legacy`](https://github.com/z-lab/paroquant/tree/legacy) branch to reproduce results from the paper.

## Quantize Your Own Model

```bash
git clone https://github.com/z-lab/paroquant && cd paroquant
pip install -e ".[optim,eval]"

# 1. Optimize rotation parameters
experiments/optimize/4bit.sh Qwen/Qwen3-8B

# 2. Export to HF checkpoint (--mode real for INT4, --mode pseudo for FP16)
python -m paroquant.cli.convert \
  --model Qwen/Qwen3-8B \
  --result-dir output/Qwen3-8B \
  --output-path models/Qwen3-8B-PARO
```

## Docker Images

| Image                                | Purpose                      |
| ------------------------------------ | ---------------------------- |
| `ghcr.io/z-lab/paroquant:chat`       | Interactive chat             |
| `ghcr.io/z-lab/paroquant:chat-cu129` | Interactive chat (CUDA 12.9) |
| `ghcr.io/z-lab/paroquant:serve`      | OpenAI-compatible API server |
| `ghcr.io/z-lab/paroquant:latest`     | Optimization & evaluation    |
| `ghcr.io/z-lab/paroquant:eval`       | Reasoning task evaluation    |

## Citation

```bibtex
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Zhang, Zihan and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
