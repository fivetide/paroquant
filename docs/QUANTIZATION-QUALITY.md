# Quantization quality evaluation

This document defines the canonical ParoQuant/llama.cpp quantization-quality evaluation protocol for Qwen3.6-35B-A3B.  The goal is to make every PPL/KLD/NLL/top-1 number comparable across HF, ParoQuant, and GGUF models.

## Canonical reference and validation source

**Reference model:** the original BF16 Hugging Face model, not a converted GGUF file.

```text
/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
```

**Default validation source:** the tx4/quality3-style held-out mix:

```text
/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-4096-val-64x2048-optfix.jsonl
```

This is the preferred held-out set for quantization-quality comparison because it is closer to the calibration/training distribution than WikiText-only and covers:

- English general text
- Japanese
- Chinese
- other multilingual text
- code breadth
- math/STEM
- chat/translation examples

The evaluator extracts each JSONL row's `text` field, joins rows with blank lines, tokenizes with the **original HF tokenizer**, and then evaluates every model on the exact same token IDs and scored token positions.

## Canonical scoring protocol

We use a rolling, overlapping-window protocol instead of disjoint blocks.

Default parameters:

```text
context/window length: 2048 tokens
warmup per window:    1025 tokens of left context for the first scored target
scored targets/window: 1023 tokens
window stride:        1023 tokens
```

For a local 2048-token window, the evaluator scores next-token predictions for target positions:

```text
local target indices 1025..2047 inclusive
```

Then it advances the window by 1023 tokens. This makes the scored target positions contiguous across windows after the initial warmup, rather than scoring only half of non-overlapping blocks.

Why this protocol:

- It avoids repeatedly scoring cold-start tokens with little context.
- It avoids llama.cpp's default non-overlapping half-block behavior that ignores half the corpus.
- It gives every compared model the exact same token IDs and score mask.
- It is compatible with llama.cpp's KLD-base format while still being a true rolling/strided evaluation.

## GGUF comparison rule

GGUF quantizations must be compared directly against the **original BF16 HF model**.

We do **not** use F16 GGUF as the reference for the primary table. F16 GGUF may appear as a sanity-check row, but primary quantization drift metrics are always:

```text
KL(P_original_BF16_HF || P_candidate)
NLL_candidate - NLL_original_BF16_HF
```

Implementation detail: the script can provide the original BF16 HF logits to llama.cpp in two ways:

1. **FIFO/live stream mode** — default and preferred when GPUs are free. A Python BF16-HF writer streams llama.cpp KLD-format logits through a named pipe while `llama-perplexity` evaluates the GGUF candidate. This avoids storing the large logits cache on disk, but recomputes BF16 logits for each GGUF candidate.
2. **Disk KLD-cache mode** — writes the BF16-HF logits base to disk once and reuses it for multiple GGUF candidates. This is faster for many GGUF models but needs substantial temporary storage.

For the default tx4/quality3 source and canonical `ctx=2048, stride=1023` layout, the dry-run layout is:

```text
HF token count:         131,317
windows:                127
prompt tokens:          260,096   # includes overlapping warmup context
scored tokens:          129,921
ignored prefix tokens:    1,025   # warmup before the first scored token
ignored tail tokens:        371   # not enough for another full canonical window
estimated KLD cache:     60.10 GiB
```

So if `--gguf-reference-mode disk` is used, reserve about **65–70 GiB free** for the BF16-HF logits cache plus filesystem headroom. If `--gguf-reference-mode fifo` is used, the large logits cache is not stored.

llama.cpp's built-in KLD path stores the reference distribution in its `_logits_` file/stream format: two float32 row parameters plus uint16 bins for each vocabulary log-probability. The source of those log-probabilities is the original BF16 HF model. This is high-fidelity but not a raw fp32 logit dump; the F16 GGUF sanity row is included to quantify the small end-to-end conversion/encoding/runtime difference.

For HF/ParoQuant candidates, the script defaults to a **CPU reference-logit cache**: it runs the original BF16 HF model once, stores only the scored fp16 logits in CPU RAM, unloads the reference model, then loads each ParoQuant candidate on a GPU. For the default canonical tx4/quality3 run this CPU cache is about **60.1 GiB**. This avoids disk usage and avoids ParoQuant custom-kernel issues on non-zero CUDA devices. There is also `--hf-compare-mode live` for compatible HF models, which keeps reference and candidate models on separate GPUs and avoids the CPU cache. On the RTX PRO 6000 Blackwell 96GB cards, the original BF16 model uses roughly the high-60 GiB range during this eval.

## Metric definitions

### PPL — perplexity

Lay explanation: perplexity is a measure of how surprised the model is by the validation text. Lower is better. A PPL of 6 means the model is, roughly, as uncertain as choosing among 6 equally likely next-token options on average.

Practitioner definition:

```text
NLL_i = -log P_model(token_i | previous context)
mean_NLL = mean_i(NLL_i)
PPL = exp(mean_NLL)
```

PPL is calculated only over the canonical scored target positions, not over prompt/warmup positions.

### NLL — negative log-likelihood

Mean NLL is the average `-log probability` assigned to the true next token, in natural-log units (`nats`). Lower is better.

### ΔNLL — quantization drift in true-token loss

```text
ΔNLL = mean_NLL_candidate - mean_NLL_original_BF16_HF
```

Lower is better. `0` means the candidate matches the original reference on true-token likelihood. Positive values mean the candidate assigns lower probability to the actual validation tokens than the original model.

### KLD / KL nats — distribution drift

The KL divergence compares the full next-token probability distribution of the original model to the candidate model:

```text
KL(P_ref || P_candidate) = sum_v P_ref(v) * [log P_ref(v) - log P_candidate(v)]
```

The table reports the mean KL over scored positions, in nats. Lower is better. KL sees distribution-level drift even when the true token's probability is similar.

### Top-1 agreement

Top-1 agreement is the percentage of scored positions where the candidate model's highest-probability next token is the same as the original BF16 HF model's highest-probability next token.

This is **not** benchmark accuracy. It is an argmax-distribution agreement metric. Higher is better.

### RMS Δp

For each scored position, let `p_ref` be the original model's probability for the actual next token and `p_candidate` be the candidate's probability for that same token.

```text
Δp = p_candidate - p_ref
RMS Δp = sqrt(mean(Δp^2))
```

The table reports this as a percentage. Lower is better.

### BPW — bits per weight

BPW is an on-disk size proxy:

```text
BPW = model_file_or_directory_bytes * 8 / denominator_parameters
```

The default denominator is `35,000,000,000` parameters for Qwen3.6-35B-A3B. The results table reports two BPW columns:

- **Artifact BPW**: active inference artifact size. For a directory export, the script uses active `model.safetensors` if present and ignores backup files such as `model.orig.safetensors`; otherwise it sums weight files (`*.safetensors`, `*.bin`, `*.gguf`, `*.pt`, `*.pth`).
- **Packed BPW estimate**: for ParoQuant safetensors, this estimates the deployable packed size after removing duplicate fp16 fallback tensors for modules that also have `qweight`. For GGUF this is the same as artifact BPW.

This distinction matters for the current ParoQuant HF bridge: active `model.safetensors` is about **5.32 BPW**. A hipENGINE-compatible stripped export that removes unused attention/linear-attention fp16 fallbacks but keeps shared-expert fp16 weights is about **4.73 BPW**. A pure no-fallback packed export is about **4.68 BPW** on the same denominator, but current hipENGINE needs a small loader/runtime change before consuming that fully stripped form.

A utility is available for producing stripped exports:

```bash
# Current hipENGINE-compatible stripped form (~4.73 BPW)
python3 scripts/strip_paro_safetensors.py \
  --input-dir /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e5 \
  --output-dir /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e5-hipengine-stripped \
  --mode hipengine

# Smallest packed estimate (~4.68 BPW), requires hipENGINE support for stripped shared_expert
python3 scripts/strip_paro_safetensors.py \
  --input-dir /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e5 \
  --output-dir /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-stripped \
  --mode packed
```

### Token/window columns

- `prompt_tokens`: number of token IDs in all evaluation windows, including overlapping warmup context.
- `scored_tokens`: number of target positions that contributed to PPL/NLL/KL/top-1.
- `windows`: number of rolling windows.
- `ctx`: context/window length.
- `stride`: window stride.

## Canonical script

Run from the repository root with an environment that can load both Transformers and ParoQuant:

```bash
PYTHONPATH=/home/lhl/paroquant/paroquant \
mamba run -n llmcompressor python3 scripts/quantization_quality_eval.py \
  --output-dir /models/eval-results/quant-quality-tx4-quality3 \
  --hf-compare-mode cache \
  --ref-device cuda:0 \
  --candidate-device cuda:0 \
  --gguf-ref-device cuda:0 \
  --gguf-cuda-visible-devices 2 \
  --gguf-reference-mode fifo
```

The script can evaluate:

- BF16 HF original baseline
- F16 GGUF sanity row vs BF16 HF original
- GGUF `UD-Q4_K_M` vs BF16 HF original
- ParoQuant HF exports vs BF16 HF original

For compatible non-Paro HF candidates, live two-GPU comparison is available by changing to `--hf-compare-mode live --ref-device cuda:0 --candidate-device cuda:1`.

For a disk-backed GGUF reference cache instead of FIFO streaming:

```bash
PYTHONPATH=/home/lhl/paroquant/paroquant \
mamba run -n llmcompressor python3 scripts/quantization_quality_eval.py \
  --output-dir /models/eval-results/quant-quality-tx4-quality3 \
  --gguf-reference-mode disk \
  --cleanup-reference-kld
```

Outputs include:

```text
results.json
RESULTS.md
canonical_eval_text.txt
canonical_eval_meta.json
```

In FIFO mode there is no persistent BF16-HF logits file. In disk mode, if `--keep-reference-kld` is used, the BF16-HF KLD-base file is retained for faster GGUF reruns; otherwise `--cleanup-reference-kld` deletes it after the results table is generated.

## Interpretation notes

- Use the canonical `RESULTS.md` table for model-to-model quantization-quality decisions.
- Do not compare absolute PPL values from different scoring protocols.
- For quantization, KL/ΔNLL/top-1 against the original BF16 model are usually more informative than absolute PPL alone.
- A converted F16 GGUF row is useful only as a runtime/conversion sanity check; it is not the primary reference.
