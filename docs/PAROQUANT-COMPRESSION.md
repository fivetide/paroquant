# Qwen3.6-35B-A3B PARO Compression Runbook

Last updated: 2026-05-10

This is the operational runbook for producing a local PARO/W4A16 checkpoint for
`Qwen/Qwen3.6-35B-A3B`, because no public `z-lab/Qwen3.6-35B-A3B-PARO`
artifact is available yet.

The current local proof is **mechanical**, not quality-final:

- Target BF16 source exists locally in the Hugging Face cache at
  `/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Config is compatible with the current ParoQuant Qwen3.5-MoE path:
  `model_type=qwen3_5_moe`, architecture `Qwen3_5MoeForConditionalGeneration`,
  `text_config.num_hidden_layers=40`.
- The `paroquant` fork can optimize and real-pack layer 0 on ROCm/W7900.
- The same layer-0 optimize + real-pack smoke also passes on this PRO 6000 CUDA
  host using `llmcompressor`, `CUDA_VISIBLE_DEVICES=0`, and the HF cache source
  path. Smoke output is under
  `/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0`.
- The full 40-layer quantization job has not been run yet.

## Can compression run on another machine?

Yes. The compression/optimization step can run on a different machine from
serving/benchmarking. The resulting HF/PARO checkpoint is portable back to the
W7900 box as normal model files (`config.json`, tokenizer assets,
`safetensors`).

Both ROCm and CUDA machines are valid compression hosts:

- **ROCm/RDNA3**, e.g. W7900 or 7900 XTX, uses the local fork's HIP rotation and
  ROCm-safe real-export bridge.
- **CUDA/NVIDIA**, e.g. RTX/PRO 6000, uses the upstream CUDA rotation kernel path
  and is likely the better place to leave a multi-day optimization run if the
  GPU is otherwise free.

The important requirements are:

- GPU-visible PyTorch (`torch.cuda.is_available() == True`). PyTorch still uses
  the `cuda` API name on both CUDA and ROCm; check `torch.version.hip` to
  distinguish ROCm.
- Enough VRAM for one Qwen3.6 layer plus calibration batches. The optimizer is
  layerwise; it does **not** need the whole 35B model resident on GPU.
- Enough CPU RAM and disk for the source model, intermediate optimizer states,
  and final checkpoint.

### CUDA / PRO 6000 viability

A free PRO 6000-class NVIDIA box is probably the best compression host. If this
is an RTX PRO 6000 Blackwell / Ada-class card with **48-96GB VRAM**, it should
have much more headroom than the 24GB 7900 XTX for `seqlen=2048`, larger
`batch-size`, and the paper/default recipe. Still run the 1-layer smoke first,
because the Qwen3.6 full job has not yet been validated on CUDA in this
workspace.

Expected status:

| Run type | PRO 6000-class CUDA? | Notes |
| --- | --- | --- |
| 1-layer smoke | **Yes** | Should be the first command on the box; confirms CUDA extension build and model/dataset paths. |
| Full 40-layer low-calibration pilot | **Yes** | Prefer this before a multi-day default-quality job. |
| Full paper/default recipe | **Likely yes on 48-96GB; tune if 24GB** | Try upstream/default `batch-size 16`, `seqlen 2048` only after a smoke. If OOM, reduce batch size and add gradient accumulation. |
| Final serving on CUDA | Not the goal here | Compression can run on CUDA; final W7900/native runtime validation still happens back on the AMD box. |

CUDA caveat: the local fork's Qwen3.5-MoE **HF runtime bridge** has been tested
for the ROCm path, not as a CUDA serving path for the real MoE PARO artifact.
That does not block using CUDA for optimization/export. For post-export runtime
validation, copy the artifact back and use the W7900 native/HF validation path.

### 24GB 7900 XTX viability

Expected status:

| Run type | 24GB 7900 XTX? | Notes |
| --- | --- | --- |
| 1-layer smoke | **Likely yes** | Use `--batch-size 1`, small `--seqlen`, and `--max-layers 1`. |
| Full 40-layer low-calibration pilot | **Likely yes** | Use small `train-size`, `batch-size 1`, and resume. Watch VRAM on expert layers. |
| Full paper/default recipe | **Maybe, not guaranteed** | Default `batch-size 16`, `seqlen 2048`, `train-size 2048`, `epochs 10 10` may exceed 24GB. Lower `batch-size`, increase `gradient_accumulation_steps`, and use `cache-shards`. |
| Final exported checkpoint serving | Separate question | The current PARO native runtime target is ~22GB peak on W7900 for 4K/4K after compact/repack work, so 24GB serving is tight but the promotion gate is exactly 24GB usability. |

For the 24GB box, start with conservative settings:

```text
batch-size: 1
gradient-accumulation-steps: increase if you need a larger effective batch
cache-shards: 4-16 if CPU RAM/disk are available
seqlen: 128-512 for pilot; 2048 only after memory is proven
```

If a layer OOMs, rerun with smaller `--batch-size`, smaller `--seqlen`, and/or
more `--cache-shards`. The optimizer is resumable with `--resume`, so completed
layers should not need to be recomputed.

## Runtime and storage estimates

Measured on W7900/gfx1100:

- Layer-0 tiny smoke, `train-size=1`, `seqlen=16`, `epochs=1 1`: layer body
  about `42.6s`.
- Layer-0 tiny smoke, `train-size=1`, `seqlen=128`, `epochs=1 1`: layer body
  `38.08s`, wall `59s` including model/process load.
- Layer-0 intermediate optimizer output: about `1.6GB`, dominated by
  `0.mlp.experts.pt`.

Measured on PRO 6000 CUDA/GPU2 with the weighted local mix:

- Layer-0 pilot profile, `train-size=128`, `validation-size=16`, `seqlen=2048`,
  `batch-size=8`, `gradient-accumulation-steps=2`, `epochs=1 1`: wall `67s`,
  layer body `61.95s`.
- The optimized step body was about `36s/layer` for the two 1-epoch stages;
  fixed per-layer capture/export overhead was about `31s`.

Planning estimates:

| Recipe | Expected time on PRO 6000 CUDA | Expected time on one W7900-class GPU | Intermediate disk |
| --- | ---: | ---: | ---: |
| 40-layer tiny smoke (`train-size=1`, `epochs=1 1`) | ~25-45 min plus conversion | ~25-60 min plus conversion | ~64GB |
| 40-layer weighted pilot (`train-size=128`, `epochs=1 1`, `seqlen=2048`) | ~45-60 min plus conversion | Several hours to ~1 day | ~64GB |
| 40-layer weighted medium (`train-size=512`, `epochs=1 1`, `seqlen=2048`) | ~2 hours plus conversion | likely ~1-2 days | ~64GB |
| 40-layer weighted full-cal, 1 epoch/stage (`train-size=2048`, `epochs=1 1`) | ~7 hours plus conversion | likely several days | ~64GB |
| Paper/default recipe (`train-size=2048`, `batch-size=16`, `seqlen=2048`, `epochs=10 10`) | ~2.5-3.5 days plus conversion | Roughly ~4-10 days | ~64GB+ |

Final exported PARO checkpoint should be in the ~20GB-class range, similar to
`z-lab/Qwen3.5-35B-A3B-PARO`, but conversion temporarily loads/saves large
state and should have generous disk/CPU RAM headroom. Keep at least **150-250GB
free** for a comfortable remote compression run.

## Source code and environment

Use the working fork, not the read-only reference copy. The fork keeps CUDA
behavior for NVIDIA while adding ROCm compatibility and Qwen3.6 BF16 optimizer
fixes:

```bash
cd /home/lhl/paroquant/paroquant
git status -sb
git log -1 --oneline
```

Required local commit for the BF16 Qwen3.6 optimizer smoke:

```text
555bfd6 feat: support BF16 optimizer smoke runs
```

Use the `llmcompressor` environment on this PRO 6000 CUDA workstation. Do not
use `base` for the run even if some packages are installed there; the current
base env has a Transformers/Hugging Face Hub version mismatch. On a separate
machine, create an environment matching that machine's GPU stack and install
the fork editable.

### ROCm environment sketch

```bash
# Example only; adapt paths/env manager to the remote box.
mamba create -n therock python=3.12 -y
mamba activate therock
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchaudio torchvision
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "rocm[libraries,devel]" -U
pip install -e /path/to/paroquant
pip install simple_parsing datasets safetensors transformers huggingface_hub tqdm
```

Do **not** install CUDA-only `flash_attn` or AutoAWQ CUDA kernels into the ROCm
environment. The local fork has a ROCm-safe real-export/runtime bridge.

### CUDA environment sketch

On a CUDA box, use the normal CUDA PyTorch wheels. The optimizer needs the CUDA
compiler toolchain because ParoQuant builds `paroquant.kernels.cuda.rotation.cu`
through `torch.utils.cpp_extension` on first import.

```bash
# Example only; choose a CUDA wheel/index matching the driver on the PRO 6000 box.
mamba create -n paroquant-cuda python=3.12 -y
mamba activate paroquant-cuda
# Pick the CUDA build appropriate for the host, e.g. cu128/cu129/cu130.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e /path/to/paroquant
pip install simple_parsing datasets safetensors transformers huggingface_hub tqdm ninja
# Optional for upstream CUDA HF serving paths; not required for optimizer-only smoke.
# pip install autoawq
```

For CUDA, make sure `nvcc` is available and compatible with the installed PyTorch
CUDA version:

```bash
nvidia-smi
nvcc --version
python3 - <<'PY'
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('torch.version.cuda:', torch.version.cuda)
print('torch.version.hip:', torch.version.hip)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Quick GPU check for either backend:

```bash
mamba run -n llmcompressor python3 - <<'PY'
import torch
print(torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('HIP:', torch.version.hip)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Inputs to copy to a separate machine

Minimum:

1. `paroquant` fork at/after `555bfd6` plus ROCm runtime fallback changes. Use
   the same fork on CUDA too, so the Qwen3.6 BF16 optimizer fixes and
   `--max-layers` smoke knob are available.
2. Source model snapshot on this host:
   `/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
3. Dataset cache or internet access for calibration datasets. On this host, use
   `/home/lhl/.cache/huggingface/datasets`.

Optional but useful:

- Existing smoke output for inspection:
  `/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0`
- Local docs: this file, `WORKLOG.md`, `docs/PARO.md`, `docs/QUALITY.md`.

Recommended local paths for this PRO 6000 host:

```text
/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/  # source BF16 snapshot
/models/qwen36-paroquant-work/          # optimizer .pt outputs
/models/qwen36-quant/Qwen3.6-35B-A3B-PARO-pilot/
/home/lhl/.cache/huggingface/datasets/  # dataset cache
```

Because the raw HF snapshot path is used as `--model`, ParoQuant names optimizer
result dirs after the snapshot hash (`995ad96e...`). If you want a nicer
result-dir name on another machine, copy or symlink the snapshot to a short path
such as `/models/source/Qwen3.6-35B-A3B` and use that as `--model`.

## 1-layer smoke command

Run this first on any new machine. The local PRO 6000 command uses the raw HF
cache snapshot path and the `llmcompressor` env:

```bash
PARO_REPO=/home/lhl/paroquant/paroquant
SOURCE_MODEL=/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
HF_DATASETS_CACHE=/home/lhl/.cache/huggingface/datasets

rm -rf /models/qwen36-paroquant-smoke-output
PYTHONPATH="$PARO_REPO" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  mamba run -n llmcompressor python3 -m paroquant.cli.optimize \
  --model "$SOURCE_MODEL" \
  --params "channel_scales:0.001,angles:0.001" "quantizer:1e-6" \
  --epochs 1 1 \
  --group-size 128 \
  --n-bit 4 \
  --num-rotations 8 \
  --skipped-modules "mlp.gate" "mlp.shared_expert_gate" "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
  --datasets wikitext2 \
  --val-dataset wikitext2 \
  --train-size 1 \
  --validation-size 1 \
  --batch-size 1 \
  --seqlen 128 \
  --cache-shards 1 \
  --output-dir /models/qwen36-paroquant-smoke-output \
  --max-layers 1 \
  --seed 0
```

Expected output:

```text
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.linear_attn.out_proj.pt
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.linear_attn.in_proj_qkv.pt
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.linear_attn.in_proj_z.pt
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.mlp.shared_expert.gate_proj.pt
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.mlp.shared_expert.up_proj.pt
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.mlp.shared_expert.down_proj.pt
/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0/0.mlp.experts.pt
```

## Real-packing smoke without full checkpoint export

This verifies that the optimizer `.pt` files can be converted to packed PARO/AWQ
buffers without writing a full model checkpoint:

```bash
PYTHONPATH=/home/lhl/paroquant/paroquant mamba run -n llmcompressor python3 - <<'PY'
from pathlib import Path
import torch
from paroquant.cli.convert import _quantize_layer, _quantize_moe

rd = Path('/models/qwen36-paroquant-smoke-output/995ad96eacd98c81ed38be0c5b274b04031597b0')
for p in sorted(p for p in rd.glob('0.*.pt') if p.name != '0.mlp.experts.pt'):
    sd = torch.load(p, map_location='cpu', weights_only=False)
    buffers, bits, group, krot = _quantize_layer(sd, device='cuda')
    torch.cuda.synchronize()
    print(p.name, bits, group, krot, tuple(buffers['qweight'].shape), buffers['qweight'].dtype)
    del sd, buffers
    torch.cuda.empty_cache()

sd = torch.load(rd / '0.mlp.experts.pt', map_location='cpu', weights_only=False)
buffers, rotations, bits, group, krot = _quantize_moe(sd, device='cuda')
torch.cuda.synchronize()
print('moe', bits, group, krot, {p: tuple(buffers[p]['qweight'].shape) for p in ('gate_proj', 'up_proj', 'down_proj')})
PY
```

Expected for Qwen3.6 layer 0:

```text
bits/group/krot = 4/128/8
moe gate/up qweight = (256, 2048, 64)
moe down qweight = (256, 512, 256)
```

## Calibration data choice

`wikitext2` is only a smoke-test source. For a quality artifact, use a
representative calibration mix that covers the model's intended distribution:
English/general text, Japanese, Chinese/multilingual, code/math, and a small
amount of chat/translation formatting. The optimizer is not training the model,
but activation/weight calibration is still distribution-sensitive.

The local fork now accepts local JSONL paths in `--datasets` and `--val-dataset`
in addition to the built-in names (`wikitext2`, `c4`, `redpajama`, `pileval`).
Supported JSONL schemas include:

- `{text: ...}` TX4 cache docs;
- `{conversations: [...]}` / `{messages: [...]}` chat datasets rendered through
  the Qwen tokenizer chat template;
- DPO `{chosen: [...], rejected: [...]}` rows, using `chosen` only;
- GAD-style `{prompt: [...], teacher: ...}` rows.

Good local calibration sources on this host:

```text
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/fineweb2-ja/*.jsonl
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/fineweb2-zh/*.jsonl
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/fineweb2-ko/*.jsonl
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/fineweb-sample/*.jsonl
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/fineweb-edu-sample/*.jsonl
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/stack-edu-python-materialized/*.jsonl
/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/swallow-math-v2-textbook/*.jsonl
/home/lhl/chotto-train/202601-chotto/chotto-20260107.sft.jsonl
/home/lhl/quantize/EAGLE3/sft.shisa-v2.1-EAGLE3.jsonl
```

Avoid using held-out eval/test JSONLs for calibration. Also avoid making the mix
all chat/translation; keep most samples as natural pretrain-style text so general
capability does not drift.

A weighted mix builder is available at
`scripts/build_qwen36_calibration_mix.py`. Default output uses 2048 training
samples and 64 validation samples at sequence length 2048, with this mix:

```text
30% English/general/reference
20% Japanese
 8% Chinese
 8% other multilingual
10% math/STEM
16% broad Stack-Edu code breadth
 8% Chotto SFT chat/translation
```

Build or rebuild the default final-calibration files with:

```bash
mamba run -n llmcompressor python scripts/build_qwen36_calibration_mix.py
```

Current generated files:

```text
/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-train-2048x2048.jsonl
/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-val-64x2048.jsonl
/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-manifest.json
```

Recommended sample counts:

| Run | train-size | validation-size | seqlen | Purpose |
| --- | ---: | ---: | ---: | --- |
| quick smoke | 1-8 | 1-2 | 128-512 | verify code path only |
| PRO 6000 pilot | 128-256 | 16 | 2048 | validate all layers, runtime, and quality trend |
| medium quality | 512-1024 | 32 | 2048 | useful if final recipe is too expensive |
| final/default quality | 2048 | 64 | 2048 | matches upstream recipe shape |

## Full low-calibration pilot

This is the recommended next real run. It produces all 40 layers with a small
calibration budget. It is **not** the final quality recipe, but it validates the
complete optimizer, conversion, and runtime path.

Example conservative 24GB-friendly pilot, suitable for 7900 XTX or any unknown
CUDA memory size:

```bash
PARO_REPO=/home/lhl/paroquant/paroquant
SOURCE_MODEL=/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
HF_DATASETS_CACHE=/home/lhl/.cache/huggingface/datasets

PYTHONPATH="$PARO_REPO" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  mamba run -n llmcompressor python3 -m paroquant.cli.optimize \
  --model "$SOURCE_MODEL" \
  --params "channel_scales:0.01,angles:0.01" "quantizer:1e-6" \
  --epochs 1 1 \
  --group-size 128 \
  --n-bit 4 \
  --num-rotations 8 \
  --skipped-modules "mlp.gate" "mlp.shared_expert_gate" "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
  --datasets wikitext2 c4 redpajama \
  --val-dataset pileval \
  --train-size 16 \
  --validation-size 8 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --seqlen 512 \
  --cache-shards 8 \
  --output-dir /models/qwen36-paroquant-work \
  --resume \
  --seed 0
```

If stable and memory allows, scale in this order:

1. `seqlen`: `512 -> 1024 -> 2048`
2. `train-size`: `16 -> 64 -> 256 -> 512+`
3. `batch-size`: only raise after VRAM headroom is proven
4. `epochs`: raise only for a final quality run

For a high-memory PRO 6000 CUDA box, a more useful pilot is to jump straight to
`seqlen=2048` while keeping only `epochs=1 1` and a modest `train-size`, e.g.:

```text
train-size=64 or 128
validation-size=16
batch-size=4, 8, or 16 depending on VRAM
seqlen=2048
epochs=1 1
cache-shards=1-4
```

This gives a much better signal about final-run memory and per-layer time than
the tiny 24GB-safe pilot.

Suggested PRO 6000 multilingual low-calibration pilot using the weighted mix:

```bash
PARO_REPO=/home/lhl/paroquant/paroquant
SOURCE_MODEL=/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
TRAIN_MIX=/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-train-2048x2048.jsonl
VAL_MIX=/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-val-64x2048.jsonl

PYTHONPATH="$PARO_REPO" \
  mamba run -n llmcompressor python3 -m paroquant.cli.optimize \
  --model "$SOURCE_MODEL" \
  --params "channel_scales:0.01,angles:0.01" "quantizer:1e-6" \
  --epochs 1 1 \
  --group-size 128 \
  --n-bit 4 \
  --num-rotations 8 \
  --skipped-modules "mlp.gate" "mlp.shared_expert_gate" "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
  --datasets "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --train-size 128 \
  --validation-size 16 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --seqlen 2048 \
  --cache-shards 2 \
  --output-dir /models/qwen36-paroquant-work \
  --resume \
  --seed 0
```

## Paper/default-style run

Recommended final/default-quality run using the weighted local mix:

```bash
PARO_REPO=/home/lhl/paroquant/paroquant
SOURCE_MODEL=/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
TRAIN_MIX=/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-train-2048x2048.jsonl
VAL_MIX=/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-val-64x2048.jsonl

PYTHONPATH="$PARO_REPO" \
  mamba run -n llmcompressor python3 -m paroquant.cli.optimize \
  --model "$SOURCE_MODEL" \
  --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
  --epochs 10 10 \
  --group-size 128 \
  --n-bit 4 \
  --num-rotations 8 \
  --skipped-modules "mlp.gate" "mlp.shared_expert_gate" "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
  --datasets "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --train-size 2048 \
  --validation-size 64 \
  --batch-size 16 \
  --seqlen 2048 \
  --cache-shards 1 \
  --output-dir /models/qwen36-paroquant-work \
  --resume \
  --seed 0
```

On a 24GB 7900 XTX, expect to reduce `--batch-size` and compensate with
`--gradient-accumulation-steps`; for example `--batch-size 1
--gradient-accumulation-steps 16`. This preserves effective batch size for
optimizer stepping but increases wall time.

On a 48-96GB PRO 6000 CUDA box, try the default `--batch-size 16` only after a
1-layer `seqlen=2048` pilot. If it fits, that box is the preferred place to let
the multi-day job run. If it OOMs, reduce to `--batch-size 8` or `4` and add
matching `--gradient-accumulation-steps`.

## Export full PARO checkpoint

After all 40 layers have `.pt` outputs, export a real packed checkpoint:

```bash
PARO_REPO=/home/lhl/paroquant/paroquant
SOURCE_MODEL=/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0

PYTHONPATH="$PARO_REPO" \
  mamba run -n llmcompressor python3 -m paroquant.cli.convert \
  --model "$SOURCE_MODEL" \
  --result-dir /models/qwen36-paroquant-work/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-path /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-pilot \
  --mode real
```

Expected checkpoint contract:

- `config.json` has `quantization_config`:
  `{quant_method: paroquant, bits: 4, group_size: 128, krot: 8}`.
- Safetensors include dense `.qweight`, `.qzeros`, `.scales`, `.theta`, `.pairs`,
  `.channel_scales` where applicable.
- MoE expert tensors are stored under per-expert `gate_proj`, `up_proj`, and
  `down_proj` PARO/AWQ keys, plus shared rotation tensors.

## Move artifacts back to the W7900 box

Copy only the exported checkpoint for inference work:

```bash
rsync -a --info=progress2 \
  remote:/models/qwen36-quant/Qwen3.6-35B-A3B-PARO-pilot/ \
  /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-pilot/
```

Keep the optimizer `.pt` work directory on the compression box until quality is
accepted; it is required for resume/re-export.

## Validation after export

Minimum metadata check, valid on either CUDA or ROCm:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/models/qwen36-quant/Qwen3.6-35B-A3B-PARO-pilot/config.json')
cfg = json.loads(p.read_text())
print(cfg.get('model_type'), cfg.get('architectures'))
print(cfg.get('quantization_config'))
PY
```

HF load smoke through the ParoQuant quantizer bridge. This has been used on
ROCm for the local bridge; on CUDA, treat it as optional until the Qwen3.6 MoE
real-runtime path is tested there. CUDA compression/export does not require this
serving smoke to pass on the compression box.

```bash
PYTHONPATH=/home/lhl/paroquant/paroquant \
  mamba run -n llmcompressor python3 - <<'PY'
import torch
import paroquant.inference.backends.transformers.quantizer  # registers HF quantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

path = '/models/qwen36-quant/Qwen3.6-35B-A3B-PARO-pilot'
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map='cuda', dtype=torch.float16, trust_remote_code=True)
ids = tok('hello', return_tensors='pt').input_ids.to('cuda')
with torch.no_grad():
    out = model(ids)
torch.cuda.synchronize()
print(tuple(out.logits.shape), out.logits.dtype, torch.isfinite(out.logits).all().item())
PY
```

Native-runtime validation should then use the existing PARO proof/bench scripts,
adjusting `--model-path` to the new exported checkpoint.

## Quality gate before spending days

Do not treat a low-calibration pilot as a final artifact. Before committing to a
multi-day default run, compare the pilot against Qwen3.6 BF16/W8A8 controls using
`docs/QUALITY.md`:

- generation sanity on fixed prompts;
- logit/KL or NLL drift on a small slice, if feasible;
- HumanEval-X or coding smoke against the W8A8 artifact;
- native runtime proof-of-life and 4K/4K benchmark once the loader accepts the
  new artifact.

If the low-calibration pilot is badly degraded, tune calibration settings before
launching the expensive default job.
