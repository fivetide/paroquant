#!/usr/bin/env python3
"""Compare quantization quality across model checkpoints.

Metrics:
  1. PPL  — perplexity on wikitext2 test set (standard benchmark)
  2. BPW  — bits per weight = file_bytes * 8 / num_params
  3. KL / NLL drift vs BF16 reference (ParoQuant models only)

Usage:
  python scripts/eval_compare_models.py \
    --models /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
             /models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e1 \
    --cuda 1 \
    --out /models/eval-results/20260512-wikitext2-ppl.json

  # From a list file
  python scripts/eval_compare_models.py \
    --models-list /models/eval-results/models-to-eval.txt \
    --cuda 1
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from functools import lru_cache

# ── helpers ──────────────────────────────────────────────────────────────────

def bytes_per_weight(path: Path) -> float:
    """Compute BPW = total_file_bytes × 8 / num_params.
    
    For a directory (HF checkpoint) sum all files.
    For a single file (GGUF) use its size directly.
    NUM_PARAMS = 35B for Qwen3.6-35B-A3B.
    """
    if path.is_dir():
        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        size_bytes = path.stat().st_size
    NUM_PARAMS = 35_000_000_000  # 35B
    return (size_bytes * 8) / NUM_PARAMS


def load_kld_json(model_dir: Path) -> dict | None:
    """Load KLD/NLL results from a ParoQuant export dir."""
    for p in [model_dir / "kld-smoke.json", model_dir / "kld.json"]:
        if p.exists():
            return json.loads(p.read_text())
    return None


# ── GGUF evaluation ───────────────────────────────────────────────────────────

def eval_gguf(gguf_path: str, llama_cpp_dir: str, cuda_id: int, ctx_size: int = 2048) -> dict:
    """Run wikitext2 perplexity via llama.cpp's perplexity tool."""
    llama_binary = Path(llama_cpp_dir) / "build" / "bin" / "llama-perplexity"
    if not llama_binary.exists():
        raise FileNotFoundError(f"llama-perplexity not found at {llama_binary}")

    wikitext_path = Path(llama_cpp_dir) / "wikitext2"
    if not wikitext_path.exists():
        raise FileNotFoundError(
            f"wikitext2 test file not found at {wikitext_path}. "
            "Run: python -c \"from datasets import load_dataset; ds=load_dataset('wikitext','wikitext-2-v1',split='test'); open('~/llama.cpp/wikitext2','w').write('\\n'.join(l.strip() for l in ds['text'] if l.strip()))\""
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

    cmd = [
        str(llama_binary),
        "-m", gguf_path,
        "-f", str(wikitext_path),
        "-c", str(ctx_size),
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    # Parse perplexity from output. Format varies; look for "Final estimate: PPL = X.XXXX"
    # or the last comma-separated value in bracket format [...,][N]PPL,
    output = result.stdout
    # Try "Final estimate: PPL = X.XXXX" or similar
    for line in output.splitlines():
        line = line.strip()
        if "final estimate" in line.lower() and "ppl" in line.lower():
            for part in line.replace("=", " ").replace(",", " ").split():
                try:
                    val = float(part)
                    if 1 < val < 1000:
                        return {"ppl_wikitext2": round(val, 4)}
                except ValueError:
                    continue
    # Fallback: extract last number in range [1, 1000] from full output
    import re
    numbers = re.findall(r'(\d+\.\d+)', output)
    candidates = [float(n) for n in numbers if 1 < float(n) < 1000]
    if candidates:
        return {"ppl_wikitext2": round(candidates[-1], 4)}

    raise RuntimeError(f"Could not parse perplexity from output:\n{output[-500:]}")


# ── ParoQuant / HF evaluation ─────────────────────────────────────────────────

def _patch_qwen3moe_config():
    """Patch transformers to properly load Qwen3.5 MoE checkpoints.
    
    Issues addressed:
    1. model_type qwen3_5_moe is not in CONFIG_MAPPING -> patch to qwen3_moe
    2. get_text_config() returns a dict -> causes GenerationConfig.to_dict() to fail
    3. vocab_size missing from top-level config -> read from text_config dict
    """
    import json as _json
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    class PatchedQwen3MoeConfig(Qwen3MoeConfig):
        """Qwen3MoeConfig with fixed get_text_config and auto vocab_size from text_config."""
        def __init__(self, **kwargs):
            # Ensure vocab_size is propagated from text_config if missing
            tc = kwargs.get("text_config", {})
            if isinstance(tc, dict) and "vocab_size" in tc:
                if "vocab_size" not in kwargs:
                    kwargs["vocab_size"] = tc["vocab_size"]
            super().__init__(**kwargs)

        def get_text_config(self, decoder=False):
            # Always return self — this model is decoder-only (no separate text encoder)
            return self

    # Register patched config for qwen3_moe model type
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING
    if "qwen3_moe" not in CONFIG_MAPPING._extra_content:
        CONFIG_MAPPING._extra_content["qwen3_moe"] = PatchedQwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    MODEL_MAPPING[PatchedQwen3MoeConfig] = Qwen3MoeForCausalLM


@lru_cache(maxsize=1)
def _cached_hf_model(model_dir: str, device: str):
    """Load and cache the ParoQuant model once per path."""
    import torch
    _patch_qwen3moe_config()

    # Use AutoConfig which picks up the registered PatchedQwen3MoeConfig
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # Ensure get_text_config is patched (belt-and-suspenders)
    cfg.get_text_config = lambda decoder=False: cfg

    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    return Qwen3MoeForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        device_map=device,
        dtype=torch.float16,
        trust_remote_code=True,
    ).eval()


def eval_hf_paroquant(model_dir: str, cuda_id: int, ctx_size: int = 2048) -> dict:
    """Run wikitext2 perplexity on wikitext2 test via transformers.
    
    Uses sliding-window perplexity over concatenated text (standard benchmark approach).
    """
    import torch
    from transformers import AutoTokenizer

    device = f"cuda:{cuda_id}"

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    text_lines = [l.strip() for l in ds["text"] if l.strip()]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = _cached_hf_model(model_dir, device)

    # Concatenate all lines and evaluate with sliding window
    full_text = "\n\n".join(text_lines)
    encodings = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=ctx_size * 4)
    input_ids = encodings["input_ids"].to(device)
    seqlen = min(ctx_size, input_ids.shape[1])
    
    stride = max(seqlen // 2, 1)
    losses = []
    n_chunks = 0
    for i in range(0, input_ids.shape[1] - seqlen + 1, stride):
        chunk = input_ids[:, i : i + seqlen]
        with torch.no_grad():
            outputs = model(input_ids=chunk, use_cache=False)
            logits = outputs.logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1),
                reduction="mean",
            )
            losses.append(loss.item())
            n_chunks += 1
            if n_chunks % 5 == 0:
                print(f"    chunk {n_chunks}/{input_ids.shape[1] // stride}, loss={loss.item():.4f}")

    ppl = math.exp(sum(losses) / len(losses))
    print(f"  {n_chunks} chunks evaluated")
    return {"ppl_wikitext2": round(ppl, 6)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare quantization quality")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--models-list", type=str)
    parser.add_argument("--llama-cpp", default=os.path.expanduser("~/llama.cpp"))
    parser.add_argument("--cuda", type=int, default=1, help="CUDA device ID")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size for PPL eval")
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    # Collect model paths
    if args.models:
        model_paths = [Path(p) for p in args.models]
    elif args.models_list:
        lines = Path(args.models_list).read_text().splitlines()
        model_paths = [Path(l.strip()) for l in lines
                       if l.strip() and not l.strip().startswith("#")]
    else:
        print("Error: specify --models or --models-list", file=sys.stderr)
        sys.exit(1)

    results = []
    for mp in model_paths:
        row = {"path": str(mp), "name": mp.name}
        print(f"\n{'='*60}\nEvaluating: {mp}")

        # BPW
        try:
            row["bpw"] = round(bytes_per_weight(mp), 3)
            print(f"  BPW: {row['bpw']:.3f}")
        except Exception as e:
            row["bpw"] = None
            print(f"  BPW: error — {e}")

        # PPL
        if str(mp).endswith(".gguf"):
            try:
                row.update(eval_gguf(str(mp), args.llama_cpp, args.cuda, args.ctx_size))
                print(f"  PPL (wikitext2, ctx={args.ctx_size}): {row['ppl_wikitext2']:.4f}")
            except Exception as e:
                row["ppl_wikitext2"] = None
                print(f"  PPL: error — {e}")
        else:
            try:
                row.update(eval_hf_paroquant(str(mp), args.cuda))
                print(f"  PPL (wikitext2): {row['ppl_wikitext2']:.4f}")
            except Exception as e:
                row["ppl_wikitext2"] = None
                print(f"  PPL: error — {e}")

        # KLD/NLL (ParoQuant dirs only)
        kld = load_kld_json(mp)
        if kld:
            row["kld_overall"] = kld.get("overall_kl", kld.get("kl_overall"))
            row["delta_nll"] = kld.get("delta_nll")
            print(f"  KL: {row.get('kld_overall')}, ΔNLL: {row.get('delta_nll')}")

        results.append(row)

    # Summary table
    print("\n" + "="*90)
    print(f"{'Model':<55} {'BPW':>6} {'PPL':>8} {'KL':>8} {'ΔNLL':>8}")
    print("-"*90)
    for r in results:
        bpw  = f"{r['bpw']:.3f}" if r.get("bpw") else "  N/A"
        ppl  = f"{r['ppl_wikitext2']:.4f}" if r.get("ppl_wikitext2") else "   N/A"
        kl   = f"{r['kld_overall']:.5f}" if r.get("kld_overall") else "    N/A"
        dnll = f"{r['delta_nll']:.5f}" if r.get("delta_nll") else "    N/A"
        print(f"{r['name']:<55} {bpw:>6} {ppl:>8} {kl:>8} {dnll:>8}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()