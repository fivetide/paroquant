#!/usr/bin/env python3
"""Small held-out KLD/NLL gate for ParoQuant checkpoints.

Loads the BF16/source model first, records logits for fixed token samples, then
loads the PARO checkpoint and compares distributions. This avoids keeping both
models resident on GPU at once.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

import paroquant.inference.backends.transformers.quantizer  # noqa: F401 registers quantizer
from paroquant.optim.util import get_calib_dataset


DEFAULT_REF = "/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
DEFAULT_VAL_MIX = "/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-val-64x2048.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref-model", default=DEFAULT_REF)
    p.add_argument("--quant-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--samples-per-source", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--sources", nargs="*", default=["wikitext2", DEFAULT_VAL_MIX])
    return p.parse_args()


def dtype_from_arg(value: str) -> torch.dtype:
    return torch.bfloat16 if value == "bf16" else torch.float16


def load_model(path: str, *, dtype: torch.dtype):
    return AutoModelForImageTextToText.from_pretrained(
        path,
        device_map="cuda",
        dtype=dtype,
        trust_remote_code=True,
    ).eval()


def source_label(source: str) -> str:
    if source == "wikitext2":
        return "wikitext2"
    return Path(source).stem


def collect_samples(tokenizer: Any, sources: list[str], *, n_samples: int, seq_len: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        split = "test" if source == "wikitext2" else "validation"
        tensors = get_calib_dataset(
            source,
            tokenizer=tokenizer,
            n_samples=n_samples,
            block_size=seq_len,
            seed=seed,
            split=split,
        )
        for i, ids in enumerate(tensors):
            rows.append({"source": source_label(source), "index": i, "input_ids": ids.long()})
    return rows


@torch.no_grad()
def compute_ref_logits(model: Any, samples: list[dict[str, Any]]) -> None:
    for row in samples:
        ids = row["input_ids"].unsqueeze(0).to("cuda")
        logits = model(input_ids=ids).logits.detach().to("cpu", dtype=torch.float16)
        row["ref_logits"] = logits
        torch.cuda.empty_cache()


@torch.no_grad()
def compare_quant(model: Any, samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, float]] = {}
    totals = {"tokens": 0.0, "kl_sum": 0.0, "ref_nll_sum": 0.0, "quant_nll_sum": 0.0}

    for row in samples:
        source = row["source"]
        by_source.setdefault(source, {"tokens": 0.0, "kl_sum": 0.0, "ref_nll_sum": 0.0, "quant_nll_sum": 0.0})

        ids = row["input_ids"].unsqueeze(0).to("cuda")
        ref_logits = row["ref_logits"].to("cuda", dtype=torch.float32)
        quant_logits = model(input_ids=ids).logits.detach().to(dtype=torch.float32)

        # Next-token positions only.
        labels = ids[:, 1:].contiguous()
        ref_lp = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        quant_lp = F.log_softmax(quant_logits[:, :-1, :], dim=-1)
        ref_p = ref_lp.exp()
        kl = (ref_p * (ref_lp - quant_lp)).sum(dim=-1)
        ref_nll = -ref_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        quant_nll = -quant_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        token_count = float(labels.numel())
        kl_sum = float(kl.sum().item())
        ref_nll_sum = float(ref_nll.sum().item())
        quant_nll_sum = float(quant_nll.sum().item())

        for bucket in (totals, by_source[source]):
            bucket["tokens"] += token_count
            bucket["kl_sum"] += kl_sum
            bucket["ref_nll_sum"] += ref_nll_sum
            bucket["quant_nll_sum"] += quant_nll_sum

        del ref_logits, quant_logits, ref_lp, quant_lp, ref_p, kl, ref_nll, quant_nll
        torch.cuda.empty_cache()

    def finalize(bucket: dict[str, float]) -> dict[str, float]:
        tokens = max(bucket["tokens"], 1.0)
        return {
            "tokens": int(bucket["tokens"]),
            "mean_kl_nats": bucket["kl_sum"] / tokens,
            "ref_nll_nats": bucket["ref_nll_sum"] / tokens,
            "quant_nll_nats": bucket["quant_nll_sum"] / tokens,
            "delta_nll_nats": (bucket["quant_nll_sum"] - bucket["ref_nll_sum"]) / tokens,
        }

    return {
        "overall": finalize(totals),
        "by_source": {k: finalize(v) for k, v in sorted(by_source.items())},
    }


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model)
    samples = collect_samples(
        tokenizer,
        args.sources,
        n_samples=args.samples_per_source,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    dtype = dtype_from_arg(args.dtype)
    print(f"Loaded {len(samples)} eval samples, seq_len={args.seq_len}")
    print("Loading reference model...")
    ref = load_model(args.ref_model, dtype=dtype)
    compute_ref_logits(ref, samples)
    del ref
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading quant model...")
    quant = load_model(args.quant_model, dtype=torch.float16)
    results = compare_quant(quant, samples)
    results.update(
        {
            "ref_model": args.ref_model,
            "quant_model": args.quant_model,
            "seq_len": args.seq_len,
            "samples_per_source": args.samples_per_source,
            "seed": args.seed,
            "sources": args.sources,
        }
    )

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["overall"], indent=2))
    for source, metrics in results["by_source"].items():
        print(source, json.dumps(metrics))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
