#!/usr/bin/env python3
"""Canonical quantization-quality evaluation for Qwen3.6-35B-A3B.

The canonical protocol is documented in docs/QUANTIZATION-QUALITY.md. In short:

* tokenize the held-out tx4/quality3 validation text with the original HF tokenizer;
* build overlapping 2048-token windows with stride 1023;
* score only llama.cpp-compatible second-half target positions;
* compare every candidate directly against the original BF16 HF model.

For HF/ParoQuant candidates the default mode caches scored BF16 reference logits
in CPU RAM, unloads the reference model, and then evaluates candidates on a GPU.
This avoids ParoQuant custom-kernel issues on non-zero CUDA devices.  A live
two-GPU mode is also available for compatible HF models.  For GGUF candidates it
writes a llama.cpp KLD-base stream from BF16 HF logits either to a FIFO (no large
disk cache) or to a reusable disk file, and then runs llama-perplexity against
that original-model reference.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer


DEFAULT_REF = "/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
DEFAULT_VAL_JSONL = "/models/qwen36-calibration/qwen36-paro-tx4-codebreadth-chotto-4096-val-64x2048-optfix.jsonl"
DEFAULT_OUTPUT_DIR = "/models/eval-results/quant-quality-tx4-quality3-canonical"
DEFAULT_LLAMA_PERPLEXITY = "/home/lhl/llama.cpp/build/bin/llama-perplexity"
DEFAULT_LLAMA_CPP_DIR = "/home/lhl/llama.cpp"
DEFAULT_GGUF_F16 = "/models/gguf/Qwen3.6-35B-A3B-F16.gguf"
DEFAULT_GGUF_Q4KM = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PARO_E1 = "/models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e1"
DEFAULT_PARO_E5 = "/models/qwen36-quant/Qwen3.6-35B-A3B-PARO-full4096-e5"
DEFAULT_PARAM_DENOMINATOR = 35_000_000_000


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


@dataclass(frozen=True)
class EvalLayout:
    ctx: int
    stride: int
    first_logit_index: int
    first_target_index: int
    rows_per_window: int

    @property
    def prompt_tokens_per_window(self) -> int:
        return self.ctx


class MetricAccumulator:
    def __init__(self) -> None:
        self.tokens = 0
        self.ref_nll_sum = 0.0
        self.cand_nll_sum = 0.0
        self.kl_sum = 0.0
        self.delta_p_sum = 0.0
        self.delta_p2_sum = 0.0
        self.top1_same = 0
        self.max_kl = 0.0
        self.kl_values_for_percentiles: list[float] = []

    @torch.no_grad()
    def update(self, ref_logits: torch.Tensor, cand_logits: torch.Tensor, labels: torch.Tensor, *, keep_percentiles: bool) -> None:
        """Accumulate metrics for one [T, V] logits block."""
        if ref_logits.ndim == 3:
            ref_logits = ref_logits[0]
        if cand_logits.ndim == 3:
            cand_logits = cand_logits[0]
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
        cand_lp = F.log_softmax(cand_logits.float(), dim=-1)
        ref_p = ref_lp.exp()

        kl = (ref_p * (ref_lp - cand_lp)).sum(dim=-1)
        ref_nll = -ref_lp.gather(-1, labels[:, None]).squeeze(-1)
        cand_nll = -cand_lp.gather(-1, labels[:, None]).squeeze(-1)
        delta_p = torch.exp(-cand_nll) - torch.exp(-ref_nll)
        same_top = (ref_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1)).sum()

        nt = int(labels.numel())
        self.tokens += nt
        self.ref_nll_sum += float(ref_nll.sum().item())
        self.cand_nll_sum += float(cand_nll.sum().item())
        self.kl_sum += float(kl.sum().item())
        self.delta_p_sum += float(delta_p.sum().item())
        self.delta_p2_sum += float((delta_p * delta_p).sum().item())
        self.top1_same += int(same_top.item())
        self.max_kl = max(self.max_kl, float(kl.max().item()))
        if keep_percentiles:
            self.kl_values_for_percentiles.extend(float(x) for x in kl.detach().cpu().tolist())

    @torch.no_grad()
    def update_baseline(self, ref_logits: torch.Tensor, labels: torch.Tensor) -> None:
        if ref_logits.ndim == 3:
            ref_logits = ref_logits[0]
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
        ref_nll = -ref_lp.gather(-1, labels[:, None]).squeeze(-1)
        nt = int(labels.numel())
        self.tokens += nt
        self.ref_nll_sum += float(ref_nll.sum().item())
        self.cand_nll_sum += float(ref_nll.sum().item())
        self.top1_same += nt

    def finalize(self) -> dict[str, Any]:
        t = max(self.tokens, 1)
        ref_mean_nll = self.ref_nll_sum / t
        cand_mean_nll = self.cand_nll_sum / t
        out: dict[str, Any] = {
            "scored_tokens": self.tokens,
            "ref_mean_nll_nats": ref_mean_nll,
            "mean_nll_nats": cand_mean_nll,
            "ppl": math.exp(cand_mean_nll),
            "ref_ppl": math.exp(ref_mean_nll),
            "delta_nll_nats": cand_mean_nll - ref_mean_nll,
            "mean_kl_nats": self.kl_sum / t,
            "max_kl_nats": self.max_kl,
            "mean_delta_p_pct": 100.0 * self.delta_p_sum / t,
            "rms_delta_p_pct": 100.0 * math.sqrt(max(self.delta_p2_sum / t, 0.0)),
            "top1_agreement_pct": 100.0 * self.top1_same / t,
        }
        if self.kl_values_for_percentiles:
            arr = np.asarray(self.kl_values_for_percentiles, dtype=np.float64)
            out["kl_percentiles"] = {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "p99_9": float(np.percentile(arr, 99.9)),
            }
        return out


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_model_spec(value: str) -> ModelSpec:
    if "=" in value:
        name, path = value.split("=", 1)
        return ModelSpec(name=name.strip(), path=path.strip())
    path = value.strip()
    return ModelSpec(name=Path(path).name, path=path)


def existing_default_specs() -> tuple[list[ModelSpec], list[ModelSpec]]:
    paro: list[ModelSpec] = []
    gguf: list[ModelSpec] = []
    if Path(DEFAULT_PARO_E1).exists():
        paro.append(ModelSpec("PARO full4096-e1", DEFAULT_PARO_E1))
    if Path(DEFAULT_PARO_E5).exists():
        paro.append(ModelSpec("PARO full4096-e5", DEFAULT_PARO_E5))
    if Path(DEFAULT_GGUF_Q4KM).exists():
        gguf.append(ModelSpec("GGUF UD-Q4_K_M", DEFAULT_GGUF_Q4KM))
    return paro, gguf


def parse_args() -> argparse.Namespace:
    default_paro, default_gguf = existing_default_specs()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--ref-model", default=DEFAULT_REF, help="Original BF16 HF reference model")
    p.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL, help="Held-out JSONL with text/content fields")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ctx", type=int, default=2048, help="Evaluation window/context length")
    p.add_argument("--stride", type=int, default=None, help="Window stride; default is ctx - 1 - ctx//2")
    p.add_argument("--max-windows", type=int, default=None, help="Limit windows for smoke tests")
    p.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-kl-percentiles", action="store_true", help="Store KL percentiles for HF/Paro rows")

    p.add_argument("--skip-hf", action="store_true", help="Skip BF16 baseline and HF/Paro candidates")
    p.add_argument("--paro-model", action="append", default=None,
                   help="HF/Paro candidate, either NAME=PATH or PATH. Repeatable. If omitted, existing default PARO exports are used.")
    p.add_argument("--hf-compare-mode", choices=["cache", "live"], default="cache",
                   help="cache stores scored BF16 logits in CPU RAM then unloads the ref; live keeps ref and candidate on GPUs simultaneously")
    p.add_argument("--ref-device", default="cuda:0", help="Device for original BF16 HF in HF/Paro eval")
    p.add_argument("--candidate-device", default="cuda:0", help="Device for HF/Paro candidates; use cuda:1+ only if kernels support non-zero devices")
    p.add_argument("--ref-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--candidate-dtype", choices=["bf16", "fp16"], default="fp16")

    p.add_argument("--skip-gguf", action="store_true", help="Skip GGUF candidates")
    p.add_argument("--gguf-model", action="append", default=None,
                   help="GGUF candidate, either NAME=PATH or PATH. Repeatable. If omitted, existing default GGUF quantizations are used.")
    p.add_argument("--gguf-f16-sanity", default=DEFAULT_GGUF_F16 if Path(DEFAULT_GGUF_F16).exists() else None,
                   help="Optional F16 GGUF sanity-check candidate vs original BF16 HF")
    p.add_argument("--skip-gguf-f16-sanity", action="store_true")
    p.add_argument("--gguf-reference-mode", choices=["fifo", "disk"], default="fifo",
                   help="fifo avoids a huge logits file; disk builds a reusable BF16-HF KLD base")
    p.add_argument("--reference-kld", default=None, help="Path for disk-mode BF16-HF KLD base")
    p.add_argument("--keep-reference-kld", action="store_true", help="Keep disk-mode KLD base after eval")
    p.add_argument("--cleanup-reference-kld", action="store_true", help="Remove disk-mode KLD base after eval")
    p.add_argument("--force-reference-kld", action="store_true", help="Rebuild disk-mode KLD even if it exists")
    p.add_argument("--gguf-ref-device", default="cuda:0", help="Device for BF16 HF writer in GGUF FIFO/disk KLD generation")
    p.add_argument("--gguf-cuda-visible-devices", default="1", help="CUDA_VISIBLE_DEVICES for llama.cpp GGUF subprocesses")
    p.add_argument("--llama-perplexity", default=DEFAULT_LLAMA_PERPLEXITY)
    p.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR)
    p.add_argument("--llama-gpu-layers", type=int, default=999)
    p.add_argument("--llama-extra-arg", action="append", default=[], help="Extra arg passed to llama-perplexity. Repeatable.")

    p.add_argument("--bpw-denominator-params", type=int, default=DEFAULT_PARAM_DENOMINATOR)
    p.add_argument("--dry-run", action="store_true", help="Prepare tokenization/layout and write metadata only")
    args = p.parse_args()
    if args.paro_model is None:
        args.paro_model = [f"{s.name}={s.path}" for s in default_paro]
    if args.gguf_model is None:
        args.gguf_model = [f"{s.name}={s.path}" for s in default_gguf]
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def maybe_import_paroquant() -> None:
    try:
        import paroquant.inference.backends.transformers.quantizer  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        log(f"[warn] Could not import ParoQuant quantizer registration: {exc}")


def load_hf_model(path: str, *, dtype: torch.dtype, device: str):
    return AutoModelForImageTextToText.from_pretrained(
        path,
        device_map={"": device},
        dtype=dtype,
        trust_remote_code=True,
    ).eval()


def read_validation_jsonl(path: Path) -> tuple[str, dict[str, Any]]:
    rows: list[str] = []
    by_group: dict[str, dict[str, int]] = {}
    by_source: dict[str, int] = {}
    raw_rows = 0
    empty_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_rows += 1
            row = json.loads(line)
            text = str(row.get("text") or row.get("content") or "").strip()
            if not text:
                empty_rows += 1
                continue
            group = str(row.get("group") or "unknown")
            source = str(row.get("source") or "unknown")
            rows.append(text)
            by_source[source] = by_source.get(source, 0) + 1
            bucket = by_group.setdefault(group, {"rows": 0, "chars": 0, "qwen_tokens_meta": 0})
            bucket["rows"] += 1
            bucket["chars"] += len(text)
            bucket["qwen_tokens_meta"] += int(row.get("qwen_tokens") or 0)
    text = "\n\n".join(rows) + "\n"
    meta = {
        "source_jsonl": str(path),
        "raw_rows": raw_rows,
        "used_rows": len(rows),
        "empty_rows": empty_rows,
        "chars": len(text),
        "by_group": dict(sorted(by_group.items())),
        "by_source_top": sorted(by_source.items(), key=lambda kv: kv[1], reverse=True)[:100],
    }
    return text, meta


def make_layout(ctx: int, stride: int | None) -> EvalLayout:
    first = ctx // 2
    rows = ctx - 1 - first
    if rows <= 0:
        raise ValueError(f"ctx={ctx} is too small")
    return EvalLayout(
        ctx=ctx,
        stride=rows if stride is None else stride,
        first_logit_index=first,
        first_target_index=first + 1,
        rows_per_window=rows,
    )


def make_windows(token_ids: list[int], layout: EvalLayout, max_windows: int | None) -> np.ndarray:
    if len(token_ids) < layout.ctx:
        raise ValueError(f"Need at least {layout.ctx} tokens; got {len(token_ids)}")
    starts = list(range(0, len(token_ids) - layout.ctx + 1, layout.stride))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError("No evaluation windows produced")
    arr = np.empty((len(starts), layout.ctx), dtype=np.int32)
    ids = np.asarray(token_ids, dtype=np.int64)
    for i, s in enumerate(starts):
        arr[i] = ids[s : s + layout.ctx].astype(np.int32, copy=False)
    return arr


def directory_weight_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    # Export directories often contain the active weights as model.safetensors
    # plus backup files such as model.orig.safetensors.  BPW should describe the
    # artifact needed for inference, not backups left beside it.
    active_model = path / "model.safetensors"
    if active_model.exists():
        return active_model.stat().st_size
    total = 0
    suffixes = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".orig.safetensors") or name.endswith(".bak.safetensors"):
                continue
            p = Path(root) / name
            if p.suffix in suffixes:
                total += p.stat().st_size
    return total


def bpw_for_path(path: str, denominator: int) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    n = directory_weight_bytes(p)
    return 8.0 * n / denominator if n else None


def safetensors_tensor_byte_map(path: Path) -> dict[str, int]:
    """Read safetensors header and return tensor payload sizes without loading tensors."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    out: dict[str, int] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[key] = int(end) - int(start)
    return out


def packed_estimate_bytes(path: Path) -> int:
    """Estimate deployable packed bytes, excluding duplicate fp16 fallback tensors.

    For ParoQuant safetensors, quantized modules currently keep both qweight
    tensors and duplicate `.weight` fallback tensors.  The active HF bridge
    artifact needs those today, but a pure packed runtime/export would not.
    """
    if path.is_file():
        return path.stat().st_size
    active_model = path / "model.safetensors"
    if not active_model.exists():
        return directory_weight_bytes(path)
    try:
        sizes = safetensors_tensor_byte_map(active_model)
    except Exception:
        return active_model.stat().st_size
    quant_prefixes = {k[: -len(".qweight")] for k in sizes if k.endswith(".qweight")}
    fallback = sum(
        b for k, b in sizes.items()
        if k.endswith(".weight") and k[: -len(".weight")] in quant_prefixes
    )
    total_tensor_bytes = sum(sizes.values())
    return max(total_tensor_bytes - fallback, 0)


def packed_bpw_for_path(path: str, denominator: int) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    n = packed_estimate_bytes(p)
    return 8.0 * n / denominator if n else None


def base_row_common(name: str, path: str, *, kind: str, reference: str, layout: EvalLayout, windows: np.ndarray, denominator: int) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "path": path,
        "reference": reference,
        "bpw": bpw_for_path(path, denominator),
        "packed_bpw_estimate": packed_bpw_for_path(path, denominator),
        "ctx": layout.ctx,
        "stride": layout.stride,
        "windows": int(windows.shape[0]),
        "prompt_tokens": int(windows.shape[0] * layout.ctx),
        "scored_tokens": int(windows.shape[0] * layout.rows_per_window),
    }


def encode_kld_rows(out: Any, logits_rows: torch.Tensor, labels: torch.Tensor, n_vocab: int, sub_rows: int) -> tuple[float, float, int]:
    """Write llama.cpp KLD-base rows for [T, V] logits and return NLL stats."""
    total_nll = 0.0
    total_nll2 = 0.0
    total_rows = 0
    rows = int(logits_rows.shape[0])
    nv = 2 * ((n_vocab + 1) // 2) + 4
    for rs in range(0, rows, sub_rows):
        re = min(rs + sub_rows, rows)
        sub = logits_rows[rs:re].float()
        tgt = labels[rs:re]
        bsz = int(sub.shape[0])

        max_logit = sub.max(dim=-1).values
        min_actual = sub.min(dim=-1).values
        min_logit = torch.maximum(min_actual, max_logit - 16.0)
        lse = torch.logsumexp(sub, dim=-1)
        scale = (max_logit - min_logit) / 65535.0
        min_log_prob = min_logit - lse

        tgt_logits = sub.gather(1, tgt[:, None]).squeeze(1)
        nll = lse - tgt_logits
        nll_cpu = nll.detach().cpu().double().numpy()
        total_nll += float(nll_cpu.sum())
        total_nll2 += float((nll_cpu * nll_cpu).sum())
        total_rows += bsz

        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q_float = torch.where(
            (scale[:, None] > 0) & (sub > min_logit[:, None]),
            torch.round((sub - min_logit[:, None]) / safe_scale[:, None]),
            torch.zeros_like(sub),
        )
        q_i32 = torch.clamp(q_float, 0, 65535).to(torch.int32)
        q_cpu = q_i32.cpu().numpy().astype("<u2", copy=False)
        scale_cpu = scale.detach().cpu().numpy().astype("<f4", copy=False)
        mlp_cpu = min_log_prob.detach().cpu().numpy().astype("<f4", copy=False)

        buf = np.zeros((bsz, nv), dtype="<u2")
        header_f = buf[:, :4].view("<f4").reshape(bsz, 2)
        header_f[:, 0] = scale_cpu
        header_f[:, 1] = mlp_cpu
        buf[:, 4 : 4 + n_vocab] = q_cpu
        out.write(buf.tobytes(order="C"))

        del sub, tgt, max_logit, min_actual, min_logit, lse, scale, min_log_prob
        del tgt_logits, nll, q_float, q_i32, q_cpu, scale_cpu, mlp_cpu, buf, header_f
    return total_nll, total_nll2, total_rows


@torch.no_grad()
def write_bf16_hf_kld_stream(
    out: Any,
    *,
    ref_model_path: str,
    windows: np.ndarray,
    layout: EvalLayout,
    device: str,
    dtype: torch.dtype,
    sub_rows: int = 16,
) -> dict[str, Any]:
    n_chunk = int(windows.shape[0])
    n_ctx = int(layout.ctx)
    log(f"[kld-writer] loading original BF16 HF reference on {device}: {ref_model_path}")
    model = load_hf_model(ref_model_path, dtype=dtype, device=device)
    # Use the model head size as the authoritative vocab size for logits/KLD.
    with torch.no_grad():
        probe = torch.from_numpy(windows[0:1]).to(device=device, dtype=torch.long)
        probe_logits = model(input_ids=probe, use_cache=False).logits
        n_vocab = int(probe_logits.shape[-1])
        del probe, probe_logits
        torch.cuda.empty_cache()

    out.write(b"_logits_")
    out.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
    out.write(np.asarray(windows, dtype="<i4", order="C").tobytes(order="C"))
    out.flush()

    total_nll = 0.0
    total_nll2 = 0.0
    total_rows = 0
    max_mem = 0.0
    t0 = time.time()
    for wi in range(n_chunk):
        ids = torch.from_numpy(windows[wi].astype(np.int64, copy=False)).unsqueeze(0).to(device=device)
        output = model(input_ids=ids, use_cache=False)
        logits_rows = output.logits[0, layout.first_logit_index : n_ctx - 1, :]
        labels = ids[0, layout.first_target_index : n_ctx]
        nll, nll2, rows = encode_kld_rows(out, logits_rows, labels, n_vocab=n_vocab, sub_rows=sub_rows)
        total_nll += nll
        total_nll2 += nll2
        total_rows += rows
        out.flush()
        if torch.cuda.is_available():
            max_mem = max(max_mem, torch.cuda.max_memory_allocated(device=device) / 1024**3)
        elapsed = time.time() - t0
        log(
            f"[kld-writer] window {wi+1:4d}/{n_chunk} ref_ppl={math.exp(total_nll/max(total_rows,1)):.6f} "
            f"elapsed={elapsed/60:.1f}m eta={(elapsed/(wi+1)*(n_chunk-wi-1))/60:.1f}m max_mem={max_mem:.1f}GiB"
        )
        del ids, output, logits_rows, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    payload = {
        "ref_model": ref_model_path,
        "n_ctx": n_ctx,
        "n_vocab": n_vocab,
        "n_chunk": n_chunk,
        "first_logit_index": layout.first_logit_index,
        "first_target_index": layout.first_target_index,
        "rows_per_window": layout.rows_per_window,
        "scored_tokens": total_rows,
        "ref_mean_nll_nats": total_nll / max(total_rows, 1),
        "ref_ppl": math.exp(total_nll / max(total_rows, 1)),
        "max_mem_gib": max_mem,
        "elapsed_seconds": elapsed,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return payload


def estimate_kld_bytes(windows: np.ndarray, layout: EvalLayout, n_vocab_hint: int = 248_320) -> int:
    nv = 2 * ((n_vocab_hint + 1) // 2) + 4
    header = 8 + 4 + 4 + 4 + windows.size * 4
    body = int(windows.shape[0]) * layout.rows_per_window * nv * 2
    return header + body


def parse_llama_log(log_path: Path) -> dict[str, Any]:
    txt = log_path.read_text(errors="replace") if log_path.exists() else ""

    def first(pattern: str) -> float | None:
        m = re.search(pattern, txt, re.MULTILINE)
        return float(m.group(1)) if m else None

    def last(pattern: str) -> float | None:
        vals = [float(m.group(1)) for m in re.finditer(pattern, txt, re.MULTILINE)]
        return vals[-1] if vals else None

    return {
        "ppl": first(r"Mean PPL\(Q\)\s*:\s*([-+0-9.eE]+)"),
        "ref_ppl": first(r"Mean PPL\(base\)\s*:\s*([-+0-9.eE]+)"),
        "delta_nll_nats": first(r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-+0-9.eE]+)"),
        "ppl_ratio": first(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([-+0-9.eE]+)"),
        "ppl_delta": first(r"Mean PPL\(Q\)-PPL\(base\)\s*:\s*([-+0-9.eE]+)"),
        "mean_kl_nats": first(r"Mean\s+KLD:\s*([-+0-9.eE]+)"),
        "max_kl_nats": first(r"Maximum KLD:\s*([-+0-9.eE]+)"),
        "rms_delta_p_pct": first(r"RMS Δp\s*:\s*([-+0-9.eE]+)"),
        "mean_delta_p_pct": first(r"Mean\s+Δp:\s*([-+0-9.eE]+)"),
        "top1_agreement_pct": first(r"Same top p:\s*([-+0-9.eE]+)"),
        "final_estimate_ppl": last(r"Final estimate:\s*PPL\s*=\s*([-+0-9.eE]+)"),
    }


def run_llama_perplexity(
    *,
    llama_perplexity: str,
    llama_cpp_dir: str,
    model_path: str,
    text_path: Path,
    kld_base_path: Path,
    log_path: Path,
    ctx: int,
    ngl: int,
    cuda_visible_devices: str,
    extra_args: list[str],
) -> int:
    cmd = [
        llama_perplexity,
        "-m", model_path,
        "-f", str(text_path),
        "-c", str(ctx),
        "-ngl", str(ngl),
        "--kl-divergence-base", str(kld_base_path),
        "--kl-divergence",
    ] + extra_args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    log(f"[llama.cpp] CUDA_VISIBLE_DEVICES={cuda_visible_devices} {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=llama_cpp_dir, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def run_gguf_with_fifo(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
    windows: np.ndarray,
    layout: EvalLayout,
    text_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fifo_path = out_dir / f"reference_{safe_name(spec.name)}.fifo"
    log_path = out_dir / f"gguf_{safe_name(spec.name)}.log"
    if fifo_path.exists():
        fifo_path.unlink()
    os.mkfifo(fifo_path)

    cmd = [
        args.llama_perplexity,
        "-m", spec.path,
        "-f", str(text_path),
        "-c", str(layout.ctx),
        "-ngl", str(args.llama_gpu_layers),
        "--kl-divergence-base", str(fifo_path),
        "--kl-divergence",
    ] + list(args.llama_extra_arg or [])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gguf_cuda_visible_devices
    log(f"[gguf:fifo] starting llama.cpp for {spec.name} on CUDA_VISIBLE_DEVICES={args.gguf_cuda_visible_devices}")
    log(f"[gguf:fifo] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=args.llama_cpp_dir, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
        writer_meta: dict[str, Any] | None = None
        try:
            with fifo_path.open("wb", buffering=64 * 1024 * 1024) as out:
                writer_meta = write_bf16_hf_kld_stream(
                    out,
                    ref_model_path=args.ref_model,
                    windows=windows,
                    layout=layout,
                    device=args.gguf_ref_device,
                    dtype=dtype_from_name(args.ref_dtype),
                )
            rc = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise
        finally:
            try:
                fifo_path.unlink()
            except FileNotFoundError:
                pass
    if rc != 0:
        raise RuntimeError(f"llama-perplexity failed for {spec.name} with exit code {rc}; see {log_path}")
    metrics = parse_llama_log(log_path)
    row = base_row_common(
        spec.name,
        spec.path,
        kind="GGUF/llama.cpp",
        reference="Original BF16 HF",
        layout=layout,
        windows=windows,
        denominator=args.bpw_denominator_params,
    )
    row.update(metrics)
    row["log"] = str(log_path)
    row["reference_mode"] = "fifo"
    row["reference_writer"] = writer_meta
    if row.get("ppl") is not None:
        row["mean_nll_nats"] = math.log(float(row["ppl"]))
    if row.get("ref_ppl") is not None:
        row["ref_mean_nll_nats"] = math.log(float(row["ref_ppl"]))
    return row


def build_reference_kld_disk(args: argparse.Namespace, windows: np.ndarray, layout: EvalLayout, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    kld_path = Path(args.reference_kld) if args.reference_kld else out_dir / "original_bf16_hf_reference.kld"
    meta_path = kld_path.with_suffix(kld_path.suffix + ".meta.json")
    if kld_path.exists() and not args.force_reference_kld:
        log(f"[gguf:disk] reusing existing reference KLD: {kld_path}")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"reused": True, "path": str(kld_path)}
        return kld_path, meta

    est = estimate_kld_bytes(windows, layout)
    free = shutil.disk_usage(kld_path.parent).free
    log(f"[gguf:disk] estimated BF16-HF KLD cache size: {est/1024**3:.2f} GiB; free: {free/1024**3:.2f} GiB")
    if free < est + 5 * 1024**3:
        raise RuntimeError(
            f"Not enough free space for disk-mode KLD cache. Need about {(est + 5*1024**3)/1024**3:.1f} GiB, "
            f"free is {free/1024**3:.1f} GiB. Use --gguf-reference-mode fifo or free disk space."
        )
    tmp = kld_path.with_suffix(kld_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb", buffering=64 * 1024 * 1024) as out:
        meta = write_bf16_hf_kld_stream(
            out,
            ref_model_path=args.ref_model,
            windows=windows,
            layout=layout,
            device=args.gguf_ref_device,
            dtype=dtype_from_name(args.ref_dtype),
        )
    os.replace(tmp, kld_path)
    meta.update({"path": str(kld_path), "file_size_bytes": kld_path.stat().st_size, "file_size_gib": kld_path.stat().st_size / 1024**3})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return kld_path, meta


def run_gguf_disk(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
    windows: np.ndarray,
    layout: EvalLayout,
    text_path: Path,
    out_dir: Path,
    kld_path: Path,
    kld_meta: dict[str, Any],
) -> dict[str, Any]:
    log_path = out_dir / f"gguf_{safe_name(spec.name)}.log"
    rc = run_llama_perplexity(
        llama_perplexity=args.llama_perplexity,
        llama_cpp_dir=args.llama_cpp_dir,
        model_path=spec.path,
        text_path=text_path,
        kld_base_path=kld_path,
        log_path=log_path,
        ctx=layout.ctx,
        ngl=args.llama_gpu_layers,
        cuda_visible_devices=args.gguf_cuda_visible_devices,
        extra_args=list(args.llama_extra_arg or []),
    )
    if rc != 0:
        raise RuntimeError(f"llama-perplexity failed for {spec.name} with exit code {rc}; see {log_path}")
    metrics = parse_llama_log(log_path)
    row = base_row_common(
        spec.name,
        spec.path,
        kind="GGUF/llama.cpp",
        reference="Original BF16 HF",
        layout=layout,
        windows=windows,
        denominator=args.bpw_denominator_params,
    )
    row.update(metrics)
    row["log"] = str(log_path)
    row["reference_mode"] = "disk"
    row["reference_kld"] = str(kld_path)
    row["reference_writer"] = kld_meta
    if row.get("ppl") is not None:
        row["mean_nll_nats"] = math.log(float(row["ppl"]))
    if row.get("ref_ppl") is not None:
        row["ref_mean_nll_nats"] = math.log(float(row["ref_ppl"]))
    return row


@torch.no_grad()
def eval_hf_baseline(args: argparse.Namespace, windows: np.ndarray, layout: EvalLayout) -> dict[str, Any]:
    log(f"[hf] loading original BF16 HF baseline on {args.ref_device}")
    ref = load_hf_model(args.ref_model, dtype=dtype_from_name(args.ref_dtype), device=args.ref_device)
    acc = MetricAccumulator()
    t0 = time.time()
    for wi in range(windows.shape[0]):
        ids = torch.from_numpy(windows[wi].astype(np.int64, copy=False)).unsqueeze(0).to(args.ref_device)
        output = ref(input_ids=ids, use_cache=False)
        logits = output.logits[:, layout.first_logit_index : layout.ctx - 1, :]
        labels = ids[0, layout.first_target_index : layout.ctx]
        acc.update_baseline(logits, labels)
        if (wi + 1) % 10 == 0 or wi + 1 == windows.shape[0]:
            cur = acc.finalize()
            log(f"[hf] baseline window {wi+1}/{windows.shape[0]} ppl={cur['ppl']:.6f} elapsed={(time.time()-t0)/60:.1f}m")
        del ids, output, logits, labels
        torch.cuda.empty_cache()
    row = base_row_common(
        "Original BF16 HF",
        args.ref_model,
        kind="HF/Transformers",
        reference="self",
        layout=layout,
        windows=windows,
        denominator=args.bpw_denominator_params,
    )
    row.update(acc.finalize())
    row["delta_nll_nats"] = 0.0
    row["mean_kl_nats"] = 0.0
    row["max_kl_nats"] = 0.0
    row["mean_delta_p_pct"] = 0.0
    row["rms_delta_p_pct"] = 0.0
    row["top1_agreement_pct"] = 100.0
    row["elapsed_seconds"] = time.time() - t0
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    return row


@torch.no_grad()
def build_hf_reference_cache(args: argparse.Namespace, windows: np.ndarray, layout: EvalLayout) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Compute baseline metrics and cache scored BF16 logits in CPU RAM."""
    log(f"[hf-cache] loading original BF16 HF reference on {args.ref_device}")
    ref = load_hf_model(args.ref_model, dtype=dtype_from_name(args.ref_dtype), device=args.ref_device)
    acc = MetricAccumulator()
    ref_cache: list[torch.Tensor] = []
    t0 = time.time()
    max_mem = 0.0
    for wi in range(windows.shape[0]):
        ids = torch.from_numpy(windows[wi].astype(np.int64, copy=False)).unsqueeze(0).to(args.ref_device)
        output = ref(input_ids=ids, use_cache=False)
        logits = output.logits[:, layout.first_logit_index : layout.ctx - 1, :]
        labels = ids[0, layout.first_target_index : layout.ctx]
        acc.update_baseline(logits, labels)
        # Cache only scored rows, fp16, on CPU.  For the full tx4/quality3
        # canonical run this is about the same size as the KLD cache (~60 GiB),
        # but it avoids disk and lets us unload the BF16 model before Paro eval.
        ref_cache.append(logits.detach().cpu().to(torch.float16).squeeze(0).contiguous())
        if torch.cuda.is_available():
            max_mem = max(max_mem, torch.cuda.max_memory_allocated(device=args.ref_device) / 1024**3)
        if (wi + 1) % 10 == 0 or wi + 1 == windows.shape[0]:
            cur = acc.finalize()
            cache_gib = sum(t.numel() * t.element_size() for t in ref_cache) / 1024**3
            log(
                f"[hf-cache] window {wi+1}/{windows.shape[0]} ppl={cur['ppl']:.6f} "
                f"cache={cache_gib:.1f}GiB elapsed={(time.time()-t0)/60:.1f}m"
            )
        del ids, output, logits, labels
        torch.cuda.empty_cache()
    row = base_row_common(
        "Original BF16 HF",
        args.ref_model,
        kind="HF/Transformers",
        reference="self",
        layout=layout,
        windows=windows,
        denominator=args.bpw_denominator_params,
    )
    row.update(acc.finalize())
    row["delta_nll_nats"] = 0.0
    row["mean_kl_nats"] = 0.0
    row["max_kl_nats"] = 0.0
    row["mean_delta_p_pct"] = 0.0
    row["rms_delta_p_pct"] = 0.0
    row["top1_agreement_pct"] = 100.0
    row["elapsed_seconds"] = time.time() - t0
    row["reference_cache_gib"] = sum(t.numel() * t.element_size() for t in ref_cache) / 1024**3
    row["max_mem_gib"] = max_mem
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    return row, ref_cache


@torch.no_grad()
def eval_hf_candidate_cached(
    args: argparse.Namespace,
    spec: ModelSpec,
    windows: np.ndarray,
    layout: EvalLayout,
    ref_cache: list[torch.Tensor],
) -> dict[str, Any]:
    maybe_import_paroquant()
    log(f"[hf-cache] loading candidate {spec.name} on {args.candidate_device}: {spec.path}")
    cand = load_hf_model(spec.path, dtype=dtype_from_name(args.candidate_dtype), device=args.candidate_device)
    acc = MetricAccumulator()
    t0 = time.time()
    max_mem = 0.0
    for wi in range(windows.shape[0]):
        ids = torch.from_numpy(windows[wi].astype(np.int64, copy=False)).unsqueeze(0).to(args.candidate_device)
        cand_out = cand(input_ids=ids, use_cache=False)
        cand_logits = cand_out.logits[:, layout.first_logit_index : layout.ctx - 1, :]
        ref_logits = ref_cache[wi].to(args.candidate_device, dtype=torch.float32, non_blocking=True)
        labels = ids[0, layout.first_target_index : layout.ctx]
        acc.update(ref_logits, cand_logits, labels, keep_percentiles=args.keep_kl_percentiles)
        if torch.cuda.is_available():
            max_mem = max(max_mem, torch.cuda.max_memory_allocated(device=args.candidate_device) / 1024**3)
        if (wi + 1) % 10 == 0 or wi + 1 == windows.shape[0]:
            cur = acc.finalize()
            log(
                f"[hf-cache] {spec.name} window {wi+1}/{windows.shape[0]} ppl={cur['ppl']:.6f} "
                f"kl={cur['mean_kl_nats']:.6f} top1={cur['top1_agreement_pct']:.3f}% elapsed={(time.time()-t0)/60:.1f}m"
            )
        del ids, cand_out, cand_logits, ref_logits, labels
        torch.cuda.empty_cache()
    row = base_row_common(
        spec.name,
        spec.path,
        kind="HF/ParoQuant",
        reference="Original BF16 HF",
        layout=layout,
        windows=windows,
        denominator=args.bpw_denominator_params,
    )
    row.update(acc.finalize())
    row["elapsed_seconds"] = time.time() - t0
    row["max_mem_gib"] = max_mem
    del cand
    gc.collect()
    torch.cuda.empty_cache()
    return row


@torch.no_grad()
def eval_hf_candidate(args: argparse.Namespace, spec: ModelSpec, windows: np.ndarray, layout: EvalLayout) -> dict[str, Any]:
    maybe_import_paroquant()
    log(f"[hf] loading original BF16 HF reference on {args.ref_device}")
    ref = load_hf_model(args.ref_model, dtype=dtype_from_name(args.ref_dtype), device=args.ref_device)
    log(f"[hf] loading candidate {spec.name} on {args.candidate_device}: {spec.path}")
    cand = load_hf_model(spec.path, dtype=dtype_from_name(args.candidate_dtype), device=args.candidate_device)
    acc = MetricAccumulator()
    t0 = time.time()
    for wi in range(windows.shape[0]):
        ids_ref = torch.from_numpy(windows[wi].astype(np.int64, copy=False)).unsqueeze(0).to(args.ref_device)
        ids_cand = ids_ref.to(args.candidate_device)
        ref_out = ref(input_ids=ids_ref, use_cache=False)
        cand_out = cand(input_ids=ids_cand, use_cache=False)
        ref_logits = ref_out.logits[:, layout.first_logit_index : layout.ctx - 1, :].to(args.candidate_device)
        cand_logits = cand_out.logits[:, layout.first_logit_index : layout.ctx - 1, :]
        labels = ids_cand[0, layout.first_target_index : layout.ctx]
        acc.update(ref_logits, cand_logits, labels, keep_percentiles=args.keep_kl_percentiles)
        if (wi + 1) % 10 == 0 or wi + 1 == windows.shape[0]:
            cur = acc.finalize()
            log(
                f"[hf] {spec.name} window {wi+1}/{windows.shape[0]} ppl={cur['ppl']:.6f} "
                f"kl={cur['mean_kl_nats']:.6f} top1={cur['top1_agreement_pct']:.3f}% elapsed={(time.time()-t0)/60:.1f}m"
            )
        del ids_ref, ids_cand, ref_out, cand_out, ref_logits, cand_logits, labels
        torch.cuda.empty_cache()
    row = base_row_common(
        spec.name,
        spec.path,
        kind="HF/ParoQuant",
        reference="Original BF16 HF",
        layout=layout,
        windows=windows,
        denominator=args.bpw_denominator_params,
    )
    row.update(acc.finalize())
    row["elapsed_seconds"] = time.time() - t0
    del ref, cand
    gc.collect()
    torch.cuda.empty_cache()
    return row


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def fmt_float(value: Any, digits: int = 6, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
    except Exception:
        return "—"
    if signed:
        return f"{f:+.{digits}f}"
    return f"{f:.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def fmt_bpw(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def write_results_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["results"]
    lines = [
        "# Canonical quantization-quality results",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        "## Protocol",
        "",
        f"- Reference: `{payload['reference_model']}`",
        f"- Validation source: `{payload['validation']['source_jsonl']}`",
        f"- Context/window length: `{payload['layout']['ctx']}` tokens",
        f"- Stride: `{payload['layout']['stride']}` tokens",
        f"- Scored target positions/window: `{payload['layout']['first_target_index']}..{payload['layout']['ctx'] - 1}` inclusive",
        f"- Windows: `{payload['layout']['windows']}`",
        f"- Scored tokens/model: `{payload['layout']['scored_tokens']:,}`",
        "- All candidate drift metrics are compared directly against the original BF16 HF model.",
        "- GGUF rows use llama.cpp's KLD stream/file format generated from original BF16 HF log-probabilities.",
        "",
        "## Full table",
        "",
        "| Model | Kind | Reference | Artifact BPW ↓ | Packed BPW est. ↓ | Windows | Prompt tokens | Scored tokens | PPL ↓ | Ref PPL | Mean NLL ↓ | Ref NLL | ΔNLL ↓ | KL nats ↓ | Max KL ↓ | RMS Δp % ↓ | Top-1 % ↑ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("name", "—")),
                    str(r.get("kind", "—")),
                    str(r.get("reference", "—")),
                    fmt_bpw(r.get("bpw")),
                    fmt_bpw(r.get("packed_bpw_estimate")),
                    fmt_int(r.get("windows")),
                    fmt_int(r.get("prompt_tokens")),
                    fmt_int(r.get("scored_tokens")),
                    fmt_float(r.get("ppl"), 4),
                    fmt_float(r.get("ref_ppl"), 4),
                    fmt_float(r.get("mean_nll_nats"), 6),
                    fmt_float(r.get("ref_mean_nll_nats"), 6),
                    fmt_float(r.get("delta_nll_nats"), 6, signed=True),
                    fmt_float(r.get("mean_kl_nats"), 6),
                    fmt_float(r.get("max_kl_nats"), 6),
                    fmt_float(r.get("rms_delta_p_pct"), 3),
                    fmt_float(r.get("top1_agreement_pct"), 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Column definitions",
            "",
            "- **PPL**: `exp(mean NLL)` over only the canonical scored target positions. Lower is better.",
            "- **Ref PPL**: PPL of the original BF16 HF model over the exact same scored positions.",
            "- **Mean NLL**: average `-log P_candidate(true next token | context)`, in nats.",
            "- **Ref NLL**: average original BF16 HF NLL, in nats.",
            "- **ΔNLL**: `Mean NLL - Ref NLL`; positive means worse true-token likelihood than the original model.",
            "- **KL nats**: mean `KL(P_original_BF16_HF || P_candidate)` over full next-token distributions. Lower is better.",
            "- **Max KL**: largest per-position KL observed. Useful for spotting outlier failures.",
            "- **RMS Δp %**: RMS percentage-point change in probability assigned to the true next token. Lower is better.",
            "- **Top-1 %**: percentage of positions where candidate and original BF16 HF choose the same argmax next token. Higher is better.",
            "- **Artifact BPW**: active model artifact bytes times 8 divided by the configured parameter denominator.",
            "- **Packed BPW est.**: estimate after removing duplicate fp16 fallback tensors from ParoQuant safetensors; for GGUF this is the same as artifact BPW.",
            "",
            "See `docs/QUANTIZATION-QUALITY.md` for the complete methodology and rationale.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False
    torch.set_grad_enabled(False)

    layout = make_layout(args.ctx, args.stride)
    if layout.stride != layout.rows_per_window:
        log(f"[warn] stride={layout.stride} differs from canonical rows_per_window={layout.rows_per_window}; scored positions may overlap or have gaps")

    text, val_meta = read_validation_jsonl(Path(args.val_jsonl))
    text_path = out_dir / "canonical_eval_text.txt"
    text_path.write_text(text, encoding="utf-8")

    log(f"[tokenize] loading tokenizer: {args.ref_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
    token_ids = tokenizer.encode(text, add_special_tokens=args.add_special_tokens)
    windows = make_windows(token_ids, layout, args.max_windows)

    token_meta = {
        **val_meta,
        "text_file": str(text_path),
        "add_special_tokens": args.add_special_tokens,
        "hf_token_count": len(token_ids),
        "ctx": layout.ctx,
        "stride": layout.stride,
        "first_logit_index": layout.first_logit_index,
        "first_target_index": layout.first_target_index,
        "rows_per_window": layout.rows_per_window,
        "windows": int(windows.shape[0]),
        "prompt_tokens": int(windows.shape[0] * layout.ctx),
        "scored_tokens": int(windows.shape[0] * layout.rows_per_window),
        "ignored_prefix_tokens_before_first_score": layout.first_target_index,
        "ignored_tail_tokens_after_last_full_window": len(token_ids) - (int(windows.shape[0] - 1) * layout.stride + layout.ctx),
        "estimated_kld_cache_bytes": estimate_kld_bytes(windows, layout),
        "estimated_kld_cache_gib": estimate_kld_bytes(windows, layout) / 1024**3,
    }
    (out_dir / "canonical_eval_meta.json").write_text(json.dumps(token_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(json.dumps(token_meta, indent=2, ensure_ascii=False))

    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_model": args.ref_model,
        "validation": val_meta,
        "text_file": str(text_path),
        "layout": token_meta,
        "bpw_denominator_params": args.bpw_denominator_params,
        "results": [],
    }

    if args.dry_run:
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_results_markdown(out_dir / "RESULTS.md", payload)
        log(f"[dry-run] wrote metadata to {out_dir}")
        return 0

    results: list[dict[str, Any]] = []

    if not args.skip_hf:
        if args.hf_compare_mode == "cache":
            baseline_row, ref_cache = build_hf_reference_cache(args, windows, layout)
            results.append(baseline_row)
            payload["results"] = results
            (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            write_results_markdown(out_dir / "RESULTS.md", payload)
            for raw in args.paro_model or []:
                spec = parse_model_spec(raw)
                if not Path(spec.path).exists():
                    log(f"[warn] skipping missing HF/Paro candidate: {spec.path}")
                    continue
                results.append(eval_hf_candidate_cached(args, spec, windows, layout, ref_cache))
                payload["results"] = results
                (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                write_results_markdown(out_dir / "RESULTS.md", payload)
            del ref_cache
            gc.collect()
        else:
            results.append(eval_hf_baseline(args, windows, layout))
            for raw in args.paro_model or []:
                spec = parse_model_spec(raw)
                if not Path(spec.path).exists():
                    log(f"[warn] skipping missing HF/Paro candidate: {spec.path}")
                    continue
                results.append(eval_hf_candidate(args, spec, windows, layout))
                payload["results"] = results
                (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                write_results_markdown(out_dir / "RESULTS.md", payload)

    if not args.skip_gguf:
        gguf_specs: list[ModelSpec] = []
        if args.gguf_f16_sanity and not args.skip_gguf_f16_sanity and Path(args.gguf_f16_sanity).exists():
            gguf_specs.append(ModelSpec("F16 GGUF sanity", args.gguf_f16_sanity))
        for raw in args.gguf_model or []:
            spec = parse_model_spec(raw)
            if Path(spec.path).exists():
                gguf_specs.append(spec)
            else:
                log(f"[warn] skipping missing GGUF candidate: {spec.path}")

        if args.gguf_reference_mode == "disk" and gguf_specs:
            kld_path, kld_meta = build_reference_kld_disk(args, windows, layout, out_dir)
            try:
                for spec in gguf_specs:
                    results.append(run_gguf_disk(spec=spec, args=args, windows=windows, layout=layout, text_path=text_path, out_dir=out_dir, kld_path=kld_path, kld_meta=kld_meta))
                    payload["results"] = results
                    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    write_results_markdown(out_dir / "RESULTS.md", payload)
            finally:
                if args.cleanup_reference_kld or (not args.keep_reference_kld and args.reference_kld is None):
                    try:
                        log(f"[gguf:disk] removing temporary KLD cache: {kld_path}")
                        kld_path.unlink()
                    except FileNotFoundError:
                        pass
        else:
            for spec in gguf_specs:
                results.append(run_gguf_with_fifo(spec=spec, args=args, windows=windows, layout=layout, text_path=text_path, out_dir=out_dir))
                payload["results"] = results
                (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                write_results_markdown(out_dir / "RESULTS.md", payload)

    payload["results"] = results
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_results_markdown(out_dir / "RESULTS.md", payload)
    log(f"[done] wrote {out_dir / 'results.json'}")
    log(f"[done] wrote {out_dir / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
