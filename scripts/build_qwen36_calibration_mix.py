#!/usr/bin/env python3
"""Build a weighted local JSONL calibration mix for Qwen3.6 ParoQuant.

The output is a plain JSONL file with a `text` field plus source metadata. It is
intended to be passed to the local ParoQuant fork via `--datasets` and
`--val-dataset`.

Default mix, with --chat-pct 0.08:
  - 30% English/general/reference TX4 cache docs
  - 20% Japanese TX4 cache docs
  -  8% Chinese TX4 cache docs
  -  8% other multilingual TX4 cache docs
  - 10% math/STEM TX4 cache docs
  - 16% broad Stack-Edu code cache docs
  -  8% Chotto SFT chat/translation rows rendered through Qwen's chat template

The non-chat proportions mirror the broad TX4 quality mix, but code uses the
TX4 code-breadth source set instead of only Python/JavaScript/TypeScript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = (
    "/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DEFAULT_OUTPUT_DIR = "/models/qwen36-calibration"
DEFAULT_TX4_CACHE_ROOTS = [
    Path("/data/truenas/sansho/outputs/data/cache/tx4-quality3-a3-12b"),
    Path("/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b"),
]
DEFAULT_CHOTTO_SFT = Path("/home/lhl/chotto-train/202601-chotto/chotto-20260107.sft.jsonl")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    group: str
    rel_weight: float
    paths: tuple[Path, ...]
    kind: str = "auto"


# Non-chat group proportions. With the default --chat-pct 0.08, these become:
# EN 30%, JA 20%, ZH 8%, other multilingual 8%, math 10%, code breadth 16%.
NON_CHAT_GROUP_WEIGHTS = {
    "english_general": 30 / 92,
    "japanese": 20 / 92,
    "chinese": 8 / 92,
    "other_multilingual": 8 / 92,
    "math_stem": 10 / 92,
    "code_breadth": 16 / 92,
}

# Source-relative weights within each group. Most are adapted from the TX4
# quality3 combo and code-breadth mixture specs, lightly compressed for a small
# calibration corpus.
GROUP_SOURCE_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "english_general": [
        ("fineweb-edu-sample", 7),
        ("fineweb-sample", 5),
        ("dclm-baseline", 3),
        ("pg19", 2),
        ("pes2o-filtered", 2),
        ("finewiki-en", 3),
        ("wikipedia-en", 3),
        ("common-pile-arxiv-filtered", 4),
        ("common-pile-pressbooks-filtered", 3),
        ("common-pile-doab-filtered", 3),
        ("common-pile-pubmed-filtered", 3),
    ],
    "japanese": [
        ("abeja-cc-ja", 15),
        ("fineweb2-ja", 6),
        ("fineweb2-ja-edu", 6),
        ("aozora-clean", 4),
        ("finewiki-ja", 3),
        ("hpprc-jawiki", 3),
        ("j-research-corpus", 2),
        ("japan-law-egov", 2),
        ("kaken-trans-ja-en", 2),
    ],
    "chinese": [
        ("fineweb2-zh", 3),
        ("finewiki-zh", 1),
    ],
    "other_multilingual": [
        ("fineweb2-ko", 3),
        ("fineweb2-mn-cyrl", 2),
        ("fineweb2-ar", 1),
        ("fineweb2-ru", 1),
        ("fineweb2-es", 1),
        ("fineweb2-fr", 1),
        ("fineweb2-de", 1),
    ],
    "math_stem": [
        ("finemath-4plus", 6),
        ("open-web-math", 2),
        ("megamath", 1),
        ("swallow-math-v2-textbook", 1),
        ("swallow-math-v2-qa", 1),
    ],
    "code_breadth": [
        ("stack-edu-python-materialized", 140),
        ("stack-edu-javascript-materialized", 100),
        ("stack-edu-typescript-materialized", 60),
        ("stack-edu-java-materialized", 140),
        ("stack-edu-cpp-materialized", 100),
        ("stack-edu-go-materialized", 90),
        ("stack-edu-markdown-materialized", 80),
        ("stack-edu-rust-materialized", 70),
        ("stack-edu-c-materialized", 50),
        ("stack-edu-csharp-materialized", 40),
        ("stack-edu-shell-materialized", 40),
        ("stack-edu-sql-materialized", 30),
        ("stack-edu-php-materialized", 30),
        ("stack-edu-ruby-materialized", 20),
        ("stack-edu-swift-materialized", 10),
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Qwen tokenizer/model path")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--name", default="qwen36-paro-tx4-codebreadth-chotto")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--validation-samples", type=int, default=64)
    p.add_argument("--chat-pct", type=float, default=0.08, help="Fraction of tokens from Chotto SFT chat data")
    p.add_argument("--seed", type=int, default=20260510)
    p.add_argument("--chotto-sft", default=str(DEFAULT_CHOTTO_SFT))
    p.add_argument(
        "--exclude-jsonl",
        action="append",
        default=None,
        help="Existing JSONL file(s) whose rendered text hashes should be excluded from the generated split",
    )
    p.add_argument(
        "--cache-root",
        action="append",
        default=None,
        help="TX4 cache root containing source-name/docs-*.jsonl dirs; may be passed multiple times",
    )
    p.add_argument("--dry-run", action="store_true", help="Print resolved weights and exit")
    return p.parse_args()


def resolve_source_paths(source_name: str, cache_roots: list[Path]) -> tuple[Path, ...]:
    for root in cache_roots:
        source_dir = root / source_name
        if not source_dir.is_dir():
            continue
        paths = tuple(sorted(source_dir.glob("*.jsonl")))
        if paths:
            return paths
    return ()


def build_sources(*, cache_roots: list[Path], chotto_sft: Path, chat_pct: float) -> list[SourceSpec]:
    if not (0 <= chat_pct < 1):
        raise ValueError("--chat-pct must be in [0, 1)")

    specs: list[SourceSpec] = []
    for group, group_weight in NON_CHAT_GROUP_WEIGHTS.items():
        source_weights = GROUP_SOURCE_WEIGHTS[group]
        denom = sum(weight for _, weight in source_weights)
        for source_name, rel in source_weights:
            paths = resolve_source_paths(source_name, cache_roots)
            if not paths:
                print(f"warning: missing source {source_name!r}; skipping")
                continue
            specs.append(
                SourceSpec(
                    name=source_name,
                    group=group,
                    rel_weight=(1 - chat_pct) * group_weight * rel / denom,
                    paths=paths,
                    kind="text",
                )
            )

    if chat_pct > 0:
        if not chotto_sft.is_file():
            raise FileNotFoundError(f"Chotto SFT JSONL not found: {chotto_sft}")
        specs.append(
            SourceSpec(
                name="chotto-20260107-sft",
                group="chat_translation",
                rel_weight=chat_pct,
                paths=(chotto_sft,),
                kind="chat",
            )
        )

    total = sum(spec.rel_weight for spec in specs)
    if total <= 0:
        raise RuntimeError("No usable sources resolved")
    return [SourceSpec(s.name, s.group, s.rel_weight / total, s.paths, s.kind) for s in specs]


def normalize_role(role: Any) -> str:
    role_s = str(role or "user").strip().lower()
    if role_s in {"human", "user", "instruction", "input"}:
        return "user"
    if role_s in {"gpt", "assistant", "model", "bot", "teacher"}:
        return "assistant"
    if role_s == "system":
        return "system"
    return "user"


def messages_to_text(messages: Any, tokenizer: Any) -> str:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return ""

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("from", msg.get("speaker", "user")))
            content = msg.get("content", msg.get("value", msg.get("text", "")))
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            role, content = msg
        else:
            continue
        if content is None:
            continue
        normalized.append({"role": normalize_role(role), "content": str(content)})

    if not normalized:
        return ""

    if getattr(tokenizer, "apply_chat_template", None):
        try:
            return tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass

    eos = getattr(tokenizer, "eos_token", "") or ""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in normalized).strip() + eos


def row_to_text(row: Any, tokenizer: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""

    for key in ("text", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value

    for key in ("conversations", "messages", "conversation", "chat", "dialogue"):
        if key in row:
            return messages_to_text(row[key], tokenizer)

    if "chosen" in row:
        return messages_to_text(row["chosen"], tokenizer)

    if "prompt" in row and "teacher" in row:
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            messages = []
        teacher = row["teacher"]
        if isinstance(teacher, dict):
            messages = [*messages, teacher]
        elif isinstance(teacher, str):
            messages = [*messages, {"role": "assistant", "content": teacher}]
        return messages_to_text(messages, tokenizer)

    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return f"{row['prompt']}\n{row['completion']}"

    return ""


def iter_rows(spec: SourceSpec, *, tokenizer: Any, seed: int) -> Iterable[tuple[str, str]]:
    rng = random.Random(f"{seed}:{spec.name}")
    paths = list(spec.paths)
    rng.shuffle(paths)

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row_to_text(row, tokenizer).strip()
                if text:
                    yield text, path.name


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def allocate_source_targets(specs: list[SourceSpec], target_tokens: int) -> dict[str, int]:
    raw = [(spec.name, target_tokens * spec.rel_weight) for spec in specs]
    targets = {name: int(value) for name, value in raw}
    remainder = target_tokens - sum(targets.values())
    if remainder > 0:
        by_fraction = sorted(raw, key=lambda x: x[1] - int(x[1]), reverse=True)
        for name, _ in by_fraction[:remainder]:
            targets[name] += 1
    return targets


def collect_split(
    *,
    split_name: str,
    specs: list[SourceSpec],
    tokenizer: Any,
    target_tokens: int,
    seed: int,
    used_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(f"{seed}:{split_name}:global")
    out_rows: list[dict[str, Any]] = []
    source_targets = allocate_source_targets(specs, target_tokens)
    summary: dict[str, Any] = {
        "target_tokens": target_tokens,
        "actual_tokens": 0,
        "by_source": defaultdict(lambda: {"rows": 0, "tokens": 0, "target_tokens": 0}),
        "by_group": defaultdict(lambda: {"rows": 0, "tokens": 0, "target_tokens": 0}),
    }

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"collect {split_name}")
    for spec in specs:
        source_target = source_targets[spec.name]
        summary["by_source"][spec.name]["target_tokens"] = source_target
        summary["by_group"][spec.group]["target_tokens"] += source_target
        source_tokens = 0
        for text, file_name in iter_rows(spec, tokenizer=tokenizer, seed=seed + (17 if split_name == "validation" else 0)):
            h = stable_hash(text)
            if h in used_hashes:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            remaining = source_target - source_tokens
            if remaining <= 0:
                break
            if len(token_ids) > remaining:
                token_ids = token_ids[:remaining]
                text = tokenizer.decode(token_ids, skip_special_tokens=False)
            token_count = len(token_ids)
            if token_count == 0:
                continue
            used_hashes.add(h)
            row = {
                "text": text,
                "source": spec.name,
                "group": spec.group,
                "source_file": file_name,
                "qwen_tokens": token_count,
            }
            out_rows.append(row)
            source_tokens += token_count
            summary["actual_tokens"] += token_count
            summary["by_source"][spec.name]["rows"] += 1
            summary["by_source"][spec.name]["tokens"] += token_count
            summary["by_group"][spec.group]["rows"] += 1
            summary["by_group"][spec.group]["tokens"] += token_count
            pbar.update(token_count)
            if source_tokens >= source_target:
                break
        if source_tokens < source_target:
            print(f"warning: source {spec.name} under target: {source_tokens}/{source_target} tokens")
    pbar.close()

    rng.shuffle(out_rows)
    summary["by_source"] = dict(sorted(summary["by_source"].items()))
    summary["by_group"] = dict(sorted(summary["by_group"].items()))
    return out_rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_excluded_hashes(paths: list[str] | None, tokenizer: Any) -> set[str]:
    hashes: set[str] = set()
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"--exclude-jsonl not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row_to_text(row, tokenizer).strip()
                if text:
                    hashes.add(stable_hash(text))
    return hashes


def main() -> int:
    args = parse_args()
    cache_roots = [Path(p) for p in args.cache_root] if args.cache_root else DEFAULT_TX4_CACHE_ROOTS
    output_dir = Path(args.output_dir)
    chotto_sft = Path(args.chotto_sft)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    specs = build_sources(cache_roots=cache_roots, chotto_sft=chotto_sft, chat_pct=args.chat_pct)

    print("Resolved calibration sources:")
    by_group = defaultdict(float)
    for spec in specs:
        by_group[spec.group] += spec.rel_weight
    for group, weight in sorted(by_group.items()):
        print(f"  {group:18s} {weight * 100:6.2f}%")
    print(f"  {'TOTAL':18s} {sum(by_group.values()) * 100:6.2f}%")

    if args.dry_run:
        print("\nSources:")
        for spec in specs:
            print(f"  {spec.rel_weight * 100:6.3f}% {spec.group:18s} {spec.name:36s} ({len(spec.paths)} files)")
        return 0

    train_tokens = args.train_samples * args.seq_len
    val_tokens = args.validation_samples * args.seq_len
    used_hashes = load_excluded_hashes(args.exclude_jsonl, tokenizer)
    if used_hashes:
        print(f"Loaded {len(used_hashes)} excluded text hashes")

    train_rows, train_summary = collect_split(
        split_name="train",
        specs=specs,
        tokenizer=tokenizer,
        target_tokens=train_tokens,
        seed=args.seed,
        used_hashes=used_hashes,
    )
    val_rows, val_summary = collect_split(
        split_name="validation",
        specs=specs,
        tokenizer=tokenizer,
        target_tokens=val_tokens,
        seed=args.seed,
        used_hashes=used_hashes,
    )

    train_path = output_dir / f"{args.name}-train-{args.train_samples}x{args.seq_len}.jsonl"
    val_path = output_dir / f"{args.name}-val-{args.validation_samples}x{args.seq_len}.jsonl"
    manifest_path = output_dir / f"{args.name}-manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "name": args.name,
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "chat_pct": args.chat_pct,
        "seed": args.seed,
        "cache_roots": [str(p) for p in cache_roots],
        "chotto_sft": str(chotto_sft),
        "outputs": {"train": str(train_path), "validation": str(val_path)},
        "resolved_source_weights": [
            {
                "name": spec.name,
                "group": spec.group,
                "weight": spec.rel_weight,
                "files": len(spec.paths),
                "example_path": str(spec.paths[0]) if spec.paths else None,
            }
            for spec in specs
        ],
        "train_summary": train_summary,
        "validation_summary": val_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nWrote:")
    print(f"  train:      {train_path} ({len(train_rows)} rows, {train_summary['actual_tokens']} Qwen tokens)")
    print(f"  validation: {val_path} ({len(val_rows)} rows, {val_summary['actual_tokens']} Qwen tokens)")
    print(f"  manifest:   {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
