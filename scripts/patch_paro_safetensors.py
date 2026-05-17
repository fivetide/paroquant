#!/usr/bin/env python3
"""Patch ParoQuant exported safetensors to include standard fp16 fallback weights
and fix expert parameter naming.

Our export has two issues:
1. Missing fp16 fallback weights for skip_modules (shared_expert, mlp.gate, etc.)
2. Expert parameters use wrong names: model expects gate_up_proj/down_proj,
   but our optimizer uses gate_proj/down_proj

This script:
1. Extracts original fp16 weights from optimizer .pt files
2. Adds them as standard .weight / .bias entries
3. Fixes expert parameter names to match model expectations:
   - experts.{id}.qweight -> experts.{id}.gate_up_proj.qweight
   - experts.{id}.qzeros -> experts.{id}.gate_up_proj.qzeros
   - experts.{id}.scales -> experts.{id}.gate_up_proj.scales
   - experts.{id}.down_proj.qweight -> experts.{id}.down_proj.weight (fp16 fallback)
   - experts.{id}.down_proj.qzeros -> experts.{id}.down_proj.qzeros (keep)
   - experts.{id}.down_proj.scales -> experts.{id}.down_proj.scales (keep)
"""

import argparse
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    work_dir = Path(args.work_dir)
    safetensors_path = export_dir / "model.safetensors"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"Safetensors not found: {safetensors_path}")

    print(f"Loading existing safetensors...")
    existing_sd = load_file(str(safetensors_path))
    print(f"  Existing keys: {len(existing_sd)}")

    new_keys = {}  # key -> tensor

    # ── 1. Add original fp16 weights from optimizer .pt files ─────────────────
    for pt_file in sorted(work_dir.glob("*.pt")):
        stem = pt_file.stem  # e.g. "0.mlp.gate_proj" or "0.mlp.experts"
        parts = stem.split(".", 1)
        if len(parts) < 2:
            continue
        try:
            layer_idx = int(parts[0])
        except ValueError:
            continue
        module_path = parts[1]  # e.g. "mlp.gate_proj" or "mlp.experts"

        try:
            sd = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  Warning: {pt_file.name}: {e}")
            continue

        # Build standard safetensors key prefix
        key_prefix = f"model.language_model.layers.{layer_idx}.{module_path}"

        # For linear layers and shared_expert: add .weight and .bias
        if "weight" in sd:
            w = sd["weight"]
            if w.dtype in (torch.float32, torch.float16, torch.bfloat16):
                weight_key = key_prefix + ".weight"
                if weight_key not in existing_sd:
                    new_keys[weight_key] = w.to(torch.float16).cpu()

        if "bias" in sd and sd["bias"] is not None:
            bias_key = key_prefix + ".bias"
            if bias_key not in existing_sd:
                new_keys[bias_key] = sd["bias"].to(torch.float16).cpu()

        # For MoE experts .pt file: extract per-expert fp16 fallback weights
        # The .pt has gate_up_weight [256, M, K] and down_weight [256, K, N]
        if "gate_up_weight" in sd and "down_weight" in sd:
            gate_up = sd["gate_up_weight"]      # [256, M, K]
            down_w = sd["down_weight"]           # [256, K, N]

            num_experts = gate_up.shape[0]

            for expert_id in range(num_experts):
                base = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}"

                # gate_up: stored as qweight/qzeros/scales, add fp16 fallback
                gate_key = base + ".gate_up_proj.weight"
                if gate_key not in existing_sd:
                    new_keys[gate_key] = gate_up[expert_id].to(torch.float16).cpu()

                # down: stored as down_proj.qweight, add fp16 fallback
                down_key = base + ".down_proj.weight"
                if down_key not in existing_sd:
                    new_keys[down_key] = down_w[expert_id].to(torch.float16).cpu()

    print(f"\nNew keys to add: {len(new_keys)}")
    for k in sorted(new_keys.keys())[:8]:
        t = new_keys[k]
        print(f"  {k}: shape={tuple(t.shape)}, dtype={t.dtype}")
    if len(new_keys) > 8:
        print(f"  ... and {len(new_keys) - 8} more")

    # ── 2. Fix expert parameter naming in existing quantized keys ─────────────
    # Our optimizer uses: experts.{id}.qweight / experts.{id}.down_proj.qweight
    # Model expects:        experts.{id}.gate_up_proj.qweight / experts.{id}.down_proj.qweight
    renamed_keys = {}
    removed_keys = []

    for key in list(existing_sd.keys()):
        if ".experts." in key:
            parts = key.split(".")
            # Find the expert ID and parameter
            try:
                exp_idx = next(i for i, p in enumerate(parts) if p.isdigit() and i > 0 and parts[i-1] == "experts")
            except StopIteration:
                continue

            expert_id = parts[exp_idx]
            param = parts[-1]  # e.g. "qweight", "qzeros", "scales"
            suffix = ".".join(parts[exp_idx+1:])  # e.g. "qweight" or "down_proj.qweight"

            # Fix gate parameter naming: .qweight -> .gate_up_proj.qweight etc.
            if suffix == "qweight":
                new_key = ".".join(parts[:exp_idx+1]) + ".gate_up_proj.qweight"
                if new_key not in existing_sd and new_key not in new_keys:
                    renamed_keys[key] = new_key
                    removed_keys.append(key)
            elif suffix == "qzeros":
                new_key = ".".join(parts[:exp_idx+1]) + ".gate_up_proj.qzeros"
                if new_key not in existing_sd and new_key not in new_keys:
                    renamed_keys[key] = new_key
                    removed_keys.append(key)
            elif suffix == "scales":
                new_key = ".".join(parts[:exp_idx+1]) + ".gate_up_proj.scales"
                if new_key not in existing_sd and new_key not in new_keys:
                    renamed_keys[key] = new_key
                    removed_keys.append(key)
            # down_proj.* stays as-is (already correct naming)

    print(f"\nExpert keys to rename: {len(renamed_keys)}")
    for old, new in list(renamed_keys.items())[:3]:
        print(f"  {old}\n  -> {new}")

    if args.dry_run:
        print("\nDry run — not writing.")
        return

    # ── 3. Build merged state dict and save ───────────────────────────────────
    merged = dict(existing_sd)

    # Remove old (incorrectly named) expert keys
    for k in removed_keys:
        del merged[k]

    # Rename expert keys
    for old_key, new_key in renamed_keys.items():
        merged[new_key] = existing_sd[old_key]

    # Add new fp16 fallback keys
    for k, v in new_keys.items():
        if k not in merged:
            merged[k] = v

    print(f"\nFinal: {len(existing_sd)} -> {len(merged)} keys")
    print(f"  Removed: {len(removed_keys)}, Renamed: {len(renamed_keys)}, Added: {len(new_keys)}")

    tmp_path = safetensors_path.with_suffix(".patched.safetensors")
    save_file(merged, str(tmp_path), metadata={"format": "pt"})

    bak = safetensors_path.with_suffix(".orig.safetensors")
    shutil.copy(safetensors_path, bak)
    print(f"Backup: {bak}")
    shutil.move(tmp_path, safetensors_path)
    print(f"Replaced: {safetensors_path}")
    print("Done.")


if __name__ == "__main__":
    main()