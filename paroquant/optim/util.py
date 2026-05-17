from __future__ import annotations

import gc
import glob
import json
import logging
import math
import random
import warnings
from pathlib import Path
from typing import Any, Iterable, TypeVar

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from contextlib import suppress


def get_blocks(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model = model.model.language_model
    elif hasattr(model, "model"):
        model = model.model

    if hasattr(model, "layers"):
        return model.layers

    raise NotImplementedError(type(model))


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(module: nn.Module, subclass: type[_Linear_T] = nn.Linear) -> dict[str, _Linear_T]:
    return {name: m for name, m in module.named_modules() if isinstance(m, subclass)}


def get_module_by_name(module, module_name):
    for name, m in module.named_modules():
        if name == module_name:
            return m
    return None


def set_module_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def load_model(model_path: str, device_map: str | None = None, dtype=torch.float32, **kwargs) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, dtype=dtype, **kwargs)
    return model


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def move_embed(model, device):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        # AutoModelForCausalLM returns Gemma4ForConditionalGeneration for Gemma 4. Need to unwrap.
        model = model.model.language_model
    elif hasattr(model, "model"):
        model = model.model
    else:
        raise NotImplementedError(type(model))

    if hasattr(model, "embed_tokens"):
        model.embed_tokens = model.embed_tokens.to(device)

    if hasattr(model, "rotary_emb"):
        model.rotary_emb = model.rotary_emb.to(device)


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _normalize_chat_role(role: Any) -> str:
    role_str = str(role or "user").strip().lower()
    if role_str in {"human", "user", "instruction", "input"}:
        return "user"
    if role_str in {"gpt", "assistant", "model", "bot", "teacher"}:
        return "assistant"
    if role_str == "system":
        return "system"
    return "user"


def _messages_to_text(messages: Any, tokenizer) -> str:
    if isinstance(messages, str):
        with suppress(json.JSONDecodeError):
            messages = json.loads(messages)
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return ""

    normalized = []
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
        normalized.append({"role": _normalize_chat_role(role), "content": str(content)})

    if not normalized:
        return ""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is not None:
        try:
            return apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass

    eos = getattr(tokenizer, "eos_token", "") or ""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in normalized).strip() + eos


def _jsonl_row_to_text(row: dict[str, Any], tokenizer) -> str:
    for key in ("text", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value

    for key in ("conversations", "messages", "conversation", "chat", "dialogue"):
        if key in row:
            text = _messages_to_text(row[key], tokenizer)
            if text:
                return text

    # Preference data: calibrate on the preferred trajectory only.
    if "chosen" in row:
        text = _messages_to_text(row["chosen"], tokenizer)
        if text:
            return text

    # GAD-style rows: prompt is a chat/message list and teacher is the answer.
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
        text = _messages_to_text(messages, tokenizer)
        if text:
            return text

    # Generic prompt/completion fallback.
    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return f"{row['prompt']}\n{row['completion']}"

    return ""


def _resolve_jsonl_paths(data: str) -> list[Path]:
    source = data.removeprefix("jsonl:")
    source = str(Path(source).expanduser())
    if any(ch in source for ch in "*?[]"):
        paths = [Path(p) for p in glob.glob(source)]
    else:
        path = Path(source)
        if path.is_dir():
            paths = list(path.glob("*.jsonl"))
        elif path.is_file() and path.suffix == ".jsonl":
            paths = [path]
        else:
            paths = []
    paths = sorted(p for p in paths if p.is_file() and p.suffix == ".jsonl")
    if not paths:
        raise FileNotFoundError(f"No JSONL calibration files found for {data!r}")
    return paths


def _is_jsonl_calib_source(data: str) -> bool:
    if data.startswith("jsonl:"):
        return True
    try:
        _resolve_jsonl_paths(data)
        return True
    except FileNotFoundError:
        return False


def _iter_jsonl_texts(data: str, tokenizer, seed: int) -> Iterable[str]:
    paths = _resolve_jsonl_paths(data)
    rand = random.Random(seed)
    rand.shuffle(paths)
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
                if isinstance(row, str):
                    text = row
                elif isinstance(row, dict):
                    text = _jsonl_row_to_text(row, tokenizer)
                else:
                    text = ""
                text = text.strip()
                if text:
                    yield text


def get_mixed_calib_dataset(
    datasets: list[str],
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    per_dataset_len = n_samples // len(datasets)
    results = []
    for i, dataset in enumerate(datasets):
        dataset_samples = per_dataset_len if i < len(datasets) - 1 else n_samples - len(results)
        results.extend(
            get_calib_dataset(
                data=dataset,
                tokenizer=tokenizer,
                n_samples=dataset_samples,
                block_size=block_size,
                seed=seed,
                split=split,
            )
        )
    assert len(results) == n_samples, f"Expected {n_samples} samples, got {len(results)}"

    rand = random.Random(seed)
    rand.shuffle(results)

    return results


# Adapted from awq-llm
def get_calib_dataset(
    data="pileval",
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    allow_long_rows = False
    if _is_jsonl_calib_source(data):
        line_iter = _iter_jsonl_texts(data, tokenizer, seed)
        allow_long_rows = True
    else:
        if data == "pileval":
            if split != "validation":
                warnings.warn("The split argument is ignored when data is 'pileval'.")
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            dataset = dataset.shuffle(seed=seed)
        elif data == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            dataset = dataset.shuffle(seed=seed)
        elif data == "c4":
            if split == "train":
                dataset = load_dataset(
                    "allenai/c4",
                    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                    split=split,
                )
            elif split == "validation":
                dataset = load_dataset(
                    "allenai/c4",
                    data_files={"validation": "en/c4-validation.00001-of-00008.json.gz"},
                    split=split,
                )
            dataset = dataset.shuffle(seed=seed)
        elif data == "redpajama":
            test_split, val_split = 0.2, 0.1
            dataset = load_dataset(
                "liang2kl/RedPajama-Data-1T-Sample-Backup",
                split="train",
                trust_remote_code=True,
            )
            dataset = dataset.shuffle(seed=seed)
            test_size = int(len(dataset) * test_split)
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - test_size - val_size
            if split == "test":
                dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
            elif split == "validation":
                dataset = dataset.select(range(len(dataset) - test_size - val_size, len(dataset) - test_size))
            elif split == "train":
                dataset = dataset.select(range(0, train_size))
            else:
                raise ValueError(f"Invalid split: {split}")
        else:
            raise NotImplementedError
        line_iter = (str(row["text"]).strip() for row in dataset if row.get("text") is not None)

    token_ids: list[int] = []
    target_len = n_samples * block_size
    for line in line_iter:
        if not line:
            continue
        line_encoded = tokenizer.encode(line)
        if not line_encoded:
            continue
        if len(line_encoded) > block_size and not allow_long_rows:
            continue
        token_ids.extend(line_encoded)
        if len(token_ids) >= target_len:
            break

    if not token_ids:
        raise ValueError(f"Calibration source {data!r} produced no usable text")
    samples = torch.tensor(token_ids[:target_len])
    n_split = min(samples.shape[0] // block_size, n_samples)

    return [samples[i * block_size : (i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def catch_first_layer_input_and_all_layer_kwargs(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: int | None,
) -> tuple[torch.Tensor, list[dict]]:
    kwargs_list: list[dict] = [{} for _ in range(len(layers))]
    batched = batch_size is not None
    inps: list[torch.Tensor] = []

    class Catcher(nn.Module):

        def __init__(self, module, layer_idx):
            super().__init__()
            # Bypass __setattr__ of nn.Module
            object.__setattr__(self, "module", module)
            object.__setattr__(self, "layer_idx", layer_idx)

        def forward(self, inp, *args, **kwargs):
            # We only capture the first input.
            if self.layer_idx == 0:
                inps.append(inp)

            # Capture kwargs for all layers.
            layer_kwargs = kwargs_list[self.layer_idx]
            if len(layer_kwargs) == 0:
                layer_kwargs.update(kwargs)
                layer_kwargs.pop("use_cache", None)
                layer_kwargs.pop("past_key_value", None)
                layer_kwargs.pop("past_key_values", None)

            # Gemma 4 has an extra `per_layer_input` arg. Ignore it since it's only for multimodal tokens.
            if len(args) > 0:
                warnings.warn(f"Silently ignoring additional positional arguments in layer forward: {args}.")

            if self.layer_idx == len(layers) - 1:
                raise ValueError

            # Return a dummy output
            return torch.empty_like(inp)

        def __getattr__(self, name):
            return getattr(self.module, name)

    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = Catcher(layer, layer_idx)

    batch_size = samples.shape[0] if not batched or batch_size <= 0 else batch_size
    num_batches = samples.shape[0] // batch_size
    samples_batch = samples.chunk(num_batches)
    for samples in samples_batch:
        with suppress(ValueError):
            model(samples.to(next(model.parameters()).device))

    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = layer.module

    if not batched:
        inps = inps[0]

    return inps, kwargs_list


class CachedTensorShards:
    def __init__(
        self,
        batches: list[torch.Tensor],
        num_shards: int,
        *,
        target_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        assert len(batches) % num_shards == 0
        if batches[0].device != offload_device:
            self.batches = [b.to(offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        if self.current_shard == shard_index:
            return
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        self.cached_shard = [b.to(self.target_device) for b in self.cached_shard]

    def _get_shard_range(self, index: int) -> tuple[int, int]:
        if self.num_shards == 1:
            return 0, len(self.batches)
        shard_size = len(self.batches) // self.num_shards
        start = shard_size * index
        if index == self.num_shards - 1:
            end = len(self.batches)
        else:
            end = shard_size * (index + 1)
        return start, end

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_len = len(self.batches) // self.num_shards
        shard_index = index // shard_len
        if self.current_shard != shard_index:
            self._switch_shard(shard_index)
        offset = index % shard_len
        return self.cached_shard[offset]

    def __iter__(self) -> "Iterator":
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.batches)

    class Iterator:
        def __init__(self, batches: "CachedTensorShards"):
            self.batches = batches
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self) -> torch.Tensor:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


class CosineAnnealingParam:
    def __init__(self, start_value: float, end_value: float, T_max: int):
        """
        Args:
            start_value (float): The initial value (equivalent to eta_max).
            end_value (float): The final value (equivalent to eta_min).
            T_max (int): Maximum number of steps.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        self._step += 1

        if self._step >= self.T_max:
            return self.end_value

        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = get_logger("ParoQuant")
