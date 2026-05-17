from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

# AutoAWQ's GEMM kernel imports PytorchGELUTanh from transformers.activations,
# but it was removed in transformers >=4.55. Provide a stub until AutoAWQ is updated.
import transformers.activations as _act

if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUActivation

try:
    if torch.version.hip is not None:
        raise ImportError("prefer ROCm AWQ fallback on HIP builds")
    from awq.modules.linear.gemm import WQLinearMMFunction
except ImportError:
    WQLinearMMFunction = None

if torch.version.hip is not None:
    from paroquant.kernels.rocm.awq import _load_awq_extension as rocm_load_awq_extension
    from paroquant.kernels.rocm.awq import awq_gemv as rocm_awq_gemv
    from paroquant.kernels.rocm.awq import awq_linear as rocm_awq_linear
    from paroquant.kernels.rocm.awq import dequantize_awq as rocm_dequantize_awq
else:
    rocm_load_awq_extension = None
    rocm_awq_gemv = None
    rocm_awq_linear = None
    rocm_dequantize_awq = None


def _rocm_direct_gemv_enabled(x: torch.Tensor, out_features: int, bits: int) -> bool:
    rows = x.numel() // x.shape[-1]
    return (
        0 < rows <= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_ROWS", "8"))
        and x.shape[-1] >= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MIN_IN_FEATURES", "1024"))
        and out_features <= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_OUT_FEATURES", "1024"))
    )


@torch.no_grad()
def _torch_dequantize_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bits != 4:
        raise RuntimeError(f"Only 4-bit AWQ fallback is supported, got bits={bits}")
    pack = 32 // bits
    in_features, out_packed = qweight.shape
    out_features = out_packed * pack
    if qzeros.shape != (in_features // group_size, out_packed):
        raise RuntimeError(f"Unexpected qzeros shape: {tuple(qzeros.shape)}")
    if scales.shape != (in_features // group_size, out_features):
        raise RuntimeError(f"Unexpected scales shape: {tuple(scales.shape)}")

    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=qweight.device, dtype=torch.int32)
    q = ((qweight.contiguous().unsqueeze(-1) >> shifts) & 0xF).reshape(in_features, out_features)
    z = ((qzeros.contiguous().unsqueeze(-1) >> shifts) & 0xF).reshape(in_features // group_size, out_features)
    z = z.repeat_interleave(group_size, dim=0)
    scale = scales.contiguous().repeat_interleave(group_size, dim=0).to(dtype=dtype)
    return (q.to(dtype=dtype) - z.to(dtype=dtype)) * scale


class AWQLinearNoRotate(nn.Module):
    """AWQ uint4 linear without a local rotation op.

    Qwen3.5 PARO fused-MoE checkpoints store expert projections as individual
    ``experts.<id>.<proj>.qweight/qzeros/scales`` tensors, while the rotation is
    shared across all experts as ``experts.gate_up_weight_*`` and
    ``experts.down_weight_*``. This helper lets the fused expert runtime apply
    the shared rotation once and then call the same ROCm AWQ bridge/direct-GEMV
    path used by RotateQuantizedLinear.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bits: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = bits
        self.group_size = group_size
        pack = 32 // bits
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.zeros(in_features, out_features // pack, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(n_groups, out_features // pack, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(n_groups, out_features, dtype=torch.float16))
        self._gemv_max_rows = int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_ROWS", "8"))
        self._direct_gemv_candidate = (
            in_features >= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MIN_IN_FEATURES", "1024"))
            and out_features <= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_OUT_FEATURES", "1024"))
        )
        self._cache_dequant_weight = (
            os.environ.get("PAROQUANT_ROCM_CACHE_EXPERT_DEQUANT", "1") != "0" and not self._direct_gemv_candidate
        )
        self._cached_dequant_weight: torch.Tensor | None = None
        self._rocm_gemv_fn = None
        self._rocm_dequant_fn = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if rocm_load_awq_extension is not None:
            if self._direct_gemv_candidate:
                rows = x.shape[0]
                if 0 < rows <= self._gemv_max_rows:
                    gemv_fn = self._rocm_gemv_fn
                    if gemv_fn is None:
                        ext = rocm_load_awq_extension()
                        gemv_fn = ext.gemv_awq
                        object.__setattr__(self, "_rocm_gemv_fn", gemv_fn)
                    return gemv_fn(x, self.qweight, self.qzeros, self.scales, None, self.w_bit, self.group_size)
            weight = self._cached_dequant_weight if self._cache_dequant_weight else None
            if weight is None or weight.device != x.device or weight.dtype != x.dtype:
                dequant_fn = self._rocm_dequant_fn
                if dequant_fn is None:
                    ext = rocm_load_awq_extension()
                    dequant_fn = ext.dequant_awq
                    object.__setattr__(self, "_rocm_dequant_fn", dequant_fn)
                weight = dequant_fn(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size)
                if self._cache_dequant_weight:
                    self._cached_dequant_weight = weight
        else:
            cache = os.environ.get("PAROQUANT_CUDA_CACHE_EXPERT_DEQUANT", "0") != "0"
            weight = self._cached_dequant_weight if cache else None
            if weight is None or weight.device != x.device or weight.dtype != x.dtype:
                weight = _torch_dequantize_awq(
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.w_bit,
                    self.group_size,
                    dtype=x.dtype,
                )
                if cache:
                    self._cached_dequant_weight = weight
        x_flat = x.reshape(-1, x.shape[-1])
        y = torch.mm(x_flat, weight)
        return y.reshape(*x.shape[:-1], self.out_features)


class _Qwen35ParoExpert(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, group_size: int, bits: int) -> None:
        super().__init__()
        self.gate_proj = AWQLinearNoRotate(hidden_dim, intermediate_dim, group_size=group_size, bits=bits)
        self.up_proj = AWQLinearNoRotate(hidden_dim, intermediate_dim, group_size=group_size, bits=bits)
        self.down_proj = AWQLinearNoRotate(intermediate_dim, hidden_dim, group_size=group_size, bits=bits)


class RotateQuantizedQwen35MoeExperts(nn.Module):
    """Correctness-first real-quantized Qwen3.5 MoE expert runtime.

    This mirrors ``Qwen3_5MoeExperts.forward`` but reads PARO's real-quantized
    per-expert AWQ tensors and applies the shared ParoQuant rotations expected by
    the checkpoint. It is intentionally simple and expert-loop based; later loop
    iterations should profile it and replace the hot pieces with fused kernels.
    """

    def __init__(self, experts, group_size: int = 128, bits: int = 4, krot: int = 8) -> None:
        super().__init__()
        self.config = experts.config
        self.num_experts = int(experts.num_experts)
        self.hidden_dim = int(experts.hidden_dim)
        self.intermediate_dim = int(experts.intermediate_dim)
        self.act_fn = ACT2FN[self.config.hidden_act]
        self.w_bit = bits
        self.group_size = group_size
        self.has_gate = getattr(experts, "has_gate", False)
        self.has_bias = False
        self.is_transposed = False

        # Register numeric child modules directly so checkpoint keys match
        # ``...mlp.experts.0.gate_proj.qweight`` instead of adding an extra
        # ``.experts`` component.
        for expert_id in range(self.num_experts):
            self.add_module(str(expert_id), _Qwen35ParoExpert(self.hidden_dim, self.intermediate_dim, group_size, bits))
        # The modules are already registered under numeric names for checkpoint
        # compatibility; keep a plain tuple for faster integer lookup in the hot
        # expert loop without adding state-dict keys.
        object.__setattr__(self, "_expert_modules", tuple(self._modules[str(i)] for i in range(self.num_experts)))

        self.register_buffer("gate_up_weight_theta", torch.zeros(krot, self.hidden_dim // 2, dtype=torch.float16))
        self.register_buffer("gate_up_weight_pairs", torch.zeros(krot, self.hidden_dim, dtype=torch.int16))
        self.register_buffer("gate_up_weight_channel_scales", torch.ones(1, self.hidden_dim, dtype=torch.float16))
        self.register_buffer("down_weight_theta", torch.zeros(krot, self.intermediate_dim // 2, dtype=torch.float16))
        self.register_buffer("down_weight_pairs", torch.zeros(krot, self.intermediate_dim, dtype=torch.int16))
        self.register_buffer("down_weight_channel_scales", torch.ones(1, self.intermediate_dim, dtype=torch.float16))

    @torch.no_grad()
    def _rotate_gate_up(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.rotation.rotate(
            x,
            self.gate_up_weight_pairs,
            self.gate_up_weight_theta,
            self.gate_up_weight_channel_scales,
            self.group_size,
        )

    @torch.no_grad()
    def _rotate_down(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.rotation.rotate(
            x,
            self.down_weight_pairs,
            self.down_weight_theta,
            self.down_weight_channel_scales,
            self.group_size,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 1:
            gate_up_hidden_states = self._rotate_gate_up(hidden_states)
            expert_modules = self._expert_modules
            num_experts = self.num_experts
            act_fn = self.act_fn
            top_k_weight_row = top_k_weights[0]
            down_work_single: list[tuple[nn.Module, int]] = []
            down_inputs_single: list[torch.Tensor] = []
            expert_ids = top_k_index[0].detach().cpu().tolist()
            for top_k_pos, expert_id in enumerate(expert_ids):
                expert_id = int(expert_id)
                if expert_id == num_experts:
                    continue
                expert = expert_modules[expert_id]
                gate = expert.gate_proj(gate_up_hidden_states)
                up = expert.up_proj(gate_up_hidden_states)
                current_hidden_states = act_fn(gate) * up
                down_work_single.append((expert, top_k_pos))
                down_inputs_single.append(current_hidden_states)

            if down_inputs_single:
                rotated_down_inputs = self._rotate_down(torch.cat(down_inputs_single, dim=0))
                down_outputs_single = []
                weight_positions = []
                for offset, (expert, top_k_pos) in enumerate(down_work_single):
                    down_in = rotated_down_inputs[offset : offset + 1]
                    down_outputs_single.append(expert.down_proj(down_in))
                    weight_positions.append(top_k_pos)
                down_outputs = torch.cat(down_outputs_single, dim=0)
                if len(weight_positions) == len(expert_ids):
                    weights = top_k_weight_row[: len(weight_positions)].view(-1, 1)
                else:
                    weights = top_k_weight_row[weight_positions].view(-1, 1)
                return (down_outputs * weights).sum(dim=0, keepdim=True).to(hidden_states.dtype)
            return torch.zeros_like(hidden_states)

        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        gate_up_hidden_states = self._rotate_gate_up(hidden_states) if expert_hit.numel() else hidden_states

        down_work: list[tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        down_inputs: list[torch.Tensor] = []
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            expert_id = int(expert_idx.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = gate_up_hidden_states[token_idx]
            expert = self._expert_modules[expert_id]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            down_work.append((expert, token_idx, top_k_pos, current_hidden_states))
            down_inputs.append(current_hidden_states)

        if down_inputs:
            rotated_down_inputs = self._rotate_down(torch.cat(down_inputs, dim=0))
            offset = 0
            for expert, token_idx, top_k_pos, pre_rot_hidden in down_work:
                rows = pre_rot_hidden.shape[0]
                down_in = rotated_down_inputs[offset : offset + rows]
                offset += rows
                current_hidden_states = expert.down_proj(down_in)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class RotateQuantizedLinear(nn.Module):
    """Pairwise Givens rotation + INT4 quantized matmul.

    On NVIDIA/CUDA this uses AutoAWQ's GEMM kernel when available. On ROCm/HIP it
    uses a local AWQ uint4 path: direct packed GEMV for selected c=1 shapes, and
    dequantization followed by torch.matmul otherwise. The ROCm path is still a
    correctness/runtime bridge for PARO checkpoints, not the final tuned W4
    kernel family.

    All parameters are stored flat (no submodules), so state dict keys like
    ``gate_proj.theta`` and ``gate_proj.qweight`` match checkpoint naming directly.
    This enables native HuggingFace weight loading via ``from_pretrained``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = bits
        self.group_size = group_size

        pack = 32 // bits
        n_groups = in_features // group_size

        # Rotation buffers
        self.register_buffer("theta", torch.zeros(krot, in_features // 2, dtype=torch.float16))
        self.register_buffer("pairs", torch.zeros(krot, in_features, dtype=torch.int16))
        self.register_buffer("channel_scales", torch.ones(1, in_features, dtype=torch.float16))

        # AWQ quantized weight buffers
        self.register_buffer("qweight", torch.zeros(in_features, out_features // pack, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(n_groups, out_features // pack, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(n_groups, out_features, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        self._cached_dequant_weight: torch.Tensor | None = None
        self._gemv_max_rows = int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_ROWS", "8"))
        self._direct_gemv_candidate = (
            in_features >= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MIN_IN_FEATURES", "1024"))
            and out_features <= int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_OUT_FEATURES", "1024"))
        )
        self._cache_dense_dequant = os.environ.get("PAROQUANT_ROCM_CACHE_DENSE_DEQUANT", "1") != "0"
        self._rocm_gemv_fn = None
        self._rocm_dequant_fn = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float16, f"Expected float16 input, got {x.dtype}"
        x = torch.ops.rotation.rotate(x, self.pairs, self.theta, self.channel_scales)
        if WQLinearMMFunction is not None:
            y = WQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )
            return y.reshape(*x.shape[:-1], self.out_features)
        if rocm_awq_linear is None or rocm_load_awq_extension is None:
            cache = os.environ.get("PAROQUANT_CUDA_CACHE_DENSE_DEQUANT", "0") != "0"
            weight = self._cached_dequant_weight if cache else None
            if weight is None or weight.device != x.device or weight.dtype != x.dtype:
                weight = _torch_dequantize_awq(
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.w_bit,
                    self.group_size,
                    dtype=x.dtype,
                )
                if cache:
                    self._cached_dequant_weight = weight
            y = torch.mm(x.reshape(-1, x.shape[-1]), weight)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*x.shape[:-1], self.out_features)
        rows = x.numel() // x.shape[-1]
        if self._direct_gemv_candidate and 0 < rows <= self._gemv_max_rows:
            gemv_fn = self._rocm_gemv_fn
            if gemv_fn is None:
                ext = rocm_load_awq_extension()
                gemv_fn = ext.gemv_awq
                object.__setattr__(self, "_rocm_gemv_fn", gemv_fn)
            return gemv_fn(x, self.qweight, self.qzeros, self.scales, self.bias, self.w_bit, self.group_size)
        if self._cache_dense_dequant and rocm_dequantize_awq is not None:
            weight = self._cached_dequant_weight
            if weight is None or weight.device != x.device or weight.dtype != x.dtype:
                dequant_fn = self._rocm_dequant_fn
                if dequant_fn is None:
                    ext = rocm_load_awq_extension()
                    dequant_fn = ext.dequant_awq
                    object.__setattr__(self, "_rocm_dequant_fn", dequant_fn)
                weight = dequant_fn(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size)
                self._cached_dequant_weight = weight
            y = torch.mm(x.reshape(-1, x.shape[-1]), weight)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*x.shape[:-1], self.out_features)
        return rocm_awq_linear(
            x,
            self.qweight,
            self.qzeros,
            self.scales,
            self.w_bit,
            self.group_size,
            self.bias,
        )
