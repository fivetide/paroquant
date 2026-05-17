from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


_AWQ_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace {

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

__device__ inline int awq_shift_for_output_col(int out_col) {
  const int d = out_col & 7;
  const int packed_pos = (d & 1) ? (4 + (d >> 1)) : (d >> 1);
  return packed_pos * 4;
}

template <typename scalar_t>
__global__ void dequant_awq_kernel(
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = in_features * out_features;
  if (linear >= total) {
    return;
  }

  const int64_t in_col = linear / out_features;
  const int64_t out_col = linear - in_col * out_features;
  const int64_t out_pack = out_col >> 3;
  const int shift = awq_shift_for_output_col(static_cast<int>(out_col));
  const int64_t group = in_col / group_size;

  const uint32_t packed_w = static_cast<uint32_t>(qweight[in_col * out_packed + out_pack]);
  const uint32_t packed_z = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
  const int q = static_cast<int>((packed_w >> shift) & 0xF);
  const int z = static_cast<int>((packed_z >> shift) & 0xF);
  const float scale = scalar_to_float(scales[group * out_features + out_col]);
  out[linear] = float_to_scalar<scalar_t>((static_cast<float>(q - z)) * scale);
}

template <typename scalar_t>
__global__ void gemv_awq_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size,
    bool has_bias) {
  const int64_t out_col = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  extern __shared__ float scratch[];

  float acc = 0.0f;
  const int64_t out_pack = out_col >> 3;
  const int shift = awq_shift_for_output_col(static_cast<int>(out_col));
  const int64_t x_base = row * in_features;
  for (int64_t in_col = tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t packed_w = static_cast<uint32_t>(qweight[in_col * out_packed + out_pack]);
    const uint32_t packed_z = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int q = static_cast<int>((packed_w >> shift) & 0xF);
    const int z = static_cast<int>((packed_z >> shift) & 0xF);
    const float w = static_cast<float>(q - z) * scalar_to_float(scales[group * out_features + out_col]);
    acc = fmaf(scalar_to_float(x[x_base + in_col]), w, acc);
  }

  scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float y = scratch[0];
    if (has_bias) {
      y += scalar_to_float(bias[out_col]);
    }
    out[row * out_features + out_col] = float_to_scalar<scalar_t>(y);
  }
}

} // namespace

torch::Tensor dequant_awq(
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    int64_t bits,
    int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA/HIP tensor");
  TORCH_CHECK(qzeros.is_cuda(), "qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA/HIP tensor");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be [in_features, out_features / pack]");
  TORCH_CHECK(qzeros.dim() == 2, "qzeros must be [in_features / group_size, out_features / pack]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [in_features / group_size, out_features]");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "qzeros must be int32");
  TORCH_CHECK(bits == 4, "only 4-bit AWQ is currently supported");
  TORCH_CHECK(group_size > 0, "group_size must be positive");

  const int64_t in_features = qweight.size(0);
  const int64_t out_packed = qweight.size(1);
  const int64_t out_features = out_packed * 8;
  TORCH_CHECK(in_features % group_size == 0, "in_features must be divisible by group_size");
  TORCH_CHECK(qzeros.size(0) == in_features / group_size, "qzeros group dimension mismatch");
  TORCH_CHECK(qzeros.size(1) == out_packed, "qzeros packed output dimension mismatch");
  TORCH_CHECK(scales.size(0) == in_features / group_size, "scales group dimension mismatch");
  TORCH_CHECK(scales.size(1) == out_features, "scales output dimension mismatch");

  at::cuda::CUDAGuard device_guard(qweight.device());
  auto qweight_c = qweight.contiguous();
  auto qzeros_c = qzeros.contiguous();
  auto scales_c = scales.contiguous();
  auto out = torch::empty({in_features, out_features}, scales_c.options());

  const int threads = 256;
  const int64_t total = in_features * out_features;
  const dim3 grid((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scales_c.scalar_type(), "paroquant_rocm_dequant_awq", [&] {
    dequant_awq_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        qweight_c.data_ptr<int32_t>(),
        qzeros_c.data_ptr<int32_t>(),
        scales_c.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        in_features,
        out_features,
        out_packed,
        static_cast<int>(group_size));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gemv_awq(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias_opt,
    int64_t bits,
    int64_t group_size) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA/HIP tensor");
  TORCH_CHECK(qzeros.is_cuda(), "qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA/HIP tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have a trailing input dimension");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be [in_features, out_features / pack]");
  TORCH_CHECK(qzeros.dim() == 2, "qzeros must be [in_features / group_size, out_features / pack]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [in_features / group_size, out_features]");
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "qzeros must be int32");
  TORCH_CHECK(bits == 4, "only 4-bit AWQ is currently supported");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), "x and scales dtypes must match");

  const int64_t in_features = qweight.size(0);
  const int64_t out_packed = qweight.size(1);
  const int64_t out_features = out_packed * 8;
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(x.size(-1) == in_features, "x trailing dimension must equal qweight input dimension");
  TORCH_CHECK(in_features % group_size == 0, "in_features must be divisible by group_size");
  TORCH_CHECK(qzeros.size(0) == in_features / group_size, "qzeros group dimension mismatch");
  TORCH_CHECK(qzeros.size(1) == out_packed, "qzeros packed output dimension mismatch");
  TORCH_CHECK(scales.size(0) == in_features / group_size, "scales group dimension mismatch");
  TORCH_CHECK(scales.size(1) == out_features, "scales output dimension mismatch");

  bool has_bias = bias_opt.has_value() && bias_opt.value().defined() && bias_opt.value().numel() > 0;
  torch::Tensor bias;
  if (has_bias) {
    bias = bias_opt.value();
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "bias must be [out_features]");
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype must match x");
  }

  at::cuda::CUDAGuard device_guard(x.device());
  auto x_2d = x.contiguous().view({rows, in_features});
  auto qweight_c = qweight.contiguous();
  auto qzeros_c = qzeros.contiguous();
  auto scales_c = scales.contiguous();
  auto bias_c = has_bias ? bias.contiguous() : torch::Tensor();
  auto out_2d = torch::empty({rows, out_features}, x.options());

  const int threads = 256;
  const dim3 grid(out_features, rows);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_2d.scalar_type(), "paroquant_rocm_gemv_awq", [&] {
    const scalar_t* bias_ptr = has_bias ? bias_c.data_ptr<scalar_t>() : nullptr;
    gemv_awq_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
        x_2d.data_ptr<scalar_t>(),
        qweight_c.data_ptr<int32_t>(),
        qzeros_c.data_ptr<int32_t>(),
        scales_c.data_ptr<scalar_t>(),
        bias_ptr,
        out_2d.data_ptr<scalar_t>(),
        rows,
        in_features,
        out_features,
        out_packed,
        static_cast<int>(group_size),
        has_bias);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = out_features;
  return out_2d.view(out_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_awq", &dequant_awq, "Dequantize AWQ uint4 weights (ROCm/HIP)");
  m.def("gemv_awq", &gemv_awq, "Direct AWQ uint4 GEMV (ROCm/HIP)");
}
'''


def _build_directory(arch: str) -> Path:
    cache_root = Path.home() / ".cache" / "paroquant" / "torch_extensions"
    abi_tag = (
        f"py{sys.version_info.major}{sys.version_info.minor}_"
        f"torch{torch.__version__}_"
        f"hip{torch.version.hip or 'none'}_"
        f"{arch}"
    )
    abi_tag = "".join(c if c.isalnum() else "_" for c in abi_tag)
    build_dir = cache_root / "paroquant_awq_rocm_v2" / abi_tag
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


@lru_cache(maxsize=1)
def _load_awq_extension():
    if torch.version.hip is None:
        raise RuntimeError("ParoQuant ROCm AWQ requested, but this PyTorch build has no HIP support")
    if not torch.cuda.is_available():
        raise RuntimeError("ParoQuant ROCm AWQ requires a visible HIP device")

    arch = os.environ.get("PAROQUANT_HIP_ARCH", "gfx1100")
    os.environ.setdefault("PYTORCH_ROCM_ARCH", arch)
    build_dir = _build_directory(arch)
    kwargs = dict(
        name=f"paroquant_awq_rocm_v2_{arch}",
        cpp_sources="",
        cuda_sources=_AWQ_SRC,
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", f"--offload-arch={arch}", "-mcumode"],
        with_cuda=True,
        verbose=bool(int(os.environ.get("PAROQUANT_VERBOSE_BUILD", "0"))),
    )
    try:
        return load_inline(**kwargs)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        return load_inline(**kwargs)


def dequantize_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, bits: int, group_size: int):
    return _load_awq_extension().dequant_awq(qweight, qzeros, scales, int(bits), int(group_size))


def awq_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    bias: torch.Tensor | None = None,
):
    return _load_awq_extension().gemv_awq(x, qweight, qzeros, scales, bias, int(bits), int(group_size))


def _gemv_max_rows() -> int:
    return int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_ROWS", "8"))


def _gemv_min_in_features() -> int:
    return int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MIN_IN_FEATURES", "1024"))


def _gemv_max_out_features() -> int:
    return int(os.environ.get("PAROQUANT_ROCM_AWQ_GEMV_MAX_OUT_FEATURES", "1024"))


def awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.device.type != "cuda":
        raise RuntimeError("ParoQuant ROCm AWQ linear requires a CUDA/HIP tensor input")
    rows = x.numel() // x.shape[-1]
    out_features = qweight.shape[1] * (32 // bits)
    if (
        0 < rows <= _gemv_max_rows()
        and x.shape[-1] >= _gemv_min_in_features()
        and out_features <= _gemv_max_out_features()
    ):
        return awq_gemv(x, qweight, qzeros, scales, bits, group_size, bias)
    weight = dequantize_awq(qweight, qzeros, scales, bits, group_size)
    x_flat = x.reshape(-1, x.shape[-1])
    y = torch.matmul(x_flat, weight)
    if bias is not None:
        y = y + bias
    return y.reshape(*x.shape[:-1], weight.shape[1])
