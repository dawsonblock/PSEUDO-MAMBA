#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void pseudo_mamba_forward_kernel(const scalar_t *__restrict__ x,
                                            const scalar_t *__restrict__ h,
                                            scalar_t *__restrict__ y,
                                            int64_t N) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;

  // Vectorized access for float (4 elements at a time)
  // Only supported if N is divisible by 4 and pointers are aligned
  // For simplicity in this benchmark kernel, we'll stick to scalar grid-stride
  // loop to avoid strict alignment requirements causing crashes in diverse
  // envs. But we WILL use grid-stride loop which is better than simple if
  // check.

  for (int64_t i = idx; i < N; i += stride) {
    const scalar_t v = x[i] + h[i];
    y[i] = tanh(v);
  }
}

template <typename scalar_t>
__global__ void pseudo_mamba_forward_with_state_kernel(
    const scalar_t *__restrict__ x, const scalar_t *__restrict__ h,
    scalar_t *__restrict__ y, scalar_t *__restrict__ v_out, int64_t N) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = idx; i < N; i += stride) {
    const scalar_t v = x[i] + h[i];
    v_out[i] = v;
    y[i] = tanh(v);
  }
}

template <typename scalar_t>
__global__ void pseudo_mamba_backward_kernel(
    const scalar_t *__restrict__ grad_out, const scalar_t *__restrict__ y,
    scalar_t *__restrict__ grad_x, scalar_t *__restrict__ grad_h, int64_t N) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = idx; i < N; i += stride) {
    const scalar_t yy = y[i];
    const scalar_t dy = static_cast<scalar_t>(1.0) - yy * yy;
    const scalar_t g = grad_out[i] * dy;
    grad_x[i] = g;
    grad_h[i] = g;
  }
}

} // namespace

torch::Tensor pseudo_mamba_forward_cuda(torch::Tensor x, torch::Tensor h) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
  TORCH_CHECK(h.is_cuda(), "h must be CUDA tensor");
  TORCH_CHECK(x.sizes() == h.sizes(), "x and h must have same shape");

  auto y = torch::empty_like(x);
  const int64_t N = x.numel();
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "pseudo_mamba_forward_cuda", [&] {
        pseudo_mamba_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), h.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(), N);
      });

  return y;
}

std::vector<torch::Tensor>
pseudo_mamba_forward_with_state_cuda(torch::Tensor x, torch::Tensor h) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
  TORCH_CHECK(h.is_cuda(), "h must be CUDA tensor");
  TORCH_CHECK(x.sizes() == h.sizes(), "x and h must have same shape");

  auto y = torch::empty_like(x);
  auto v = torch::empty_like(x);
  const int64_t N = x.numel();
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "pseudo_mamba_forward_with_state_cuda", [&] {
        pseudo_mamba_forward_with_state_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), h.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), N);
      });

  return {y, v};
}

std::vector<torch::Tensor> pseudo_mamba_backward_cuda(torch::Tensor grad_out,
                                                      torch::Tensor x,
                                                      torch::Tensor h,
                                                      torch::Tensor y) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be CUDA tensor");

  auto grad_x = torch::empty_like(x);
  auto grad_h = torch::empty_like(h);

  const int64_t N = grad_out.numel();
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "pseudo_mamba_backward_cuda", [&] {
        pseudo_mamba_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(), grad_h.data_ptr<scalar_t>(), N);
      });

  return {grad_x, grad_h};
}
