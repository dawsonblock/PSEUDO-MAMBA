#include <torch/extension.h>
#include <vector>

// ------------------- Forward / Backward CPU -------------------

torch::Tensor pseudo_mamba_forward_cpu(torch::Tensor x, torch::Tensor h) {
  // x, h: [B, D]
  // Simple placeholder: y = tanh(x + h)
  auto y = torch::tanh(x + h);
  return y;
}

std::vector<torch::Tensor> pseudo_mamba_backward_cpu(torch::Tensor grad_out,
                                                     torch::Tensor x,
                                                     torch::Tensor h,
                                                     torch::Tensor y) {
  // dy/dx = dy/dh = (1 - y^2)
  auto one = torch::ones_like(y);
  auto dy = one - y * y;
  auto grad_common = grad_out * dy; // [B, D]
  auto grad_x = grad_common.clone();
  auto grad_h = grad_common.clone();
  return {grad_x, grad_h};
}

// ------------------- CUDA declarations -------------------

// Implemented in pseudo_mamba_ext_kernel.cu
torch::Tensor pseudo_mamba_forward_cuda(torch::Tensor x, torch::Tensor h);
std::vector<torch::Tensor> pseudo_mamba_backward_cuda(torch::Tensor grad_out,
                                                      torch::Tensor x,
                                                      torch::Tensor h,
                                                      torch::Tensor y);

// ------------------- Unified API -------------------

torch::Tensor pseudo_mamba_forward(torch::Tensor x, torch::Tensor h) {
  TORCH_CHECK(x.sizes() == h.sizes(), "x and h must have same shape");
  TORCH_CHECK(x.dim() == 2, "Expected [B, D] tensors");

  if (x.is_cuda()) {
    return pseudo_mamba_forward_cuda(x, h);
  } else {
    return pseudo_mamba_forward_cpu(x, h);
  }
}

std::vector<torch::Tensor> pseudo_mamba_backward(torch::Tensor grad_out,
                                                 torch::Tensor x,
                                                 torch::Tensor h,
                                                 torch::Tensor y) {
  TORCH_CHECK(grad_out.sizes() == y.sizes(),
              "grad_out and y must have same shape");

  if (grad_out.is_cuda()) {
    return pseudo_mamba_backward_cuda(grad_out, x, h, y);
  } else {
    return pseudo_mamba_backward_cpu(grad_out, x, h, y);
  }
}

// ------------------- PyBind module -------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pseudo_mamba_forward", &pseudo_mamba_forward,
        "Pseudo-Mamba forward (x, h -> y)");
  m.def("pseudo_mamba_backward", &pseudo_mamba_backward,
        "Pseudo-Mamba backward (grad_out, x, h, y -> grad_x, grad_h)");
}
