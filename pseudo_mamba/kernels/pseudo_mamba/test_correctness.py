import torch
import torch.nn as nn
import time

try:
    import pseudo_mamba_ext
    from pseudo_mamba_core import pseudo_mamba_forward
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("CUDA extension not found. Skipping tests.")

def test_correctness():
    if not HAS_CUDA:
        return

    B, D = 32, 128
    device = torch.device("cuda")
    
    x = torch.randn(B, D, device=device, requires_grad=True)
    h = torch.randn(B, D, device=device, requires_grad=True)
    
    # PyTorch Reference
    y_ref = torch.tanh(x + h)
    y_ref.sum().backward()
    grad_x_ref = x.grad.clone()
    grad_h_ref = h.grad.clone()
    
    x.grad.zero_()
    h.grad.zero_()
    
    # CUDA Kernel via Function
    y_cuda = pseudo_mamba_forward(x, h)
    y_cuda.sum().backward()
    grad_x_cuda = x.grad.clone()
    grad_h_cuda = h.grad.clone()
    
    # Compare
    print(f"Output Max Diff: {(y_ref - y_cuda).abs().max().item()}")
    print(f"Grad X Max Diff: {(grad_x_ref - grad_x_cuda).abs().max().item()}")
    print(f"Grad H Max Diff: {(grad_h_ref - grad_h_cuda).abs().max().item()}")
    
    assert torch.allclose(y_ref, y_cuda, atol=1e-5)
    assert torch.allclose(grad_x_ref, grad_x_cuda, atol=1e-5)
    assert torch.allclose(grad_h_ref, grad_h_cuda, atol=1e-5)
    print("Correctness Test Passed!")

if __name__ == "__main__":
    test_correctness()
