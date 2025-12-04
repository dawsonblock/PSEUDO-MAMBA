"""
Test autograd equivalence between CUDA extension and PyTorch fallback.

This ensures the Pseudo-Mamba CUDA kernel produces identical forward and
backward passes compared to the pure PyTorch implementation.
"""

import torch
import pytest


def test_pseudo_mamba_forward_equivalence():
    """Verify CUDA extension matches PyTorch forward pass."""
    try:
        import pseudo_mamba_ext
        HAS_EXT = True
    except ImportError:
        pytest.skip("CUDA extension not available")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch_size, hidden_dim = 16, 128
    x = torch.randn(batch_size, hidden_dim, device='cuda', requires_grad=True)
    h = torch.randn(batch_size, hidden_dim, device='cuda', requires_grad=True)
    
    # CUDA path
    y_cuda = pseudo_mamba_ext.pseudo_mamba_forward(x, h)
    
    # PyTorch fallback
    y_torch = torch.tanh(x + h)
    
    # Forward equivalence
    torch.testing.assert_close(y_cuda, y_torch, rtol=1e-5, atol=1e-5)


def test_pseudo_mamba_backward_equivalence():
    """Verify CUDA extension matches PyTorch backward pass."""
    try:
        import pseudo_mamba_ext
        HAS_EXT = True
    except ImportError:
        pytest.skip("CUDA extension not available")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch_size, hidden_dim = 16, 128
    
    # Test backward equivalence
    for _ in range(3):  # Multiple random seeds
        x_cuda = torch.randn(batch_size, hidden_dim, device='cuda', requires_grad=True)
        h_cuda = torch.randn(batch_size, hidden_dim, device='cuda', requires_grad=True)
        
        x_torch = x_cuda.clone().detach().requires_grad_(True)
        h_torch = h_cuda.clone().detach().requires_grad_(True)
        
        # CUDA forward
        y_cuda = pseudo_mamba_ext.pseudo_mamba_forward(x_cuda, h_cuda)
        
        # PyTorch forward
        y_torch = torch.tanh(x_torch + h_torch)
        
        # Same gradient output
        grad_out = torch.randn_like(y_cuda)
        
        # CUDA backward
        y_cuda.backward(grad_out)
        grad_x_cuda = x_cuda.grad.clone()
        grad_h_cuda = h_cuda.grad.clone()
        
        # PyTorch backward
        y_torch.backward(grad_out)
        grad_x_torch = x_torch.grad
        grad_h_torch = h_torch.grad
        
        # Backward equivalence
        torch.testing.assert_close(grad_x_cuda, grad_x_torch, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad_h_cuda, grad_h_torch, rtol=1e-5, atol=1e-5)


def test_pseudo_mamba_cpu_fallback():
    """Verify CPU fallback works correctly."""
    from pseudo_mamba.kernels.pseudo_mamba.pseudo_mamba_core import pseudo_mamba_forward
    
    batch_size, hidden_dim = 8, 64
    x = torch.randn(batch_size, hidden_dim, requires_grad=True)
    h = torch.randn(batch_size, hidden_dim, requires_grad=True)
    
    # Should use PyTorch fallback on CPU
    y = pseudo_mamba_forward(x, h)
    
    # Verify it's tanh(x + h)
    y_expected = torch.tanh(x + h)
    torch.testing.assert_close(y, y_expected, rtol=1e-6, atol=1e-6)
    
    # Verify backward
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    
    assert x.grad is not None
    assert h.grad is not None


@pytest.mark.parametrize("batch_size,hidden_dim", [
    (1, 32),
    (8, 64),
    (16, 128),
    (32, 256),
])
def test_pseudo_mamba_shapes(batch_size, hidden_dim):
    """Test various input shapes."""
    try:
        import pseudo_mamba_ext
        HAS_EXT = True
    except ImportError:
        pytest.skip("CUDA extension not available")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x = torch.randn(batch_size, hidden_dim, device='cuda')
    h = torch.randn(batch_size, hidden_dim, device='cuda')
    
    y = pseudo_mamba_ext.pseudo_mamba_forward(x, h)
    
    assert y.shape == (batch_size, hidden_dim)
    assert y.device == x.device
    assert y.dtype == x.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
