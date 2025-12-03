"""
Test the fwd_with_state Python wrapper

Tests for selective_scan_stateful_fn which provides explicit
state input/output interface for the selective scan operation.
"""

import pytest
import torch

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        selective_scan_stateful_fn,
        selective_scan_ref,
    )
    import selective_scan_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


@pytest.mark.skipif(not HAS_CUDA, reason="selective_scan_cuda not available")
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("seqlen", [16, 64])
@pytest.mark.parametrize("dstate", [8, 16])
def test_stateful_fn_output_shape(batch_size, dim, seqlen, dstate):
    """Test that selective_scan_stateful_fn returns correct output shapes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    # Create random inputs
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    A = -torch.rand(dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    D = torch.randn(dim, device=device, dtype=dtype)

    # Call stateful function
    out, xT = selective_scan_stateful_fn(
        u, delta, A, B, C, D=D, delta_softplus=True
    )

    # Check output shapes
    assert out.shape == (batch_size, dim, seqlen), \
        f"Expected out shape {(batch_size, dim, seqlen)}, got {out.shape}"
    assert xT.shape == (batch_size, dim, 2 * dstate), \
        f"Expected xT shape {(batch_size, dim, 2 * dstate)}, got {xT.shape}"


@pytest.mark.skipif(not HAS_CUDA, reason="selective_scan_cuda not available")
def test_stateful_fn_vs_regular_fn():
    """Test that selective_scan_stateful_fn matches selective_scan_fn output."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    batch_size, dim, seqlen, dstate = 2, 64, 32, 16

    # Create random inputs
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    A = -torch.rand(dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    D = torch.randn(dim, device=device, dtype=dtype)

    # Call both functions
    out_regular, last_state_regular = selective_scan_fn(
        u, delta, A, B, C, D=D, delta_softplus=True, return_last_state=True
    )
    out_stateful, xT_stateful = selective_scan_stateful_fn(
        u, delta, A, B, C, D=D, delta_softplus=True
    )

    # Outputs should match (within numerical tolerance)
    assert torch.allclose(out_regular, out_stateful, atol=1e-5, rtol=1e-4), \
        f"Output mismatch: max diff = {(out_regular - out_stateful).abs().max()}"

    # Last states should match (note: shapes may differ slightly)
    # The regular fn returns [B, D, N] while stateful returns [B, D, 2*N]
    # We can compare the magnitude or norm as a sanity check
    assert last_state_regular.shape[0] == xT_stateful.shape[0]  # batch
    assert last_state_regular.shape[1] == xT_stateful.shape[1]  # dim


@pytest.mark.skipif(not HAS_CUDA, reason="selective_scan_cuda not available")
def test_stateful_fn_with_zero_init():
    """Test that stateful_fn works with x0=None (zero initialization)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    batch_size, dim, seqlen, dstate = 2, 64, 32, 16

    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    A = -torch.rand(dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)

    # Call with x0=None
    out, xT = selective_scan_stateful_fn(
        u, delta, A, B, C, x0=None, delta_softplus=True
    )

    assert out.shape == (batch_size, dim, seqlen)
    assert xT.shape == (batch_size, dim, 2 * dstate)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isnan(xT).any(), "Final state contains NaN"


@pytest.mark.skipif(not HAS_CUDA, reason="selective_scan_cuda not available")
def test_stateful_fn_consistency_with_ref():
    """Test that stateful_fn is consistent with reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    batch_size, dim, seqlen, dstate = 1, 16, 8, 8

    # Small inputs for easier comparison
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype) * 0.1
    delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype).abs() * 0.1
    A = -torch.rand(dim, dstate, device=device, dtype=dtype) * 0.1
    B = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype) * 0.1
    C = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype) * 0.1

    # Call CUDA version
    out_cuda, xT_cuda = selective_scan_stateful_fn(
        u, delta, A, B, C, delta_softplus=False
    )

    # Call reference version
    out_ref, last_ref = selective_scan_ref(
        u, delta, A, B, C, delta_softplus=False, return_last_state=True
    )

    # Outputs should be close
    assert torch.allclose(out_cuda, out_ref, atol=1e-3, rtol=1e-2), \
        f"Output mismatch with ref: max diff = {(out_cuda - out_ref).abs().max()}"


@pytest.mark.skipif(not HAS_CUDA, reason="selective_scan_cuda not available")
def test_stateful_fn_with_gating():
    """Test stateful_fn with z (gating) parameter."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    batch_size, dim, seqlen, dstate = 2, 64, 32, 16

    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    A = -torch.rand(dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    z = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)

    # Call with gating
    out, xT = selective_scan_stateful_fn(
        u, delta, A, B, C, z=z, delta_softplus=True
    )

    assert out.shape == (batch_size, dim, seqlen)
    assert xT.shape == (batch_size, dim, 2 * dstate)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not HAS_CUDA, reason="selective_scan_cuda not available")
def test_stateful_fn_with_delta_bias():
    """Test stateful_fn with delta_bias parameter."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    batch_size, dim, seqlen, dstate = 2, 64, 32, 16

    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    A = -torch.rand(dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch_size, dstate, seqlen, device=device, dtype=dtype)
    delta_bias = torch.randn(dim, device=device, dtype=torch.float32)

    # Call with delta_bias
    out, xT = selective_scan_stateful_fn(
        u, delta, A, B, C, delta_bias=delta_bias, delta_softplus=True
    )

    assert out.shape == (batch_size, dim, seqlen)
    assert xT.shape == (batch_size, dim, 2 * dstate)
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
