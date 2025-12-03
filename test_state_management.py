#!/usr/bin/env python3
"""
Quick verification test for Mamba state management patches.

Tests:
1. Python API - State extraction and restoration
2. Shape validation
3. Device transfer
4. Basic functional correctness

Run after installing mamba-ssm with patches:
    cd /Users/dawsonblock/Downloads/mamba-main
    pip install -e .
    python test_state_management.py
"""

import torch
import torch.nn as nn


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from mamba_ssm.modules.mamba_simple import Mamba, MambaInferenceState
        from mamba_ssm.utils.generation import InferenceParams
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  → Make sure mamba-ssm is installed: pip install -e .")
        return False


def test_state_extraction():
    """Test basic state extraction."""
    print("\nTesting state extraction...")
    from mamba_ssm.modules.mamba_simple import Mamba, MambaInferenceState
    from mamba_ssm.utils.generation import InferenceParams
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = Mamba(d_model=128, d_state=16, expand=2, layer_idx=0).to(device)
    
    # Create input
    batch, seqlen, dim = 2, 50, 128
    x = torch.randn(batch, seqlen, dim, device=device)
    
    # Forward pass without inference params (stateless)
    y1 = model(x)
    assert y1.shape == (batch, seqlen, dim), f"Output shape mismatch: {y1.shape}"
    print(f"  ✓ Stateless forward: {y1.shape}")
    
    # Forward pass with inference params
    params = InferenceParams(max_seqlen=seqlen, max_batch_size=batch)
    y2 = model(x, inference_params=params)
    assert y2.shape == (batch, seqlen, dim)
    print(f"  ✓ Stateful forward: {y2.shape}")
    
    # Extract state
    state = model.get_inference_state(params)
    assert state is not None, "State should not be None"
    
    d_inner = 128 * 2  # d_model * expand
    d_conv = 4  # default
    d_state = 16
    
    assert state.conv_state.shape == (batch, d_inner, d_conv - 1), \
        f"Conv state shape: {state.conv_state.shape}"
    assert state.ssm_state.shape == (batch, d_inner, d_state), \
        f"SSM state shape: {state.ssm_state.shape}"
    
    print(f"  ✓ State extracted: conv={state.conv_state.shape}, ssm={state.ssm_state.shape}")
    return True


def test_state_restoration():
    """Test state save/restore."""
    print("\nTesting state restoration...")
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.utils.generation import InferenceParams
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Mamba(d_model=128, layer_idx=0).to(device)
    x1 = torch.randn(2, 50, 128, device=device)
    
    # First sequence
    params1 = InferenceParams(max_seqlen=50, max_batch_size=2)
    y1 = model(x1, inference_params=params1)
    state1 = model.get_inference_state(params1)
    
    # Second sequence (continued)
    x2 = torch.randn(2, 50, 128, device=device)
    params1.seqlen_offset = 50  # Continue from where we left off
    y2_continued = model(x2, inference_params=params1)
    
    # Second sequence (fresh start)
    params2 = InferenceParams(max_seqlen=50, max_batch_size=2)
    y2_fresh = model(x2, inference_params=params2)
    
    # They should be different (state matters)
    diff = (y2_continued - y2_fresh).abs().mean().item()
    assert diff > 1e-5, f"Outputs too similar (diff={diff}), state might not be working"
    print(f"  ✓ State affects output (diff={diff:.6f})")
    
    # Restore state1 to params2
    params2.seqlen_offset = 0  # Reset to beginning
    model.set_inference_state(state1, params2)
    state2 = model.get_inference_state(params2)
    
    # Verify restoration
    conv_diff = (state1.conv_state - state2.conv_state).abs().max().item()
    ssm_diff = (state1.ssm_state - state2.ssm_state).abs().max().item()
    
    assert conv_diff < 1e-6, f"Conv state not restored correctly (diff={conv_diff})"
    assert ssm_diff < 1e-6, f"SSM state not restored correctly (diff={ssm_diff})"
    
    print(f"  ✓ State restored correctly")
    return True


def test_device_transfer():
    """Test state device transfer."""
    print("\nTesting device transfer...")
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.utils.generation import InferenceParams
    
    if not torch.cuda.is_available():
        print("  ⊘ Skipping (CUDA not available)")
        return True
    
    model = Mamba(d_model=64, layer_idx=0).cuda()
    x = torch.randn(1, 20, 64, device='cuda')
    
    params = InferenceParams(max_seqlen=20, max_batch_size=1)
    _ = model(x, inference_params=params)
    
    state_gpu = model.get_inference_state(params)
    assert state_gpu.conv_state.device.type == 'cuda'
    
    # Transfer to CPU
    state_cpu = state_gpu.to(device='cpu')
    assert state_cpu.conv_state.device.type == 'cpu'
    assert state_cpu.ssm_state.device.type == 'cpu'
    
    # Transfer back to GPU
    state_gpu2 = state_cpu.to(device='cuda')
    assert state_gpu2.conv_state.device.type == 'cuda'
    
    print("  ✓ Device transfer works")
    return True


def test_shape_validation():
    """Test that shape mismatches are caught."""
    print("\nTesting shape validation...")
    from mamba_ssm.modules.mamba_simple import Mamba, MambaInferenceState
    from mamba_ssm.utils.generation import InferenceParams
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Mamba(d_model=128, layer_idx=0).to(device)
    params = InferenceParams(max_seqlen=20, max_batch_size=2)
    
    # Create wrong-sized state
    wrong_state = MambaInferenceState(
        conv_state=torch.zeros(3, 256, 3, device=device),  # batch=3 instead of 2
        ssm_state=torch.zeros(3, 256, 16, device=device)
    )
    
    try:
        model.set_inference_state(wrong_state, params)
        print("  ✗ Should have raised ValueError for shape mismatch")
        return False
    except ValueError as e:
        if "mismatch" in str(e).lower():
            print(f"  ✓ Shape validation works: {e}")
            return True
        else:
            print(f"  ✗ Unexpected error: {e}")
            return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 80)
    print("Mamba State Management Verification")
    print("=" * 80)
    
    tests = [
        ("Import test", test_imports),
        ("State extraction", test_state_extraction),
        ("State restoration", test_state_restoration),
        ("Device transfer", test_device_transfer),
        ("Shape validation", test_shape_validation)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests passed! State management patches are working correctly.")
        print("=" * 80)
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
