import torch
import argparse
import sys
import os

def test_introspection():
    print("--- Testing Mamba Introspection (pseudo_mamba_introspect.py) ---")
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        from pseudo_mamba_introspect import trace_mamba_sequence
        
        print("Initializing Mamba layer...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        d_model = 16
        layer = Mamba(d_model=d_model, d_state=4, d_conv=2, expand=2).to(device)
        
        B, L = 2, 10
        x = torch.randn(B, L, d_model, device=device)
        
        print(f"Running trace_mamba_sequence on {device}...")
        outputs, trace = trace_mamba_sequence(layer, x)
        
        print(f"Outputs shape: {outputs.shape}")
        print(f"Conv states shape: {trace.conv_states.shape}")
        print(f"SSM states shape: {trace.ssm_states.shape}")
        
        assert outputs.shape == (B, L, d_model)
        # trace.conv_states: [L+1, B, D*expand, d_conv]
        assert trace.conv_states.shape[0] == L + 1
        assert trace.ssm_states.shape[0] == L + 1
        
        print("✓ Introspection test passed!")
        return True
    except ImportError as e:
        print(f"⚠ Skipping introspection test: {e}")
        return False
    except Exception as e:
        print(f"✗ Introspection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triton_debug_kernel():
    print("\n--- Testing Triton Debug Kernel (selective_scan_fn_debug) ---")
    if not torch.cuda.is_available():
        print("⚠ Skipping Triton test (CUDA not available)")
        return True

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn_debug
        
        B, D, L, N = 2, 32, 16, 8
        device = "cuda"
        
        u = torch.randn(B, D, L, device=device)
        delta = torch.randn(B, D, L, device=device)
        A = -torch.rand(D, N, device=device)
        B_mat = torch.randn(B, 1, N, L, device=device)
        C_mat = torch.randn(B, 1, N, L, device=device)
        D_val = torch.randn(D, device=device)
        z = torch.randn(B, D, L, device=device)
        delta_bias = torch.randn(D, device=device)
        
        print("Running selective_scan_fn_debug...")
        out, trace = selective_scan_fn_debug(u, delta, A, B_mat, C_mat, D_val, z, delta_bias)
        
        print(f"Output shape: {out.shape}")
        print(f"Trace shape: {trace.shape}")
        
        assert out.shape == (B, D, L)
        assert trace.shape == (B, D, L, N)
        
        print("✓ Triton debug kernel test passed!")
        return True
    except ImportError as e:
        print(f"⚠ Skipping Triton test: {e}")
        return False
    except Exception as e:
        print(f"✗ Triton debug kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pseudo_mamba_ext():
    print("\n--- Testing Pseudo-Mamba Extension (C++/CUDA) ---")
    if not torch.cuda.is_available():
        print("⚠ Skipping Extension test (CUDA not available)")
        # We can test CPU path if compiled?
        # The extension exposes 'pseudo_mamba_forward' which dispatches.
        # Let's try CPU if available.
    
    try:
        import pseudo_mamba_ext
        print("pseudo_mamba_ext imported successfully.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        B, D = 2, 16
        x = torch.randn(B, D, device=device)
        h = torch.randn(B, D, device=device)
        
        print(f"Running pseudo_mamba_forward_with_state on {device}...")
        # Expecting [y, v]
        # Note: The C++ signature returns std::vector<Tensor>
        results = pseudo_mamba_ext.pseudo_mamba_forward_with_state(x, h)
        
        if isinstance(results, (list, tuple)):
            y, v = results
            print(f"y shape: {y.shape}, v shape: {v.shape}")
            assert y.shape == (B, D)
            assert v.shape == (B, D)
            
            # Check math: v = x + h, y = tanh(v)
            v_ref = x + h
            y_ref = torch.tanh(v_ref)
            
            diff_v = (v - v_ref).abs().max().item()
            diff_y = (y - y_ref).abs().max().item()
            
            print(f"Max diff v: {diff_v}")
            print(f"Max diff y: {diff_y}")
            
            if diff_v < 1e-5 and diff_y < 1e-5:
                print("✓ Extension math check passed!")
            else:
                print("✗ Extension math check failed!")
                return False
        else:
            print(f"✗ Unexpected return type: {type(results)}")
            return False
            
        return True

    except ImportError:
        print("⚠ pseudo_mamba_ext not found. Did you build it?")
        return False
    except Exception as e:
        print(f"✗ Extension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("===================================================")
    print("   Mamba Upgrade & Debug Verification Suite")
    print("===================================================")
    
    passed_intro = test_introspection()
    passed_triton = test_triton_debug_kernel()
    passed_ext = test_pseudo_mamba_ext()
    
    print("\n===================================================")
    print("Summary:")
    print(f"Introspection: {'PASS' if passed_intro else 'FAIL/SKIP'}")
    print(f"Triton Debug:  {'PASS' if passed_triton else 'FAIL/SKIP'}")
    print(f"Extension:     {'PASS' if passed_ext else 'FAIL/SKIP'}")
    print("===================================================")

if __name__ == "__main__":
    main()
