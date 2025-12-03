"""
Test model-level state management APIs

Tests for mamba_ssm.utils.state_lab module, which provides
utilities for managing state across all Mamba layers in a model.
"""

import pytest
import torch

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.utils.state_lab import (
    MambaModelState,
    get_model_state,
    set_model_state,
    zero_model_state,
    clone_model_state,
    save_model_state,
    load_model_state,
)


class SimpleMambaModel(torch.nn.Module):
    """Simple multi-layer Mamba model for testing."""

    def __init__(self, d_model=64, n_layers=3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])

    def forward(self, x, inference_params=None):
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        return x


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_get_model_state_basic(device, dtype):
    """Test basic get_model_state functionality."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    model = SimpleMambaModel(d_model=64, n_layers=3).to(device=device, dtype=dtype)

    batch_size = 2
    seqlen = 10
    x = torch.randn(batch_size, seqlen, 64, device=device, dtype=dtype)

    # Create inference params and run forward to initialize state
    inference_params = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    _ = model(x, inference_params=inference_params)

    # Get model state
    state = get_model_state(model, inference_params)

    # Check that state was extracted
    assert isinstance(state, MambaModelState)
    assert len(state.layer_states) == 3
    for i in range(3):
        assert i in state.layer_states
        layer_state = state.layer_states[i]
        assert layer_state.conv_state.shape[0] == batch_size
        assert layer_state.ssm_state.shape[0] == batch_size


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_set_model_state(device):
    """Test set_model_state functionality."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    model = SimpleMambaModel(d_model=64, n_layers=3).to(device=device)

    batch_size = 2
    seqlen = 10
    x = torch.randn(batch_size, seqlen, 64, device=device)

    # Run forward twice to get two different states
    inference_params1 = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    out1 = model(x, inference_params=inference_params1)
    state1 = get_model_state(model, inference_params1)

    inference_params2 = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    out2 = model(x * 2, inference_params=inference_params2)  # Different input
    state2 = get_model_state(model, inference_params2)

    # States should be different
    for i in range(3):
        assert not torch.allclose(
            state1.layer_states[i].ssm_state,
            state2.layer_states[i].ssm_state
        )

    # Set state1 into inference_params2
    set_model_state(model, state1, inference_params2)
    state2_restored = get_model_state(model, inference_params2)

    # Now they should match
    for i in range(3):
        assert torch.allclose(
            state1.layer_states[i].conv_state,
            state2_restored.layer_states[i].conv_state,
            atol=1e-5
        )
        assert torch.allclose(
            state1.layer_states[i].ssm_state,
            state2_restored.layer_states[i].ssm_state,
            atol=1e-5
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_model_state(device):
    """Test zero_model_state functionality."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    model = SimpleMambaModel(d_model=64, n_layers=3).to(device=device)

    batch_size = 2
    seqlen = 10
    x = torch.randn(batch_size, seqlen, 64, device=device)

    # Run forward to initialize non-zero state
    inference_params = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    _ = model(x, inference_params=inference_params)

    # Get state before zeroing
    state_before = get_model_state(model, inference_params)

    # Check that state is non-zero
    for i in range(3):
        assert state_before.layer_states[i].ssm_state.abs().max() > 0

    # Zero the state
    zero_model_state(model, inference_params)

    # Get state after zeroing
    state_after = get_model_state(model, inference_params)

    # Check that all states are zero
    for i in range(3):
        assert torch.allclose(
            state_after.layer_states[i].conv_state,
            torch.zeros_like(state_after.layer_states[i].conv_state)
        )
        assert torch.allclose(
            state_after.layer_states[i].ssm_state,
            torch.zeros_like(state_after.layer_states[i].ssm_state)
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_clone_model_state(device):
    """Test clone_model_state functionality."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    model = SimpleMambaModel(d_model=64, n_layers=3).to(device=device)

    batch_size = 2
    seqlen = 10
    x = torch.randn(batch_size, seqlen, 64, device=device)

    # Run forward to initialize state
    inference_params = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    _ = model(x, inference_params=inference_params)

    # Get and clone state
    state = get_model_state(model, inference_params)
    cloned_state = clone_model_state(state)

    # Check that values match
    for i in range(3):
        assert torch.allclose(
            state.layer_states[i].conv_state,
            cloned_state.layer_states[i].conv_state
        )
        assert torch.allclose(
            state.layer_states[i].ssm_state,
            cloned_state.layer_states[i].ssm_state
        )

    # Check that tensors are different objects
    for i in range(3):
        assert state.layer_states[i].conv_state.data_ptr() != \
               cloned_state.layer_states[i].conv_state.data_ptr()
        assert state.layer_states[i].ssm_state.data_ptr() != \
               cloned_state.layer_states[i].ssm_state.data_ptr()

    # Modify original
    state.layer_states[0].ssm_state.zero_()

    # Cloned should be unchanged
    assert not torch.allclose(
        state.layer_states[0].ssm_state,
        cloned_state.layer_states[0].ssm_state
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_save_load_model_state(device, tmp_path):
    """Test save_model_state and load_model_state."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    model = SimpleMambaModel(d_model=64, n_layers=3).to(device=device)

    batch_size = 2
    seqlen = 10
    x = torch.randn(batch_size, seqlen, 64, device=device)

    # Run forward to initialize state
    inference_params = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    _ = model(x, inference_params=inference_params)

    # Get state
    state = get_model_state(model, inference_params)

    # Save and load
    save_path = tmp_path / "model_state.pt"
    save_model_state(state, str(save_path))
    loaded_state = load_model_state(str(save_path), device=device)

    # Check that values match
    assert len(loaded_state.layer_states) == len(state.layer_states)
    for i in range(3):
        assert torch.allclose(
            state.layer_states[i].conv_state,
            loaded_state.layer_states[i].conv_state,
            atol=1e-5
        )
        assert torch.allclose(
            state.layer_states[i].ssm_state,
            loaded_state.layer_states[i].ssm_state,
            atol=1e-5
        )


def test_model_state_to_device():
    """Test MambaModelState.to() method."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = SimpleMambaModel(d_model=64, n_layers=3).to(device="cpu")

    batch_size = 2
    seqlen = 10
    x = torch.randn(batch_size, seqlen, 64)

    # Run forward to initialize state
    inference_params = InferenceParams(max_batch_size=batch_size, max_seqlen=seqlen)
    _ = model(x, inference_params=inference_params)

    # Get state (on CPU)
    state_cpu = get_model_state(model, inference_params)

    # Move to CUDA
    state_cuda = state_cpu.to(device="cuda")

    # Check devices
    for i in range(3):
        assert state_cpu.layer_states[i].conv_state.device.type == "cpu"
        assert state_cuda.layer_states[i].conv_state.device.type == "cuda"

        # Values should match
        assert torch.allclose(
            state_cpu.layer_states[i].conv_state,
            state_cuda.layer_states[i].conv_state.cpu(),
            atol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
