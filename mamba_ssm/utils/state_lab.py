"""
Model-level State Orchestration for Mamba

Provides utilities for managing inference state across all Mamba layers in a model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from mamba_ssm.modules.mamba_simple import Mamba, MambaInferenceState


@dataclass
class MambaModelState:
    """
    Container for per-layer MambaInferenceState, keyed by layer_idx.

    This class holds the complete state for all Mamba layers in a model,
    enabling easy save/load/reset operations at the model level.
    """
    layer_states: Dict[int, MambaInferenceState]

    def to(self, device=None, dtype=None):
        """Move all layer states to specified device and/or dtype."""
        return MambaModelState(
            layer_states={
                idx: state.to(device=device, dtype=dtype)
                for idx, state in self.layer_states.items()
            }
        )


def iter_mamba_layers(model) -> List[Mamba]:
    """
    Find all Mamba layers in a model that have a layer_idx set.

    Args:
        model: PyTorch module (typically a MambaLMHeadModel or custom model)

    Yields:
        Mamba layers with non-None layer_idx
    """
    for module in model.modules():
        if isinstance(module, Mamba) and module.layer_idx is not None:
            yield module


def get_model_state(model, inference_params) -> MambaModelState:
    """
    Extract the current state for all Mamba layers in the model.

    Args:
        model: PyTorch module containing Mamba layers
        inference_params: InferenceParams object with key_value_memory_dict

    Returns:
        MambaModelState containing all layer states
    """
    layer_states = {}
    for layer in iter_mamba_layers(model):
        state = layer.get_inference_state(inference_params)
        if state is not None:
            layer_states[layer.layer_idx] = state
    return MambaModelState(layer_states=layer_states)


def set_model_state(model, state: MambaModelState, inference_params) -> None:
    """
    Restore state for all Mamba layers in the model.

    Args:
        model: PyTorch module containing Mamba layers
        state: MambaModelState with saved layer states
        inference_params: InferenceParams object to update
    """
    for layer in iter_mamba_layers(model):
        if layer.layer_idx in state.layer_states:
            layer.set_inference_state(
                state.layer_states[layer.layer_idx],
                inference_params
            )


def zero_model_state(model, inference_params) -> None:
    """
    Zero out the state for all Mamba layers in the model.

    Useful for episode resets in RL or clearing context between generations.

    Args:
        model: PyTorch module containing Mamba layers
        inference_params: InferenceParams object to update
    """
    for layer in iter_mamba_layers(model):
        layer.zero_inference_state(inference_params)


def clone_model_state(state: MambaModelState) -> MambaModelState:
    """
    Deep clone a MambaModelState.

    Args:
        state: MambaModelState to clone

    Returns:
        A new MambaModelState with cloned tensors
    """
    return MambaModelState(
        layer_states={
            idx: MambaInferenceState(
                conv_state=layer_state.conv_state.clone(),
                ssm_state=layer_state.ssm_state.clone()
            )
            for idx, layer_state in state.layer_states.items()
        }
    )


def save_model_state(state: MambaModelState, path: str) -> None:
    """
    Save a MambaModelState to disk.

    Args:
        state: MambaModelState to save
        path: File path to save to
    """
    torch.save({
        'layer_states': {
            idx: {
                'conv_state': layer_state.conv_state,
                'ssm_state': layer_state.ssm_state
            }
            for idx, layer_state in state.layer_states.items()
        }
    }, path)


def load_model_state(path: str, device=None) -> MambaModelState:
    """
    Load a MambaModelState from disk.

    Args:
        path: File path to load from
        device: Optional device to load tensors to

    Returns:
        Loaded MambaModelState
    """
    data = torch.load(path, map_location=device)
    layer_states = {}
    for idx, layer_data in data['layer_states'].items():
        layer_states[idx] = MambaInferenceState(
            conv_state=layer_data['conv_state'],
            ssm_state=layer_data['ssm_state']
        )
    return MambaModelState(layer_states=layer_states)
