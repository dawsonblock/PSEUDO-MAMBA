"""
State Tracing for Mamba Layers

Provides tools to trace the internal state evolution of Mamba layers over time,
enabling black-box analysis and visualization of SSM dynamics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from mamba_ssm.modules.mamba_simple import Mamba


@dataclass
class LayerTrace:
    """
    Container for a single Mamba layer's state trajectory.

    Attributes:
        outputs: Layer outputs [L, B, D] over sequence
        conv_states: Convolution state trajectory [L+1, B, D*expand, d_conv]
        ssm_states: SSM state trajectory [L+1, B, D*expand, d_state]
    """
    outputs: Tensor      # [L, B, D]
    conv_states: Tensor  # [L+1, B, D*expand, d_conv]
    ssm_states: Tensor   # [L+1, B, D*expand, d_state]

    def to(self, device):
        """Move all tensors to specified device."""
        return LayerTrace(
            outputs=self.outputs.to(device),
            conv_states=self.conv_states.to(device),
            ssm_states=self.ssm_states.to(device)
        )


@torch.no_grad()
def trace_layer(
    layer: Mamba,
    hidden_states: Tensor,  # [B, L, D]
    initial_conv_state: Optional[Tensor] = None,
    initial_ssm_state: Optional[Tensor] = None,
) -> LayerTrace:
    """
    Trace a single Mamba layer's internal state evolution over a sequence.

    This function runs the layer step-by-step and records the state at each timestep,
    enabling detailed analysis of the layer's memory dynamics.

    Args:
        layer: Mamba layer to trace
        hidden_states: Input sequence [B, L, D]
        initial_conv_state: Optional initial conv state [B, D*expand, d_conv]
        initial_ssm_state: Optional initial SSM state [B, D*expand, d_state]

    Returns:
        LayerTrace containing outputs and state trajectories

    Example:
        >>> layer = Mamba(d_model=256, d_state=16, d_conv=4)
        >>> x = torch.randn(2, 100, 256)  # [B=2, L=100, D=256]
        >>> trace = trace_layer(layer, x)
        >>> print(trace.ssm_states.shape)  # [101, 2, 512, 16]
    """
    B, L, D = hidden_states.shape
    device = hidden_states.device

    # Allocate initial states if not provided
    if initial_conv_state is None or initial_ssm_state is None:
        conv_state, ssm_state = layer.allocate_inference_cache(
            batch_size=B,
            max_seqlen=L,
        )
        if initial_conv_state is not None:
            conv_state = initial_conv_state
        if initial_ssm_state is not None:
            ssm_state = initial_ssm_state
    else:
        conv_state = initial_conv_state
        ssm_state = initial_ssm_state

    D_inner = layer.d_inner  # == d_model * expand

    # Allocate trajectory buffers
    conv_traj = hidden_states.new_zeros(L + 1, B, D_inner, layer.d_conv)
    ssm_traj = hidden_states.new_zeros(L + 1, B, D_inner, layer.d_state)
    out_traj = hidden_states.new_zeros(L, B, D)

    # Record initial state
    conv_traj[0].copy_(conv_state)
    ssm_traj[0].copy_(ssm_state)

    # Step through sequence
    for t in range(L):
        x_t = hidden_states[:, t:t+1, :]  # [B, 1, D]
        y_t, conv_state, ssm_state = layer.step(x_t, conv_state, ssm_state)
        out_traj[t].copy_(y_t.squeeze(1))
        conv_traj[t+1].copy_(conv_state)
        ssm_traj[t+1].copy_(ssm_state)

    return LayerTrace(
        outputs=out_traj,
        conv_states=conv_traj,
        ssm_states=ssm_traj,
    )


@dataclass
class ModelTrace:
    """
    Container for full model state trajectories across all layers.

    Attributes:
        layer_traces: Dict mapping layer_idx to LayerTrace
        embeddings: Optional input embeddings [L, B, D]
    """
    layer_traces: Dict[int, LayerTrace]
    embeddings: Optional[Tensor] = None


@torch.no_grad()
def trace_model(
    model,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
) -> ModelTrace:
    """
    Trace all Mamba layers in a model.

    Args:
        model: Model containing Mamba layers (e.g., MambaLMHeadModel)
        input_ids: Optional input token IDs [B, L]
        inputs_embeds: Optional pre-computed embeddings [B, L, D]

    Returns:
        ModelTrace containing per-layer traces

    Note:
        This is a basic implementation. For more complex models with
        residual connections, layer norms, etc., you may need to adapt
        the tracing logic to match your model architecture.
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Must provide either input_ids or inputs_embeds")

    # Get embeddings
    if inputs_embeds is None:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'embedding'):
            inputs_embeds = model.backbone.embedding(input_ids)
        elif hasattr(model, 'embedding'):
            inputs_embeds = model.embedding(input_ids)
        else:
            raise ValueError("Cannot find embedding layer in model")

    layer_traces = {}
    hidden_states = inputs_embeds

    # Find and trace each Mamba layer
    for module in model.modules():
        if isinstance(module, Mamba) and module.layer_idx is not None:
            trace = trace_layer(module, hidden_states)
            layer_traces[module.layer_idx] = trace
            # Use the outputs for next layer (simplified, ignores residuals/norms)
            hidden_states = trace.outputs.transpose(0, 1)  # [L, B, D] -> [B, L, D]

    return ModelTrace(
        layer_traces=layer_traces,
        embeddings=inputs_embeds
    )


def extract_ssm_state_sequence(trace: LayerTrace, batch_idx: int = 0) -> Tensor:
    """
    Extract SSM state sequence for a single batch element.

    Args:
        trace: LayerTrace from trace_layer
        batch_idx: Which batch element to extract (default: 0)

    Returns:
        Tensor [L+1, D*expand, d_state]
    """
    return trace.ssm_states[:, batch_idx, :, :]


def extract_conv_state_sequence(trace: LayerTrace, batch_idx: int = 0) -> Tensor:
    """
    Extract convolution state sequence for a single batch element.

    Args:
        trace: LayerTrace from trace_layer
        batch_idx: Which batch element to extract (default: 0)

    Returns:
        Tensor [L+1, D*expand, d_conv]
    """
    return trace.conv_states[:, batch_idx, :, :]


def save_trace(trace: LayerTrace, path: str) -> None:
    """
    Save a LayerTrace to disk.

    Args:
        trace: LayerTrace to save
        path: File path
    """
    torch.save({
        'outputs': trace.outputs,
        'conv_states': trace.conv_states,
        'ssm_states': trace.ssm_states,
    }, path)


def load_trace(path: str, device=None) -> LayerTrace:
    """
    Load a LayerTrace from disk.

    Args:
        path: File path
        device: Optional device to load to

    Returns:
        Loaded LayerTrace
    """
    data = torch.load(path, map_location=device)
    return LayerTrace(
        outputs=data['outputs'],
        conv_states=data['conv_states'],
        ssm_states=data['ssm_states'],
    )
