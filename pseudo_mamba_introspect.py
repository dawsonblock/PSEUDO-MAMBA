"""
Mamba introspection utilities.

Goal:
    1. Run a Mamba layer step-by-step using its .step() function.
    2. Capture internal recurrent states:
         - conv_state: [B, D*expand, d_conv]
         - ssm_state:  [B, D*expand, d_state]
    3. Return:
         - full output sequence
         - state trajectories for analysis / plotting / RL.

Usage:
    from mamba_ssm.modules.mamba_simple import Mamba
    from pseudo_mamba_introspect import trace_mamba_sequence

    m = Mamba(d_model=256, d_state=16, d_conv=4, expand=2, ...)
    x = torch.randn(B, L, 256, device="cuda")

    out, trace = trace_mamba_sequence(m, x)

    # trace.conv_states: [L+1, B, D*expand, d_conv]
    # trace.ssm_states:  [L+1, B, D*expand, d_state]
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import Tensor
from mamba_ssm.modules.mamba_simple import Mamba


@dataclass
class MambaStateTrace:
    """
    Stores a full trajectory of internal states.

    Shapes:
        outputs      : [L, B, D]
        conv_states  : [L+1, B, D*expand, d_conv]
        ssm_states   : [L+1, B, D*expand, d_state]
    """
    outputs: Tensor
    conv_states: Tensor
    ssm_states: Tensor

    def to(self, device: torch.device | str) -> "MambaStateTrace":
        return MambaStateTrace(
            outputs=self.outputs.to(device),
            conv_states=self.conv_states.to(device),
            ssm_states=self.ssm_states.to(device),
        )


def _alloc_states_from_layer(layer: Mamba, batch_size: int, max_seqlen: int) -> tuple[Tensor, Tensor]:
    """
    Use the layer's own helper to allocate states on the correct device/dtype.
    """
    # allocate_inference_cache might return a tuple or object depending on version/patch
    # Assuming the patched version returns (conv_state, ssm_state) or similar
    cache = layer.allocate_inference_cache(
        batch_size=batch_size,
        max_seqlen=max_seqlen,
    )
    if isinstance(cache, tuple):
        return cache
    # If it returns an object with attributes
    return cache.conv_state, cache.ssm_state


@torch.no_grad()
def trace_mamba_sequence(
    layer: Mamba,
    hidden_states: Tensor,
    *,
    clone_states: bool = True,
    detach: bool = True,
) -> tuple[Tensor, MambaStateTrace]:
    """
    Run a Mamba layer step-by-step and capture internal states.

    Args:
        layer:
            A mamba_ssm.modules.mamba_simple.Mamba instance.
        hidden_states:
            [B, L, D] input sequence.
        clone_states:
            If True, each timestep's states are cloned so later
            modifications don't overwrite history.
        detach:
            If True, returned tensors have requires_grad=False
            (this is a pure introspection run).

    Returns:
        outputs:
            [B, L, D] full sequence output.
        trace: MambaStateTrace with shapes:
            outputs     : [L, B, D]
            conv_states : [L+1, B, D*expand, d_conv]
            ssm_states  : [L+1, B, D*expand, d_state]
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"Expected [B, L, D] input, got {hidden_states.shape}")

    B, L, D = hidden_states.shape
    device = hidden_states.device

    # Allocate initial states using the layer's own helper.
    conv_state, ssm_state = _alloc_states_from_layer(layer, batch_size=B, max_seqlen=L)

    # Pre-allocate trajectories
    D_expand = layer.d_model * layer.expand
    conv_traj = hidden_states.new_zeros(L + 1, B, D_expand, layer.d_conv)
    ssm_traj = hidden_states.new_zeros(L + 1, B, D_expand, layer.d_state)
    out_traj = hidden_states.new_zeros(L, B, D)

    # Initial states at t=0
    conv_traj[0].copy_(conv_state)
    ssm_traj[0].copy_(ssm_state)

    # Step through sequence
    for t in range(L):
        x_t = hidden_states[:, t : t + 1, :]  # [B, 1, D]
        y_t, conv_state, ssm_state = layer.step(x_t, conv_state, ssm_state)
        # y_t: [B, D]

        if clone_states:
            conv_traj[t + 1].copy_(conv_state)
            ssm_traj[t + 1].copy_(ssm_state)
        else:
            conv_traj[t + 1] = conv_state
            ssm_traj[t + 1] = ssm_state

        out_traj[t].copy_(y_t)

    if detach:
        out_traj = out_traj.detach()
        conv_traj = conv_traj.detach()
        ssm_traj = ssm_traj.detach()

    outputs = out_traj.transpose(0, 1)  # [B, L, D]
    trace = MambaStateTrace(
        outputs=out_traj,          # [L, B, D]
        conv_states=conv_traj,     # [L+1, B, D*expand, d_conv]
        ssm_states=ssm_traj,       # [L+1, B, D*expand, d_state]
    )
    return outputs, trace


class IntrospectMambaWrapper(torch.nn.Module):
    """
    Thin nn.Module wrapper that:
        - wraps a single Mamba layer
        - exposes a forward_with_trace() API for RL / analysis

    Typical use:
        m_core = Mamba(...)
        m_wrap = IntrospectMambaWrapper(m_core)

        y, trace = m_wrap.forward_with_trace(x)
    """

    def __init__(self, core: Mamba):
        super().__init__()
        self.core = core

    def forward(self, x: Tensor) -> Tensor:
        return self.core(x)

    @torch.no_grad()
    def forward_with_trace(
        self,
        x: Tensor,
        *,
        clone_states: bool = True,
        detach: bool = True,
    ) -> tuple[Tensor, MambaStateTrace]:
        return trace_mamba_sequence(
            self.core,
            x,
            clone_states=clone_states,
            detach=detach,
        )
