"""
RL-ready Mamba policy with internal state exposure.

Plugs Mamba into recurrent RL (PPO/A2C) while:
    - keeping everything on-device
    - exposing internal state for logging / gating analysis.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from mamba_ssm.modules.mamba_simple import Mamba
from pseudo_mamba_introspect import IntrospectMambaWrapper, MambaStateTrace


@dataclass
class MambaPolicyState:
    conv_state: Tensor
    ssm_state: Tensor


class MambaActorCritic(nn.Module):
    """
    Minimal Actor-Critic on top of a single Mamba layer.

    Obs: [B, obs_dim]
    Action: discrete (action_dim)

    Use in RL:
        - Maintain MambaPolicyState across timesteps/rollouts.
        - Log or inspect conv_state / ssm_state as needed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.in_proj = nn.Linear(obs_dim, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_wrap = IntrospectMambaWrapper(self.mamba)
        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Linear(d_model, 1)

    def init_state(self, batch_size: int, max_seqlen: int = 1) -> MambaPolicyState:
        # allocate_inference_cache might return tuple or object
        cache = self.mamba.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
        )
        if isinstance(cache, tuple):
            conv_state, ssm_state = cache
        else:
            conv_state, ssm_state = cache.conv_state, cache.ssm_state
            
        return MambaPolicyState(conv_state=conv_state, ssm_state=ssm_state)

    def forward_step(
        self,
        obs: Tensor,                   # [B, obs_dim]
        state: MambaPolicyState,
    ) -> tuple[Tensor, Tensor, MambaPolicyState, Dict[str, Tensor]]:
        """
        One RL step.

        Returns:
            logits: [B, action_dim]
            value : [B, 1]
            new_state: updated MambaPolicyState
            info: dict with optional internal traces
        """
        B = obs.size(0)
        x = self.in_proj(obs).unsqueeze(1)  # [B, 1, D]

        y, conv_state, ssm_state = self.mamba.step(x, state.conv_state, state.ssm_state)

        logits = self.policy_head(y)        # [B, action_dim]
        value = self.value_head(y)          # [B, 1]

        new_state = MambaPolicyState(conv_state=conv_state, ssm_state=ssm_state)

        info = {
            "conv_state": conv_state,
            "ssm_state": ssm_state,
        }
        return logits, value, new_state, info
