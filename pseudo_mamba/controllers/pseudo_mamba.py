import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseController

# Try to import CUDA extension
try:
    import pseudo_mamba_ext
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


class PseudoMambaController(BaseController):
    """
    Controller using the 'Pseudo-Mamba' recurrence:

        y = tanh(x + h)

    If the CUDA extension is available, we call:
        pseudo_mamba_ext.pseudo_mamba_forward(x_emb, state)
    Otherwise we fall back to pure PyTorch.

    Interface:
        - init_state(batch_size, device) -> [B, hidden_dim]
        - forward_step(obs, state) -> (features, new_state)
    """
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__(input_dim, hidden_dim, feature_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        
        if feature_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_step(
        self,
        obs: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs:   [B, obs_dim]
        state: [B, hidden_dim]

        Returns:
            features: [B, feature_dim]  (projected output)
            new_state: [B, hidden_dim] (recurrent state)
        """
        x_emb = self.in_proj(obs)  # [B, hidden_dim]

        if HAS_CUDA_EXT:
            # Use CUDA extension (x, h) -> y
            new_state = pseudo_mamba_ext.pseudo_mamba_forward(x_emb, state)
        else:
            # Pure PyTorch fallback
            new_state = torch.tanh(x_emb + state)

        features = self.proj(new_state)
        return features, new_state

    def reset_mask(self, state: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        """
        Zero state where episodes are done.
        done_mask: [B] with 1.0 where done, 0.0 otherwise.
        """
        mask = (1.0 - done_mask.float()).unsqueeze(1)
        return state * mask
