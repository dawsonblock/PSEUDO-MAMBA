import torch
import torch.nn as nn
from typing import Tuple, Any
from .base import BaseController

class GRUController(BaseController):
    """
    Standard GRU-based controller.
    State is a single tensor [B, hidden_dim].
    """
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__(input_dim, hidden_dim, feature_dim)
        # If feature_dim != hidden_dim, we need a projection.
        # But usually we just use hidden state as features.
        if feature_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()
            
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # GRU state: [B, hidden_dim]
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_step(self, x_t: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t: [B, input_dim]
        # state: [B, hidden_dim]
        new_state = self.gru(x_t, state)
        features = self.proj(new_state)
        return features, new_state

    def reset_mask(self, state: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        # state: [B, hidden_dim]
        # done_mask: [B] float or boolean
        mask = (1.0 - done_mask.float()).unsqueeze(1) # [B, 1]
        return state * mask
