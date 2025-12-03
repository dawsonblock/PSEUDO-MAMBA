import torch
import torch.nn as nn
from typing import Tuple, Any
from .base import BaseController

# Try to import CUDA extension
try:
    import pseudo_mamba_ext
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

class PseudoMambaController(BaseController):
    """
    Controller using the 'Pseudo-Mamba' recurrence: y = tanh(x + h).
    Uses CUDA kernel if available, otherwise PyTorch fallback.
    """
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__(input_dim, hidden_dim, feature_dim)
        
        self.encoder = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        if feature_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # State: [B, hidden_dim]
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_step(self, x_t: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t: [B, input_dim]
        # state: [B, hidden_dim]
        
        x_emb = self.encoder(x_t) # [B, hidden_dim]
        
        if HAS_CUDA_EXT and x_emb.is_cuda:
            # Use CUDA extension
            output = pseudo_mamba_ext.forward(x_emb, state)
            new_state = output 
        else:
            # PyTorch Fallback
            new_state = torch.tanh(x_emb + state)
            output = new_state
            
        features = self.proj(output)
        return features, new_state

    def reset_mask(self, state: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        mask = (1.0 - done_mask.float()).unsqueeze(1)
        return state * mask
