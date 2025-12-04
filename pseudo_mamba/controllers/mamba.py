import torch
import torch.nn as nn
from typing import Tuple, Any, Optional
from .base import BaseController

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class MambaController(BaseController):
    """
    Mamba-based controller using the official mamba_ssm library
    with explicit state management patches.
    """
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, layer_idx: int = 0, 
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, require_patched_mamba: bool = False):
        super().__init__(input_dim, hidden_dim, feature_dim)
        
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is not installed or not working (e.g. missing triton). Cannot use MambaController.")
        
        # Projections if dims don't match
        self.encoder = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Mamba Block
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx
        )
        
        # Runtime check for patched Mamba
        if require_patched_mamba:
            if not hasattr(self.mamba, "get_inference_state"):
                raise ImportError(
                    "Your installed mamba_ssm does not support 'get_inference_state'. "
                    "This feature requires the patched version of Mamba described in "
                    "STATE_MANAGEMENT_README.md and ENHANCED_PATCHES.md. "
                    "Please patch your local mamba-ssm installation or set require_patched_mamba=False."
                )
        
        if feature_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (conv_state, ssm_state) initialized to zeros.
        """
        try:
            cache = self.mamba.allocate_inference_cache(batch_size, max_seqlen=1)
            if isinstance(cache, tuple):
                return cache
            return cache.conv_state, cache.ssm_state
        except AttributeError:
            d_inner = self.mamba.d_inner
            d_conv = self.mamba.d_conv
            d_state = self.mamba.d_state
            
            conv_state = torch.zeros(batch_size, d_inner, d_conv, device=device)
            ssm_state = torch.zeros(batch_size, d_inner, d_state, device=device)
            return conv_state, ssm_state

    def forward_step(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x_t: [B, input_dim]
        # state: (conv_state, ssm_state)
        
        # Encode
        x_emb = self.encoder(x_t) # [B, hidden_dim]
        
        # Mamba step expects [B, 1, D]
        x_seq = x_emb.unsqueeze(1)
        
        conv_state, ssm_state = state
        
        # Step
        y_seq, new_conv, new_ssm = self.mamba.step(x_seq, conv_state, ssm_state)
        
        # y_seq: [B, 1, D] -> [B, D]
        y_t = y_seq.squeeze(1)
        
        features = self.proj(y_t)
        
        return features, (new_conv, new_ssm)

    def reset_mask(self, state: Tuple[torch.Tensor, torch.Tensor], done_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_state, ssm_state = state
        mask = (1.0 - done_mask.float()) # [B]
        
        mask_conv = mask.view(-1, 1, 1)
        mask_ssm = mask.view(-1, 1, 1)
        
        return conv_state * mask_conv, ssm_state * mask_ssm
