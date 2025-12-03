import torch
import torch.nn as nn
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not found. Real Mamba controller will not work.")

class PseudoMambaRLCore(nn.Module):
    """
    RL-facing wrapper around MambaSimple.

    - Input: obs_t [B, obs_dim]
    - Hidden state: (conv_state, ssm_state)
    - Output: core_out_t [B, d_model], new_state
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        bias: bool = True,
    ):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required for PseudoMambaRLCore")
            
        self.obs_dim = obs_dim
        self.d_model = d_model

        self.obs_proj = nn.Linear(obs_dim, d_model, bias=bias)
        self.core = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=bias,
        )

    def init_state(self, batch_size: int, device: torch.device):
        """
        Returns:
            conv_state: [B, d_inner, d_conv]
            ssm_state:  [B, d_inner, d_state]
        """
        # allocate_inference_cache returns a MambaCache object or tuple depending on version
        # We use the method we verified in mamba_simple.py
        cache = self.core.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=1,   # step-wise; we only ever do 1 token at a time
            dtype=self.obs_proj.weight.dtype,
        )
        
        # Handle both tuple return and object return if implementation varies
        if isinstance(cache, tuple):
            conv_state, ssm_state = cache
        else:
            # Assuming our patched version returns tuple or has attributes
            # If it's the standard Mamba implementation it might be different, 
            # but our patch in mamba_simple.py returns (conv_state, ssm_state)
            conv_state, ssm_state = cache

        return conv_state.to(device), ssm_state.to(device)

    def forward_step(self, obs_t: torch.Tensor, state):
        """
        obs_t:  [B, obs_dim]
        state:  (conv_state, ssm_state)
        Returns:
            out_t: [B, d_model]
            new_state: (conv_state, ssm_state)
        """
        conv_state, ssm_state = state
        x = self.obs_proj(obs_t).unsqueeze(1)  # (B, 1, d_model)
        
        # Mamba.step signature: (hidden_states, conv_state, ssm_state)
        out, conv_state, ssm_state = self.core.step(x, conv_state, ssm_state)
        
        return out.squeeze(1), (conv_state, ssm_state)
