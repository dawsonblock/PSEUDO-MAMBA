import torch
import torch.nn as nn
from typing import Tuple, Any
from pseudo_mamba.controllers.base import BaseController

class ActorCritic(nn.Module):
    """
    Wraps a BaseController with Policy (Actor) and Value (Critic) heads.
    """
    def __init__(self, controller: BaseController, act_dim: int):
        super().__init__()
        self.controller = controller
        self.act_dim = act_dim
        
        feature_dim = controller.feature_dim
        
        # Actor Head: Features -> Logits
        self.actor = nn.Linear(feature_dim, act_dim)
        
        # Critic Head: Features -> Value
        self.critic = nn.Linear(feature_dim, 1)

    def init_state(self, batch_size: int, device: torch.device) -> Any:
        return self.controller.init_state(batch_size, device)

    def forward_step(self, x_t: torch.Tensor, state: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Process one step.
        Returns:
            logits: [B, act_dim]
            value: [B, 1]
            new_state: Updated state
        """
        features, new_state = self.controller.forward_step(x_t, state)
        
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value, new_state

    def reset_mask(self, state: Any, done_mask: torch.Tensor) -> Any:
        return self.controller.reset_mask(state, done_mask)
