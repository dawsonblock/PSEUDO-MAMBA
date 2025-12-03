import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

class BaseController(nn.Module, ABC):
    """
    Abstract base class for all memory controllers (GRU, Mamba, Pseudo-Mamba).
    Enforces a unified interface for state management and forward passes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> Any:
        """
        Initialize the recurrent state for a new episode.
        Returns a state object (Tensor, tuple, or dataclass) compatible with forward_step.
        """
        pass

    @abstractmethod
    def forward_step(self, x_t: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        """
        Process a single timestep.
        Args:
            x_t: Input tensor [B, input_dim]
            state: Current recurrent state
        Returns:
            features: [B, feature_dim] (to be fed to actor/critic heads)
            new_state: Updated recurrent state
        """
        pass

    @abstractmethod
    def reset_mask(self, state: Any, done_mask: torch.Tensor) -> Any:
        """
        Reset state for environments that have finished.
        Args:
            state: Current recurrent state
            done_mask: Boolean tensor [B], True where env is done
        Returns:
            masked_state: State with finished envs reset to initial values
        """
        pass

    @property
    def state_size(self) -> int:
        """Return the size of the hidden state (for logging/analysis)."""
        return self.hidden_dim
