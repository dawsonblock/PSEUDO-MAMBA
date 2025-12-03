import torch
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class VectorizedEnv(ABC):
    """
    Abstract base class for vectorized memory environments.
    All environments must support batch processing on GPU.
    """
    def __init__(self, batch_size: int, device: torch.device):
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset all environments in the batch.
        Returns:
            observation: [B, obs_dim]
        """
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Step the environment.
        Args:
            action: [B] or [B, act_dim]
        Returns:
            observation: [B, obs_dim]
            reward: [B]
            done: [B] (boolean)
            info: Dict
        """
        pass

    # Properties that should be set by subclasses
    obs_dim: int
    act_dim: int
