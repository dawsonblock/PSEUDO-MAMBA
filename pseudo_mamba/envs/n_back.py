import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class NBackEnv(VectorizedEnv):
    """
    N-Back Task:
    Continuous stream of tokens.
    At each step t, agent must output the token from t-N.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 vocab_size: int = 8, n_back: int = 2, seq_len: int = 50):
        super().__init__(batch_size, device)
        self.vocab_size = vocab_size
        self.n_back = n_back
        self.seq_len = seq_len
        
        # Obs: [Token (V)]
        self.obs_dim = vocab_size
        self.act_dim = vocab_size
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        # History buffer: [B, L]
        # We generate the full sequence upfront for simplicity
        self.sequence = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.sequence = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Check reward
        # Target is sequence[t - N]
        # Valid only if t >= N
        
        valid_mask = (self.time_step >= self.n_back)
        
        if valid_mask.any():
            idx = self.time_step[valid_mask] - self.n_back
            batch_indices = torch.arange(self.batch_size, device=self.device)[valid_mask]
            targets = self.sequence[batch_indices, idx]
            
            actions = action[valid_mask]
            correct = (actions == targets)
            reward[valid_mask] = correct.float()
            
        # Check done
        is_done = (self.time_step == self.seq_len - 1)
        done[is_done] = True
        
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.sequence[done] = torch.randint(0, self.vocab_size, (done.sum(), self.seq_len), device=self.device)
            
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # Present current token sequence[t]
        # If t < seq_len
        
        valid = (self.time_step < self.seq_len)
        if valid.any():
            idx = self.time_step[valid]
            batch_indices = torch.arange(self.batch_size, device=self.device)[valid]
            tokens = self.sequence[batch_indices, idx]
            
            obs[batch_indices, tokens] = 1.0
            
        return obs
