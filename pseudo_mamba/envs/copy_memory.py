import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class CopyMemoryEnv(VectorizedEnv):
    """
    Copy Memory Task:
    1. Present sequence of K tokens.
    2. Delay H steps.
    3. Agent must reproduce the sequence.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 vocab_size: int = 8, seq_len: int = 10, delay_len: int = 10):
        super().__init__(batch_size, device)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.delay_len = delay_len
        self.total_len = seq_len + delay_len + seq_len
        
        # Obs: [Tokens (V), Write_Flag (1), Read_Flag (1)]
        self.obs_dim = vocab_size + 2
        self.act_dim = vocab_size
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.sequences = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.sequences = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Check reward during read phase
        read_start = self.seq_len + self.delay_len
        read_end = read_start + self.seq_len
        
        is_read = (self.time_step >= read_start) & (self.time_step < read_end)
        
        if is_read.any():
            # Which token index?
            idx = self.time_step[is_read] - read_start
            # Get target tokens
            # We need to gather from self.sequences[batch_idx, idx]
            # self.sequences is [B, K]
            # idx is [B_active]
            
            # We need batch indices for the active envs
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_read]
            targets = self.sequences[batch_indices, idx]
            
            actions = action[is_read]
            correct = (actions == targets)
            reward[is_read] = correct.float()
            
        # Check done
        is_done = (self.time_step == self.total_len - 1)
        done[is_done] = True
        
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.sequences[done] = torch.randint(0, self.vocab_size, (done.sum(), self.seq_len), device=self.device)
            
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # Write Phase: t < K
        is_write = (self.time_step < self.seq_len)
        if is_write.any():
            # Write Flag
            obs[is_write, -2] = 1.0
            
            # Token
            idx = self.time_step[is_write]
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_write]
            tokens = self.sequences[batch_indices, idx]
            
            # One-hot token
            # obs[batch, token] = 1
            # We need to scatter
            # Create index tensor for scatter
            # We want to set obs[b, token] = 1
            # We can use scatter_
            
            # obs[is_write] is [B_active, obs_dim]
            # tokens is [B_active]
            
            # We can't easily index into obs[is_write] with scatter directly if we don't have a view.
            # But we can do:
            # obs[batch_indices, tokens] = 1.0
            obs[batch_indices, tokens] = 1.0
            
        # Read Phase: t >= K + H
        read_start = self.seq_len + self.delay_len
        is_read = (self.time_step >= read_start) & (self.time_step < read_start + self.seq_len)
        if is_read.any():
            # Read Flag
            obs[is_read, -1] = 1.0
            
        return obs
