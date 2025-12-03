import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class PermutedCopyEnv(VectorizedEnv):
    """
    Permuted Copy Task:
    1. Present sequence of K tokens.
    2. Delay H steps.
    3. Agent must reproduce the sequence in a PERMUTED order.
       The permutation is fixed per episode (generated at reset).
       We provide the target index as observation during read phase?
       Or just "output token at index p[t]".
       Usually, explicit target index is not given, but the rule is fixed (e.g. reverse).
       If random permutation, we must provide the target index or some cue.
       Let's assume Random Permutation, and we provide the "Read Index" as input during read phase.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 vocab_size: int = 8, seq_len: int = 10, delay_len: int = 10):
        super().__init__(batch_size, device)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.delay_len = delay_len
        self.total_len = seq_len + delay_len + seq_len
        
        # Obs: [Tokens (V), Write_Flag (1), Read_Flag (1), Read_Index (K - one hot? or scalar?)]
        # Providing Read Index as one-hot is cleaner for neural nets.
        self.obs_dim = vocab_size + 2 + seq_len
        self.act_dim = vocab_size
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.sequences = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        self.permutations = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.sequences = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        # Generate random permutations
        for b in range(self.batch_size):
            self.permutations[b] = torch.randperm(self.seq_len, device=self.device)
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        read_start = self.seq_len + self.delay_len
        read_end = read_start + self.seq_len
        is_read = (self.time_step >= read_start) & (self.time_step < read_end)
        
        if is_read.any():
            step_idx = self.time_step[is_read] - read_start
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_read]
            
            # Which index to read?
            # permutations[b, step_idx]
            target_indices = self.permutations[batch_indices, step_idx]
            
            targets = self.sequences[batch_indices, target_indices]
            
            actions = action[is_read]
            correct = (actions == targets)
            reward[is_read] = correct.float()
            
        is_done = (self.time_step == self.total_len - 1)
        done[is_done] = True
        
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.sequences[done] = torch.randint(0, self.vocab_size, (done.sum(), self.seq_len), device=self.device)
            for i, b_idx in enumerate(torch.where(done)[0]):
                self.permutations[b_idx] = torch.randperm(self.seq_len, device=self.device)
                
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # Write
        is_write = (self.time_step < self.seq_len)
        if is_write.any():
            obs[is_write, -self.seq_len - 2] = 1.0 # Write Flag
            
            idx = self.time_step[is_write]
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_write]
            tokens = self.sequences[batch_indices, idx]
            obs[batch_indices, tokens] = 1.0
            
        # Read
        read_start = self.seq_len + self.delay_len
        is_read = (self.time_step >= read_start) & (self.time_step < read_start + self.seq_len)
        if is_read.any():
            obs[is_read, -self.seq_len - 1] = 1.0 # Read Flag
            
            # Provide Read Index
            step_idx = self.time_step[is_read] - read_start
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_read]
            
            # We want to tell the agent: "Output the token at index X"
            # X = permutations[b, step_idx]
            target_indices = self.permutations[batch_indices, step_idx]
            
            # One-hot encode target index at the end of obs
            # obs is [V + 2 + K]
            # Last K dims are for index
            # obs[..., -K:]
            
            # Offset = V + 2
            offset = self.vocab_size + 2
            obs[batch_indices, offset + target_indices] = 1.0
            
        return obs
