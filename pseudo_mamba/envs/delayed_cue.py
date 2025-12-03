import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class DelayedCueEnv(VectorizedEnv):
    """
    Delayed Cue Task:
    1. t=0: Present one of N cues (one-hot).
    2. t=1..H: Zero input (delay).
    3. t=H+1: Query signal. Agent must output the cue index.
    """
    def __init__(self, batch_size: int, device: torch.device, num_cues: int = 8, delay_len: int = 10):
        super().__init__(batch_size, device)
        self.num_cues = num_cues
        self.delay_len = delay_len
        self.total_len = delay_len + 2 # Cue + Delay + Query
        
        # Obs: [Cue (N), Query_Flag (1)]
        self.obs_dim = num_cues + 1
        self.act_dim = num_cues
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.cues = torch.zeros(batch_size, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.cues = torch.randint(0, self.num_cues, (self.batch_size,), device=self.device)
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # action: [B] (indices)
        
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Check reward at query step
        is_query = (self.time_step == self.delay_len + 1)
        
        if is_query.any():
            correct = (action == self.cues)
            reward[is_query] = correct[is_query].float()
            done[is_query] = True
            
        # Increment time
        self.time_step += 1
        
        # Auto-reset done envs (vectorized auto-reset logic usually handled by runner, 
        # but here we can just wrap around or let runner handle it. 
        # For PPO, we usually return done=True and let the runner reset.
        # But to keep internal state consistent, we can reset internal counters for done envs.)
        if done.any():
            self.time_step[done] = 0
            self.cues[done] = torch.randint(0, self.num_cues, (done.sum(),), device=self.device)
            
        obs = self._get_obs()
        
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # t=0: Present Cue
        is_start = (self.time_step == 0)
        if is_start.any():
            # One-hot cue
            # obs[is_start, cue] = 1
            # We need to scatter
            cues_start = self.cues[is_start]
            obs[is_start, cues_start] = 1.0
            
        # t=H+1: Query Flag
        is_query = (self.time_step == self.delay_len + 1)
        if is_query.any():
            obs[is_query, -1] = 1.0
            
        return obs
