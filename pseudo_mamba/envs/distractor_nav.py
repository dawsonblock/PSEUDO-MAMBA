import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class DistractorNavEnv(VectorizedEnv):
    """
    Distractor Navigation Task:
    1. 1D Line environment [-10, 10].
    2. t=0: Target location is shown (e.g., +5).
    3. t>0: Target is hidden. Distractors (fake targets) may appear.
    4. Agent starts at 0. Must navigate to target and stay there.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 horizon: int = 20, map_size: int = 10):
        super().__init__(batch_size, device)
        self.horizon = horizon
        self.map_size = map_size # +/- map_size
        
        # Obs: [Current_Pos (1), Target_Signal (1), Distractor_Signal (1)]
        self.obs_dim = 3
        self.act_dim = 3 # -1, 0, +1
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.pos = torch.zeros(batch_size, dtype=torch.float, device=device)
        self.target = torch.zeros(batch_size, dtype=torch.float, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.pos.zero_()
        # Random target in [-map_size, map_size]
        self.target = (torch.rand(self.batch_size, device=self.device) * 2 * self.map_size) - self.map_size
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # action: 0 (-1), 1 (0), 2 (+1)
        move = action.float() - 1.0
        self.pos += move
        self.pos = torch.clamp(self.pos, -self.map_size, self.map_size)
        
        # Reward: 1.0 if close to target, else 0.0
        dist = torch.abs(self.pos - self.target)
        reward = (dist < 1.0).float()
        
        # Done
        is_done = (self.time_step == self.horizon - 1)
        done = is_done
        
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.pos[done] = 0
            self.target[done] = (torch.rand(done.sum(), device=self.device) * 2 * self.map_size) - self.map_size
            
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # 0: Pos (normalized)
        obs[:, 0] = self.pos / self.map_size
        
        # 1: Target Signal (only at t=0)
        is_start = (self.time_step == 0)
        if is_start.any():
            obs[is_start, 1] = self.target[is_start] / self.map_size
            
        # 2: Distractor Signal (random noise at t > 0)
        is_later = (self.time_step > 0)
        if is_later.any():
            # Random signal
            noise = (torch.rand(is_later.sum(), device=self.device) * 2 - 1)
            obs[is_later, 2] = noise
            
        return obs
