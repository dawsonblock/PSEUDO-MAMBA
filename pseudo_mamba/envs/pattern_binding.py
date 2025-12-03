import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class PatternBindingEnv(VectorizedEnv):
    """
    Pattern Binding Task:
    1. Present Pattern A (Sequence of L tokens).
    2. Present Pattern B (Sequence of L tokens).
    3. Delay.
    4. Present Pattern A.
    5. Agent must output Pattern B.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 vocab_size: int = 8, pattern_len: int = 3, delay_len: int = 10):
        super().__init__(batch_size, device)
        self.vocab_size = vocab_size
        self.pattern_len = pattern_len
        self.delay_len = delay_len
        # Seq: A (L) + B (L) + Delay (H) + Query A (L) + Response B (L)
        # Actually usually: Present A, Present B. Delay. Query A -> Expect B.
        # Total len: L + L + H + L + L
        self.total_len = 4 * pattern_len + delay_len
        
        # Obs: [Token (V), Phase_Flag (2?)]
        # Let's just use Token + 1 flag for "Output Phase"
        self.obs_dim = vocab_size + 1
        self.act_dim = vocab_size
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.pattern_a = torch.zeros(batch_size, pattern_len, dtype=torch.long, device=device)
        self.pattern_b = torch.zeros(batch_size, pattern_len, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.pattern_a = torch.randint(0, self.vocab_size, (self.batch_size, self.pattern_len), device=self.device)
        self.pattern_b = torch.randint(0, self.vocab_size, (self.batch_size, self.pattern_len), device=self.device)
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Response Phase: Last L steps
        resp_start = 3 * self.pattern_len + self.delay_len
        is_resp = (self.time_step >= resp_start)
        
        if is_resp.any():
            idx = self.time_step[is_resp] - resp_start
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_resp]
            targets = self.pattern_b[batch_indices, idx]
            
            actions = action[is_resp]
            correct = (actions == targets)
            reward[is_resp] = correct.float()
            
        is_done = (self.time_step == self.total_len - 1)
        done[is_done] = True
        
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.pattern_a[done] = torch.randint(0, self.vocab_size, (done.sum(), self.pattern_len), device=self.device)
            self.pattern_b[done] = torch.randint(0, self.vocab_size, (done.sum(), self.pattern_len), device=self.device)
            
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # Phase 1: Present A (0 to L-1)
        is_p1 = (self.time_step < self.pattern_len)
        if is_p1.any():
            idx = self.time_step[is_p1]
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_p1]
            tokens = self.pattern_a[batch_indices, idx]
            obs[batch_indices, tokens] = 1.0
            
        # Phase 2: Present B (L to 2L-1)
        is_p2 = (self.time_step >= self.pattern_len) & (self.time_step < 2 * self.pattern_len)
        if is_p2.any():
            idx = self.time_step[is_p2] - self.pattern_len
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_p2]
            tokens = self.pattern_b[batch_indices, idx]
            obs[batch_indices, tokens] = 1.0
            
        # Phase 3: Delay (2L to 2L+H-1) -> Zero input
        
        # Phase 4: Query A (2L+H to 3L+H-1)
        q_start = 2 * self.pattern_len + self.delay_len
        is_q = (self.time_step >= q_start) & (self.time_step < q_start + self.pattern_len)
        if is_q.any():
            idx = self.time_step[is_q] - q_start
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_q]
            tokens = self.pattern_a[batch_indices, idx]
            obs[batch_indices, tokens] = 1.0
            obs[batch_indices, -1] = 1.0 # Query Flag
            
        # Phase 5: Response (3L+H to 4L+H-1) -> Zero input, expect output
        # Usually we keep the query flag on? Or off?
        # Let's keep query flag on to indicate "Output Mode"
        resp_start = 3 * self.pattern_len + self.delay_len
        is_resp = (self.time_step >= resp_start)
        if is_resp.any():
            obs[is_resp, -1] = 1.0
            
        return obs
