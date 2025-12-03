import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class MultiCueDelayEnv(VectorizedEnv):
    """
    Multi-Cue Delay Task:
    1. Presentation Phase (Length L_p): N cues appear at random times.
    2. Delay Phase (Length L_d).
    3. Query Phase (Length N): Agent must recall cues in order of appearance (or fixed order).
       Let's assume order of appearance for simplicity, or just N slots.
       Actually, standard is usually "recall cue i".
       Let's do: Recall all N cues in order.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 vocab_size: int = 8, num_cues: int = 3, presentation_len: int = 10, delay_len: int = 10):
        super().__init__(batch_size, device)
        self.vocab_size = vocab_size
        self.num_cues = num_cues
        self.presentation_len = presentation_len
        self.delay_len = delay_len
        self.total_len = presentation_len + delay_len + num_cues
        
        # Obs: [Cue (V), Query_Flag (1)]
        self.obs_dim = vocab_size + 1
        self.act_dim = vocab_size
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Cues: [B, N]
        self.cues = torch.zeros(batch_size, num_cues, dtype=torch.long, device=device)
        # Cue Times: [B, N] - when each cue appears
        self.cue_times = torch.zeros(batch_size, num_cues, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        self.cues = torch.randint(0, self.vocab_size, (self.batch_size, self.num_cues), device=self.device)
        
        # Generate random unique times for cues within presentation_len
        # We can use randperm
        # This is tricky to vectorize perfectly without a loop or sort.
        # [B, L_p] rand -> topk N indices
        
        rand_vals = torch.rand(self.batch_size, self.presentation_len, device=self.device)
        _, indices = torch.topk(rand_vals, self.num_cues, dim=1)
        self.cue_times, _ = torch.sort(indices, dim=1)
        
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Query Phase
        query_start = self.presentation_len + self.delay_len
        is_query = (self.time_step >= query_start) & (self.time_step < query_start + self.num_cues)
        
        if is_query.any():
            idx = self.time_step[is_query] - query_start
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_query]
            targets = self.cues[batch_indices, idx]
            
            actions = action[is_query]
            correct = (actions == targets)
            reward[is_query] = correct.float()
            
        # Check done
        is_done = (self.time_step == self.total_len - 1)
        done[is_done] = True
        
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.cues[done] = torch.randint(0, self.vocab_size, (done.sum(), self.num_cues), device=self.device)
            # Re-gen times
            rand_vals = torch.rand(done.sum(), self.presentation_len, device=self.device)
            _, indices = torch.topk(rand_vals, self.num_cues, dim=1)
            times, _ = torch.sort(indices, dim=1)
            self.cue_times[done] = times
            
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # Presentation
        is_pres = (self.time_step < self.presentation_len)
        if is_pres.any():
            # Check if current time matches any cue time
            # cue_times: [B, N]
            # time_step: [B] -> [B, 1]
            
            t_expanded = self.time_step.unsqueeze(1) # [B, 1]
            matches = (self.cue_times == t_expanded) # [B, N]
            
            # We assume unique times, so at most one match per batch
            has_match = matches.any(dim=1) # [B]
            
            if has_match.any():
                # Get the cue index (0..N-1)
                # We can use max(dim=1) to get index of True
                _, cue_indices = matches.max(dim=1) # [B]
                
                # Get the cue value
                # cues: [B, N]
                # cue_indices: [B]
                # We need to gather
                
                batch_indices = torch.arange(self.batch_size, device=self.device)
                
                # Only for those with match
                active_batch = batch_indices[has_match]
                active_cue_indices = cue_indices[has_match]
                
                active_cues = self.cues[active_batch, active_cue_indices]
                
                obs[active_batch, active_cues] = 1.0
                
        # Query
        query_start = self.presentation_len + self.delay_len
        is_query = (self.time_step >= query_start)
        if is_query.any():
            obs[is_query, -1] = 1.0
            
        return obs
