import torch
from typing import Tuple, Dict, Any
from .base import VectorizedEnv

class AssocRecallEnv(VectorizedEnv):
    """
    Associative Recall Task:
    1. Present K pairs of (Key, Value).
    2. Delay H steps.
    3. Present a Key, Agent must output the associated Value.
    """
    def __init__(self, batch_size: int, device: torch.device, 
                 vocab_size: int = 8, num_pairs: int = 4, delay_len: int = 10):
        super().__init__(batch_size, device)
        self.vocab_size = vocab_size
        self.num_pairs = num_pairs
        self.delay_len = delay_len
        # Sequence: (Key, Value) * K + Delay + Query (Key)
        self.total_len = (2 * num_pairs) + delay_len + 1
        
        # Obs: [Item (V), Key_Flag (1), Value_Flag (1), Query_Flag (1)]
        self.obs_dim = vocab_size + 3
        self.act_dim = vocab_size
        
        self.time_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Store pairs: [B, K, 2] (Key, Value)
        self.pairs = torch.zeros(batch_size, num_pairs, 2, dtype=torch.long, device=device)
        self.query_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        self.time_step.zero_()
        # Generate unique keys to avoid ambiguity
        # For simplicity, just random keys/values. 
        # Ideally keys should be unique per sequence.
        # Let's generate random pairs.
        self.pairs = torch.randint(0, self.vocab_size, (self.batch_size, self.num_pairs, 2), device=self.device)
        # Pick a query index
        self.query_idx = torch.randint(0, self.num_pairs, (self.batch_size,), device=self.device)
        return self._get_obs()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Check reward at query step (last step)
        is_query_step = (self.time_step == self.total_len - 1)
        
        if is_query_step.any():
            # Target value
            # pairs: [B, K, 2]
            # query_idx: [B]
            # We want pairs[b, query_idx[b], 1] (Value)
            
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_query_step]
            q_indices = self.query_idx[is_query_step]
            targets = self.pairs[batch_indices, q_indices, 1]
            
            actions = action[is_query_step]
            correct = (actions == targets)
            reward[is_query_step] = correct.float()
            done[is_query_step] = True
            
        self.time_step += 1
        
        if done.any():
            self.time_step[done] = 0
            self.pairs[done] = torch.randint(0, self.vocab_size, (done.sum(), self.num_pairs, 2), device=self.device)
            self.query_idx[done] = torch.randint(0, self.num_pairs, (done.sum(),), device=self.device)
            
        obs = self._get_obs()
        return obs, reward, done, {}

    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        
        # Presentation Phase: t < 2*K
        presentation_len = 2 * self.num_pairs
        is_pres = (self.time_step < presentation_len)
        
        if is_pres.any():
            # Which pair? t // 2
            # Key or Value? t % 2
            ts = self.time_step[is_pres]
            pair_idx = ts // 2
            is_val = (ts % 2 == 1)
            
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_pres]
            
            # Get item
            # pairs[b, pair_idx, 0 or 1]
            # We can gather
            # But simpler:
            # We iterate? No, vectorized.
            
            # items = self.pairs[batch_indices, pair_idx, is_val.long()]
            # This indexing is tricky.
            # Let's use advanced indexing.
            items = self.pairs[batch_indices, pair_idx, is_val.long()]
            
            # Set item
            obs[batch_indices, items] = 1.0
            
            # Set flags
            # Key flag (-3) if not is_val
            # Value flag (-2) if is_val
            
            # obs[batch_indices, -3] where ~is_val
            # obs[batch_indices, -2] where is_val
            
            # We need to sub-index batch_indices
            mask_val = is_val
            mask_key = ~is_val
            
            if mask_key.any():
                obs[batch_indices[mask_key], -3] = 1.0
            if mask_val.any():
                obs[batch_indices[mask_val], -2] = 1.0
                
        # Query Phase: t == total_len - 1
        is_query = (self.time_step == self.total_len - 1)
        if is_query.any():
            # Present Key
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_query]
            q_indices = self.query_idx[is_query]
            keys = self.pairs[batch_indices, q_indices, 0]
            
            obs[batch_indices, keys] = 1.0
            obs[batch_indices, -1] = 1.0 # Query Flag
            
        return obs
