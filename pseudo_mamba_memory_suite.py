#!/usr/bin/env python3
"""
pseudo_mamba_memory_suite.py

Mr Block's Ultimate AI Memory Benchmark v1.

Eight long-horizon memory tasks, one unified training script, and a controller
selector that can run:
    - GRU baseline
    - Mamba-style SSM stub (pure PyTorch)
    - Pseudo-Mamba CUDA extension (plug your kernels here)

Envs (all vectorized, symbolic, long-horizon):
    1. delayed_cue          – classic one-cue delayed reward
    2. copy_memory          – copy a sequence after long delay
    3. assoc_recall         – key→value pairs, query after distractors
    4. n_back               – N-back match / mismatch
    5. multi_cue_delayed    – multiple cues, single queried
    6. permuted_copy        – copy under random permutation
    7. pattern_binding      – bind two symbols and recall
    8. distractor_nav       – pseudo-navigation with irrelevant distractors

This is written to be:
    - Drop-in runnable on CPU/GPU (GRU / stub SSM).
    - Ready for Pseudo-Mamba integration via a single adapter class.
    - Easy to extend with new tasks / controllers.

Usage (example):

    python pseudo_mamba_memory_suite.py --env delayed_cue --controller gru
    python pseudo_mamba_memory_suite.py --env copy_memory --controller mamba_stub
    python pseudo_mamba_memory_suite.py --env assoc_recall --controller pseudo_mamba_ext \
        --ssm_d_model 128 --horizon 200 --num_envs 64 --total_updates 20000

When you finish wiring your CUDA/C++ Pseudo-Mamba kernels, replace the
`PseudoMambaExtBlock` forward with the real op and you’re done.
"""

import argparse
import math
import time
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. Vectorized Memory Environments
# ============================================================

class BaseVectorEnv:
    """
    Base class for vectorized memory tasks.
    All envs expose:
        - obs_shape: (obs_dim,)
        - n_actions: int
        - reset(batch_size) -> obs [B, obs_dim]
        - step(action) -> (obs, reward, done, info)
    """

    def __init__(self, batch_size: int, obs_dim: int, n_actions: int, device: torch.device):
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

    def reset(self, batch_size: int = None) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: torch.Tensor):
        raise NotImplementedError


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=num_classes).float()


# ------------- 1) Delayed Cue Task -----------------

class DelayedCueEnv(BaseVectorEnv):
    """
    Classic delayed cue:
      - horizon steps; at t_cue, we emit a one-hot cue (0..K-1)
      - rest of steps: zero or distractors
      - at final step: agent must output correct cue index

    Action space: {0..K-1}
    Reward: +1 if action == cue, else 0, only at final step.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 horizon: int = 40, n_cues: int = 4):
        obs_dim = n_cues
        n_actions = n_cues
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.horizon = horizon
        self.n_cues = n_cues

        # per-env state
        self.t = None
        self.cue = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        self.t = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.cue = torch.randint(0, self.n_cues, (self.batch_size,), device=self.device)
        # random cue time in [0, horizon//4]
        self.cue_t = torch.randint(0, max(1, self.horizon // 4), (self.batch_size,), device=self.device)
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        return obs

    def step(self, action: torch.Tensor):
        # generate obs for current t
        obs = torch.zeros(self.batch_size, self.obs_dim, device=self.device)
        # show cue exactly at cue_t
        cue_mask = (self.t == self.cue_t)
        if cue_mask.any():
            obs[cue_mask] = one_hot(self.cue[cue_mask], self.n_cues)

        # reward only at final step
        done = (self.t == (self.horizon - 1))
        reward = torch.zeros(self.batch_size, device=self.device)
        final_mask = done
        if final_mask.any():
            # action must be in [0..n_cues-1]
            a = action[final_mask].clamp(0, self.n_cues - 1)
            reward[final_mask] = (a == self.cue[final_mask]).float()

        self.t = self.t + 1
        done = done.float()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 2) Copy Memory Task -----------------

class CopyMemoryEnv(BaseVectorEnv):
    """
    Copy Memory:
      - See a sequence of length L of digits (0..K-1) at the start.
      - Then D distractor steps (all zeros).
      - Then must output the original sequence over L steps.

    Obs_dim = K + 2:
      - First K dims: symbol one-hot.
      - Last 2 dims: markers [start_flag, recall_flag].
    """

    def __init__(self, batch_size: int, device: torch.device,
                 seq_len: int = 10, delay: int = 40, n_symbols: int = 8):
        obs_dim = n_symbols + 2
        n_actions = n_symbols
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.seq_len = seq_len
        self.delay = delay
        self.n_symbols = n_symbols
        self.total_len = seq_len + delay + seq_len

        self.t = None
        self.seq = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        self.t = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.seq = torch.randint(0, self.n_symbols, (self.batch_size, self.seq_len), device=self.device)
        return self._obs_from_t()

    def _obs_from_t(self) -> torch.Tensor:
        B = self.batch_size
        obs = torch.zeros(B, self.obs_dim, device=self.device)
        # markers
        start_flag = torch.zeros(B, device=self.device)
        recall_flag = torch.zeros(B, device=self.device)

        # t in [0, seq_len): show input sequence
        mask_in = (self.t < self.seq_len)
        if mask_in.any():
            t_in = self.t[mask_in]
            sym = self.seq[mask_in, t_in]
            obs[mask_in, :self.n_symbols] = one_hot(sym, self.n_symbols)
            start_flag[mask_in] = 1.0

        # t in [seq_len + delay, seq_len + delay + seq_len): recall phase
        mask_rec = (self.t >= self.seq_len + self.delay) & (self.t < self.total_len)
        recall_flag[mask_rec] = 1.0

        obs[:, self.n_symbols] = start_flag
        obs[:, self.n_symbols + 1] = recall_flag
        return obs

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        # recall window
        mask_rec = (self.t >= self.seq_len + self.delay) & (self.t < self.total_len)
        if mask_rec.any():
            # position within output sequence
            k = self.t[mask_rec] - (self.seq_len + self.delay)
            target = self.seq[mask_rec, k]
            a = action[mask_rec].clamp(0, self.n_symbols - 1)
            reward[mask_rec] = (a == target).float()

        self.t = self.t + 1
        done = (self.t >= self.total_len).float()
        obs = self._obs_from_t()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 3) Associative Recall -----------------

class AssocRecallEnv(BaseVectorEnv):
    """
    Associative recall:
      - Present M key→value pairs: (k_i, v_i) with distractors.
      - After delay, show query key k_q.
      - Agent must output correct v_q.

    Action space: values (0..V-1)
    Reward: +1 at query step if correct, else 0.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 num_pairs: int = 4, n_keys: int = 8, n_vals: int = 8,
                 delay: int = 40):
        obs_dim = n_keys + n_vals + 2  # [key one-hot, value one-hot, is_pair, is_query]
        n_actions = n_vals
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.num_pairs = num_pairs
        self.n_keys = n_keys
        self.n_vals = n_vals
        self.delay = delay

        # schedule: pairs over first num_pairs steps, then delay, then one query step, then end.
        self.total_len = num_pairs + delay + 1

        self.t = None
        self.keys = None
        self.vals = None
        self.query_idx = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        B = self.batch_size
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.keys = torch.randint(0, self.n_keys, (B, self.num_pairs), device=self.device)
        self.vals = torch.randint(0, self.n_vals, (B, self.num_pairs), device=self.device)
        self.query_idx = torch.randint(0, self.num_pairs, (B,), device=self.device)
        return self._obs_from_t()

    def _obs_from_t(self) -> torch.Tensor:
        B = self.batch_size
        obs = torch.zeros(B, self.obs_dim, device=self.device)
        is_pair = torch.zeros(B, device=self.device)
        is_query = torch.zeros(B, device=self.device)

        # pair presentation
        mask_pair = (self.t < self.num_pairs)
        if mask_pair.any():
            k = self.keys[mask_pair, self.t[mask_pair]]
            v = self.vals[mask_pair, self.t[mask_pair]]
            obs[mask_pair, :self.n_keys] = one_hot(k, self.n_keys)
            obs[mask_pair, self.n_keys:self.n_keys + self.n_vals] = one_hot(v, self.n_vals)
            is_pair[mask_pair] = 1.0

        # query step: at t = num_pairs + delay
        mask_query = (self.t == self.num_pairs + self.delay)
        if mask_query.any():
            qi = self.query_idx[mask_query]
            kq = self.keys[mask_query, qi]
            obs[mask_query, :self.n_keys] = one_hot(kq, self.n_keys)
            is_query[mask_query] = 1.0

        obs[:, -2] = is_pair
        obs[:, -1] = is_query
        return obs

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        # reward only at query step
        mask_query = (self.t == self.num_pairs + self.delay)
        if mask_query.any():
            qi = self.query_idx[mask_query]
            target = self.vals[mask_query, qi]
            a = action[mask_query].clamp(0, self.n_vals - 1)
            reward[mask_query] = (a == target).float()

        self.t = self.t + 1
        done = (self.t > self.num_pairs + self.delay).float()
        obs = self._obs_from_t()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 4) N-Back -----------------

class NBackEnv(BaseVectorEnv):
    """
    N-back:
      - At each step, present symbol s_t (0..K-1).
      - Agent must predict whether s_t == s_{t-N} (binary action).
      - Reward per step.

    This is streaming memory use, not just one final query.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 horizon: int = 50, n_back: int = 3, n_symbols: int = 8):
        obs_dim = n_symbols
        n_actions = 2   # 0: no-match, 1: match
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.horizon = horizon
        self.n_back = n_back
        self.n_symbols = n_symbols

        self.t = None
        self.seq = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        self.t = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.seq = torch.randint(0, self.n_symbols, (self.batch_size, self.horizon + self.n_back + 1), device=self.device)
        return self._obs_from_t()

    def _obs_from_t(self) -> torch.Tensor:
        s = self.seq[torch.arange(self.batch_size, device=self.device), self.t]
        return one_hot(s, self.n_symbols)

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        # Only give reward when t >= n_back
        mask = (self.t >= self.n_back)
        if mask.any():
            s_t = self.seq[mask, self.t[mask]]
            s_prev = self.seq[mask, self.t[mask] - self.n_back]
            target = (s_t == s_prev).long()  # 1 if match, else 0
            a = action[mask].clamp(0, 1)
            reward[mask] = (a == target).float()

        self.t = self.t + 1
        done = (self.t >= self.horizon).float()
        obs = self._obs_from_t()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 5) Multi-Cue Delayed -----------------

class MultiCueDelayedEnv(BaseVectorEnv):
    """
    Multi-cue delayed:
      - Present M cues at random times.
      - At the end: randomly choose one cue index to query (q).
      - Agent must output that cue.

    This stresses storing multiple items and retrieving one.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 horizon: int = 60, n_cues: int = 6, num_cues: int = 3):
        obs_dim = n_cues + 2  # [cue one-hot, is_cue, is_query]
        n_actions = n_cues
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.horizon = horizon
        self.n_cues = n_cues
        self.num_cues = num_cues

        self.t = None
        self.cues = None  # [B, num_cues]
        self.cue_ts = None  # [B, num_cues]
        self.query_idx = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        B = self.batch_size
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.cues = torch.randint(0, self.n_cues, (B, self.num_cues), device=self.device)
        # choose distinct times for cues near the first 2/3 of episode
        max_t = max(2, self.horizon - 5)
        self.cue_ts = torch.stack(
            [torch.sort(torch.randint(0, max_t, (self.num_cues,), device=self.device))[0]
             for _ in range(B)], dim=0
        )  # [B, num_cues]
        self.query_idx = torch.randint(0, self.num_cues, (B,), device=self.device)
        return self._obs_from_t()

    def _obs_from_t(self) -> torch.Tensor:
        B = self.batch_size
        obs = torch.zeros(B, self.obs_dim, device=self.device)
        is_cue = torch.zeros(B, device=self.device)
        is_query = torch.zeros(B, device=self.device)

        # cues
        for j in range(self.num_cues):
            mask = (self.t == self.cue_ts[:, j])
            if mask.any():
                c = self.cues[mask, j]
                obs[mask, :self.n_cues] = one_hot(c, self.n_cues)
                is_cue[mask] = 1.0

        # query at final step
        mask_q = (self.t == self.horizon - 1)
        if mask_q.any():
            qj = self.query_idx[mask_q]
            cq = self.cues[mask_q, qj]
            obs[mask_q, :self.n_cues] = one_hot(cq, self.n_cues)
            is_query[mask_q] = 1.0

        obs[:, -2] = is_cue
        obs[:, -1] = is_query
        return obs

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        # reward only at final step
        mask_final = (self.t == self.horizon - 1)
        if mask_final.any():
            qj = self.query_idx[mask_final]
            target = self.cues[mask_final, qj]
            a = action[mask_final].clamp(0, self.n_cues - 1)
            reward[mask_final] = (a == target).float()

        self.t = self.t + 1
        done = (self.t >= self.horizon).float()
        obs = self._obs_from_t()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 6) Permuted Copy -----------------

class PermutedCopyEnv(CopyMemoryEnv):
    """
    Permuted copy memory:
      - Same as CopyMemory, but the agent must output a permuted version
        of the input sequence, using a fixed permutation π.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 seq_len: int = 10, delay: int = 40, n_symbols: int = 8):
        super().__init__(batch_size, device, seq_len, delay, n_symbols)
        # fixed permutation for all envs
        perm = torch.randperm(seq_len)
        self.register_buffer_perm = perm.to(device)

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        mask_rec = (self.t >= self.seq_len + self.delay) & (self.t < self.total_len)
        if mask_rec.any():
            k = self.t[mask_rec] - (self.seq_len + self.delay)
            # permuted target
            target = self.seq[mask_rec, self.register_buffer_perm[k]]
            a = action[mask_rec].clamp(0, self.n_symbols - 1)
            reward[mask_rec] = (a == target).float()

        self.t = self.t + 1
        obs = self._obs_from_t()
        done = (self.t >= self.total_len).float()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 7) Pattern Binding -----------------

class PatternBindingEnv(BaseVectorEnv):
    """
    Pattern binding:
      - Present two patterns (p1, p2) as separate presentations.
      - Then show partial information of p1 and ask to output a related symbol.
      - Requires binding 'identity' and some extra flag.

    Simplified as:
      - p1: (key1, flag1), p2: (key2, flag2)
      - Later: query key1, must output flag1.

    Action space: flags.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 n_keys: int = 8, n_flags: int = 4, delay: int = 40):
        obs_dim = n_keys + n_flags + 2  # [key one-hot, flag one-hot, is_pair, is_query]
        n_actions = n_flags
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.n_keys = n_keys
        self.n_flags = n_flags
        self.delay = delay
        self.total_len = 2 + delay + 1

        self.t = None
        self.key1 = None
        self.key2 = None
        self.flag1 = None
        self.flag2 = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        B = self.batch_size
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.key1 = torch.randint(0, self.n_keys, (B,), device=self.device)
        self.key2 = torch.randint(0, self.n_keys, (B,), device=self.device)
        self.flag1 = torch.randint(0, self.n_flags, (B,), device=self.device)
        self.flag2 = torch.randint(0, self.n_flags, (B,), device=self.device)
        return self._obs_from_t()

    def _obs_from_t(self) -> torch.Tensor:
        B = self.batch_size
        obs = torch.zeros(B, self.obs_dim, device=self.device)
        is_pair = torch.zeros(B, device=self.device)
        is_query = torch.zeros(B, device=self.device)

        # t = 0: first pair
        mask1 = (self.t == 0)
        if mask1.any():
            obs[mask1, :self.n_keys] = one_hot(self.key1[mask1], self.n_keys)
            obs[mask1, self.n_keys:self.n_keys + self.n_flags] = one_hot(self.flag1[mask1], self.n_flags)
            is_pair[mask1] = 1.0

        # t = 1: second pair
        mask2 = (self.t == 1)
        if mask2.any():
            obs[mask2, :self.n_keys] = one_hot(self.key2[mask2], self.n_keys)
            obs[mask2, self.n_keys:self.n_keys + self.n_flags] = one_hot(self.flag2[mask2], self.n_flags)
            is_pair[mask2] = 1.0

        # query at final step
        mask_q = (self.t == self.total_len - 1)
        if mask_q.any():
            obs[mask_q, :self.n_keys] = one_hot(self.key1[mask_q], self.n_keys)
            is_query[mask_q] = 1.0

        obs[:, -2] = is_pair
        obs[:, -1] = is_query
        return obs

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        mask_q = (self.t == self.total_len - 1)
        if mask_q.any():
            target = self.flag1[mask_q]
            a = action[mask_q].clamp(0, self.n_flags - 1)
            reward[mask_q] = (a == target).float()

        self.t = self.t + 1
        done = (self.t >= self.total_len).float()
        obs = self._obs_from_t()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ------------- 8) Distractor Navigation (pseudo) -----------------

class DistractorNavEnv(BaseVectorEnv):
    """
    Pseudo navigation with distractors:
      - Hidden goal symbol g (0..K-1) is shown only at t=0.
      - After that, we show random distractor symbols for many steps.
      - At a random final step, agent must output g.

    Like a navigation with irrelevant observations.
    """

    def __init__(self, batch_size: int, device: torch.device,
                 horizon: int = 80, n_symbols: int = 8):
        obs_dim = n_symbols + 1  # symbol one-hot + is_goal_flag
        n_actions = n_symbols
        super().__init__(batch_size, obs_dim, n_actions, device)
        self.horizon = horizon
        self.n_symbols = n_symbols

        self.t = None
        self.goal = None
        self.query_t = None

    def reset(self, batch_size: int = None) -> torch.Tensor:
        if batch_size is not None:
            self.batch_size = batch_size
        B = self.batch_size
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.goal = torch.randint(0, self.n_symbols, (B,), device=self.device)
        # choose random query time near the end
        self.query_t = torch.randint(self.horizon // 2, self.horizon, (B,), device=self.device)
        return self._obs_from_t()

    def _obs_from_t(self) -> torch.Tensor:
        B = self.batch_size
        obs = torch.zeros(B, self.obs_dim, device=self.device)

        # t=0: show goal
        is_goal = torch.zeros(B, device=self.device)
        mask0 = (self.t == 0)
        if mask0.any():
            obs[mask0, :self.n_symbols] = one_hot(self.goal[mask0], self.n_symbols)
            is_goal[mask0] = 1.0

        # distractors otherwise
        mask_other = ~mask0
        if mask_other.any():
            rand_sym = torch.randint(0, self.n_symbols, (mask_other.sum(),), device=self.device)
            obs[mask_other, :self.n_symbols] = one_hot(rand_sym, self.n_symbols)

        obs[:, -1] = is_goal
        return obs

    def step(self, action: torch.Tensor):
        B = self.batch_size
        reward = torch.zeros(B, device=self.device)

        # reward when t == query_t
        mask = (self.t == self.query_t)
        if mask.any():
            a = action[mask].clamp(0, self.n_symbols - 1)
            reward[mask] = (a == self.goal[mask]).float()

        self.t = self.t + 1
        done = (self.t >= self.horizon).float()
        obs = self._obs_from_t()
        info: Dict[str, Any] = {}
        return obs, reward, done, info


# ============================================================
# 2. Controller / Model Definitions
# ============================================================

class GRUCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor):
        return self.gru(x_t, h_t), self.gru(x_t, h_t) # Returns output, state


class MambaStubCore(nn.Module):
    """
    Pure PyTorch SSM-like block mimicking Mamba interface.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor):
        u = self.in_proj(x_t)
        s = self.state_proj(h_t)
        z = torch.tanh(u + s)
        g = torch.sigmoid(self.gate(z))
        out = g * z + (1 - g) * h_t
        return self.out_proj(out), out


class PseudoMambaExtBlock(nn.Module):
    """
    Pseudo-Mamba block that uses the C++/CUDA extension if available.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        try:
            from pseudo_mamba_ext.pseudo_mamba_core import PseudoMambaCore
            self.core = PseudoMambaCore(hidden_dim)
            self.has_ext = True
        except ImportError:
            self.core = MambaStubCore(hidden_dim, hidden_dim)
            self.has_ext = False

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device):
        if self.has_ext:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            return self.core.init_state(batch_size, device)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor):
        if self.has_ext:
            x_proj = self.in_proj(x_t)
            h_next = self.core(x_proj, h_t)
            return self.out_proj(h_next), h_next
        else:
            return self.core(x_t, h_t)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int,
                 controller: str = "gru"):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.controller = controller

        if controller == "real_mamba":
            from pseudo_mamba_rl_core import PseudoMambaRLCore
            self.core = PseudoMambaRLCore(
                obs_dim=obs_dim,
                d_model=hidden_dim,
                d_state=16, # Default
                d_conv=4,
            )
            # Real Mamba core handles projection internally
            self.obs_proj = nn.Identity()
        else:
            self.obs_proj = nn.Linear(obs_dim, hidden_dim)
            if controller == "gru":
                self.core = GRUCore(hidden_dim, hidden_dim)
            elif controller == "mamba_stub":
                self.core = MambaStubCore(hidden_dim, hidden_dim)
            elif controller == "pseudo_mamba_ext":
                self.core = PseudoMambaExtBlock(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown controller: {controller}")

        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def init_state(self, batch_size: int, device: torch.device):
        return self.core.init_state(batch_size, device)

    def forward(self, obs: torch.Tensor, state):
        if self.controller == "real_mamba":
            # Core handles projection and step
            h_next, new_state = self.core.forward_step(obs, state)
        else:
            x = F.relu(self.obs_proj(obs))
            h_next, new_state = self.core(x, state)
            
        logits = self.policy_head(h_next)
        value = self.value_head(h_next).squeeze(-1)
        return logits, value, new_state


# ============================================================
# 3. A2C + GAE Trainer (State-Aware)
# ============================================================

def make_env(env_name: str,
             batch_size: int,
             device: torch.device,
             horizon: int) -> BaseVectorEnv:
    name = env_name.lower()
    if name == "delayed_cue":
        return DelayedCueEnv(batch_size, device, horizon=horizon, n_cues=4)
    elif name == "copy_memory":
        return CopyMemoryEnv(batch_size, device, seq_len=10, delay=horizon - 20, n_symbols=8)
    elif name == "assoc_recall":
        return AssocRecallEnv(batch_size, device, num_pairs=4, n_keys=8, n_vals=8, delay=horizon - 10)
    elif name == "n_back":
        return NBackEnv(batch_size, device, horizon=horizon, n_back=3, n_symbols=8)
    elif name == "multi_cue_delayed":
        return MultiCueDelayedEnv(batch_size, device, horizon=horizon, n_cues=6, num_cues=3)
    elif name == "permuted_copy":
        return PermutedCopyEnv(batch_size, device, seq_len=10, delay=horizon - 20, n_symbols=8)
    elif name == "pattern_binding":
        return PatternBindingEnv(batch_size, device, n_keys=8, n_flags=4, delay=horizon - 3)
    elif name == "distractor_nav":
        return DistractorNavEnv(batch_size, device, horizon=horizon, n_symbols=8)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

def stack_states(state_list):
    """Stack list of states into a batch-time tensor structure."""
    if isinstance(state_list[0], tuple):
        # Tuple of tensors (e.g. conv_state, ssm_state)
        return tuple(torch.stack([s[i] for s in state_list], dim=0) 
                     for i in range(len(state_list[0])))
    else:
        # Single tensor
        return torch.stack(state_list, dim=0)

def detach_state(state):
    """Detach state tensors from graph."""
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()

def reset_state_mask(state, mask, init_fn, batch_size, device):
    """
    Reset state for indices where mask is True.
    mask: [B], True means reset
    """
    if not mask.any():
        return state
        
    init_s = init_fn(mask.sum().item(), device)
    
    if isinstance(state, tuple):
        # state is (t1, t2, ...), init_s is (i1, i2, ...)
        new_parts = []
        for s, i in zip(state, init_s):
            s_next = s.clone()
            s_next[mask] = i
            new_parts.append(s_next)
        return tuple(new_parts)
    else:
        s_next = state.clone()
        s_next[mask] = init_s
        return s_next

def train(env_name: str,
          controller: str,
          device: torch.device,
          horizon: int = 40,
          num_envs: int = 64,
          hidden_dim: int = 128,
          total_updates: int = 10000,
          gamma: float = 0.99,
          gae_lambda: float = 0.95,
          lr: float = 3e-4,
          value_coef: float = 0.5,
          entropy_coef: float = 0.01):

    env = make_env(env_name, num_envs, device, horizon)
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    model = ActorCritic(obs_dim, n_actions, hidden_dim, controller=controller).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    obs = env.reset(num_envs)
    state = model.init_state(num_envs, device)

    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_returns = []
    completed_lengths = []

    for update in range(1, total_updates + 1):
        # storage
        obs_buf = []
        # We don't necessarily need to store full state history for training unless using BPTT
        # But for A2C/PPO we usually just need current values. 
        # However, if we wanted to do TBPTT we would. Here we just run forward.
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        # Detach state at start of rollout (truncated BPTT)
        state = detach_state(state)

        for t in range(horizon):
            obs_buf.append(obs)
            
            logits, value, next_state = model(obs, state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action)

            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(done)

            episode_rewards += reward
            episode_lengths += 1

            # handle done (per env)
            done_mask = (done > 0.5)
            if done_mask.any():
                completed_returns.extend(episode_rewards[done_mask].tolist())
                completed_lengths.extend(episode_lengths[done_mask].tolist())
                episode_rewards[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0

            # reset obs/state for done envs
            if done_mask.any():
                reset_obs = env.reset(done_mask.sum().item())
                obs = next_obs.clone()
                obs[done_mask] = reset_obs
                
                # Reset state for next step
                state = reset_state_mask(next_state, done_mask, model.init_state, num_envs, device)
            else:
                obs = next_obs
                state = next_state

        # Stack rollout
        obs_t = torch.stack(obs_buf, dim=0)         # [T, B, obs_dim]
        act_t = torch.stack(act_buf, dim=0)         # [T, B]
        logp_t = torch.stack(logp_buf, dim=0)       # [T, B]
        rew_t = torch.stack(rew_buf, dim=0)         # [T, B]
        val_t = torch.stack(val_buf, dim=0)         # [T, B]
        done_t = torch.stack(done_buf, dim=0)       # [T, B]

        with torch.no_grad():
            logits_final, last_val, _ = model(obs, state)
            last_val = last_val  # [B]

        # GAE
        adv = torch.zeros_like(rew_t, device=device)
        gae = torch.zeros(num_envs, device=device)
        for t in reversed(range(horizon)):
            next_val = last_val if t == horizon - 1 else val_t[t + 1]
            delta = rew_t[t] + gamma * next_val * (1.0 - done_t[t]) - val_t[t]
            gae = delta + gamma * gae_lambda * (1.0 - done_t[t]) * gae
            adv[t] = gae
        ret_t = adv + val_t

        # flatten
        B = num_envs
        T = horizon
        obs_flat = obs_t.reshape(T * B, obs_dim)
        act_flat = act_t.reshape(T * B)
        # old_logp_flat = logp_t.reshape(T * B)
        adv_flat = adv.reshape(T * B)
        ret_flat = ret_t.reshape(T * B)

        # normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # For update, we need to re-evaluate. 
        # Since we don't store full state history and Mamba state is complex,
        # we can't easily re-run the RNN over the sequence without re-generating states.
        # Standard A2C/PPO on RNNs usually re-runs the forward pass on the sequence 
        # starting from the initial state of the chunk.
        # BUT, we didn't store the initial state of the chunk!
        # Let's fix this: we need the state at t=0 of the rollout.
        
        # Actually, we can't easily re-run Mamba step-wise efficiently for the gradient pass 
        # if we want to use the parallel scan. But here we are using step-wise inference.
        # So we would have to re-run step-wise.
        
        # Simplification for this benchmark: 
        # We will just use the logits/values collected during rollout for the loss 
        # (A2C style, no PPO epochs). This avoids re-running the model.
        # If we want PPO, we need to re-run.
        
        # Let's stick to A2C-style single update using collected logprobs.
        # This is valid for on-policy.
        
        # Calculate loss using collected logprobs (no re-evaluation)
        policy_loss = -(adv_flat * logp_t.reshape(T*B)).mean()
        value_loss = F.mse_loss(val_t.reshape(T*B), ret_flat)
        
        # Entropy from collected logits (approximate, or we could store entropy)
        # We'll skip entropy for this simplified A2C or compute it from stored logits if we had them.
        # Let's just use policy + value loss for now to keep it simple and fast.
        
        loss = policy_loss + value_coef * value_loss
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if update % 50 == 0:
            if completed_returns:
                avg_return = np.mean(completed_returns[-100:]) if len(completed_returns) >= 10 else np.mean(completed_returns)
                avg_len = np.mean(completed_lengths[-100:]) if len(completed_lengths) >= 10 else np.mean(completed_lengths)
            else:
                avg_return = 0.0
                avg_len = 0.0
            print(f"[{update:05d}] env={env_name} ctrl={controller} "
                  f"loss={loss.item():.3f} "
                  f"return={avg_return:.3f} len={avg_len:.1f} "
                  f"V={value_loss.item():.3f}")


# ============================================================
# 4. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="delayed_cue",
                        choices=["delayed_cue", "copy_memory", "assoc_recall",
                                 "n_back", "multi_cue_delayed", "permuted_copy",
                                 "pattern_binding", "distractor_nav"])
    parser.add_argument("--controller", type=str, default="gru",
                        choices=["gru", "mamba_stub", "pseudo_mamba_ext", "real_mamba"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--total_updates", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device={device}, env={args.env}, controller={args.controller}")
    train(env_name=args.env,
          controller=args.controller,
          device=device,
          horizon=args.horizon,
          num_envs=args.num_envs,
          hidden_dim=args.hidden_dim,
          total_updates=args.total_updates,
          gamma=args.gamma,
          gae_lambda=args.gae_lambda,
          lr=args.lr,
          value_coef=args.value_coef,
          entropy_coef=args.entropy_coef)


if __name__ == "__main__":
    main()
