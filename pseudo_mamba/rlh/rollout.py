import torch
from typing import List, Dict, Any, Optional

class RolloutBuffer:
    """
    Buffer for storing trajectories and computing GAE.
    Supports recurrent states.
    """
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Buffers
        self.obs = torch.zeros(num_steps + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device) # Discrete actions
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps + 1, num_envs, device=device)
        self.values = torch.zeros(num_steps + 1, num_envs, device=device)
        self.logprobs = torch.zeros(num_steps, num_envs, device=device)
        
        # Recurrent states
        # We store the state at each step to allow re-starting BPTT from any point if needed,
        # or just the initial state of the rollout.
        # For full-sequence BPTT, we only strictly need the initial state of the rollout.
        # But if we minibatch over time (chunked BPTT), we need states at chunk boundaries.
        # Let's store all states for maximum flexibility (memory permitting).
        # Since state structure varies (tuple, tensor), we use a list of objects or a specialized container.
        # For simplicity, we'll store a list of states.
        self.states = [None] * (num_steps + 1)
        
        self.step = 0

    def insert(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, 
               done: torch.Tensor, value: torch.Tensor, logprob: torch.Tensor, state: Any):
        """
        Insert a step into the buffer.
        """
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step + 1].copy_(done)
        self.values[self.step].copy_(value)
        self.logprobs[self.step].copy_(logprob)
        self.states[self.step + 1] = state # Store state AFTER step (for next input)
        # Note: states[0] should be set at reset/init.
        
        self.step = (self.step + 1) % self.num_steps

    def compute_gae(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute GAE and Returns.
        """
        self.values[self.num_steps].copy_(last_value)
        
        advantage = torch.zeros(self.num_envs, device=self.device)
        self.advantages = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.returns = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        
        for t in reversed(range(self.num_steps)):
            # If done, next value is 0 (masked by 1 - done)
            next_non_terminal = 1.0 - self.dones[t + 1]
            next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            
            self.advantages[t] = advantage
            self.returns[t] = advantage + self.values[t]
            
    def get_batch(self) -> Dict[str, Any]:
        """
        Return the full batch for PPO update.
        We return tensors shaped [T, B, ...]
        """
        return {
            "obs": self.obs[:-1], # [T, B, D]
            "actions": self.actions, # [T, B]
            "values": self.values[:-1], # [T, B]
            "logprobs": self.logprobs, # [T, B]
            "advantages": self.advantages, # [T, B]
            "returns": self.returns, # [T, B]
            "dones": self.dones[:-1], # [T, B]
            "initial_state": self.states[0] # State at t=0
        }
