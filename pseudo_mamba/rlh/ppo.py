import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any
from .rollout import RolloutBuffer
from .actor_critic import ActorCritic

class PPO:
    """
    Proximal Policy Optimization (PPO) with Recurrent Support.
    Performs full-sequence BPTT by re-running the model on rollout trajectories.
    """
    def __init__(self, 
                 actor_critic: ActorCritic,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_param: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 num_minibatches: int = 4):
        
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        batch = buffer.get_batch()
        
        # [T, B, ...]
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        old_values = batch["values"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        dones = batch["dones"] # [T, B] - masks for step t (reset at t means state at t was reset)
        # Actually dones[t] means step t-1 resulted in done.
        # So state at t should be reset if dones[t] is True.
        
        # Initial state of the rollout (at t=0)
        initial_state = batch["initial_state"]
        
        T, B = obs.shape[:2]
        batch_size = B
        minibatch_size = batch_size // self.num_minibatches
        
        total_loss_sum = 0
        pg_loss_sum = 0
        val_loss_sum = 0
        entropy_sum = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Shuffle env indices
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size
                mb_indices = indices[start_idx:end_idx]
                
                # Get minibatch data (slice along batch dim)
                mb_obs = obs[:, mb_indices] # [T, mb, D]
                mb_actions = actions[:, mb_indices] # [T, mb]
                mb_old_logprobs = old_logprobs[:, mb_indices]
                mb_advantages = advantages[:, mb_indices]
                mb_returns = returns[:, mb_indices]
                mb_dones = dones[:, mb_indices]
                
                # Initial state for this minibatch
                # We need to slice the state object.
                # State structure depends on controller.
                # We need a helper to slice state.
                mb_state = self._slice_state(initial_state, mb_indices)
                
                # Re-run forward pass (BPTT)
                new_logprobs_list = []
                new_values_list = []
                entropy_list = []
                
                current_state = mb_state
                
                for t in range(T):
                    # Reset state if done
                    # dones[t] indicates if the previous step finished.
                    # If dones[t] is True, the current state should be reset (or masked).
                    # Our Controller.reset_mask handles this.
                    current_state = self.actor_critic.reset_mask(current_state, mb_dones[t])
                    
                    logits, val, current_state = self.actor_critic.forward_step(mb_obs[t], current_state)
                    
                    dist = Categorical(logits=logits)
                    new_logprob = dist.log_prob(mb_actions[t])
                    entropy = dist.entropy()
                    
                    new_logprobs_list.append(new_logprob)
                    new_values_list.append(val.squeeze(-1))
                    entropy_list.append(entropy)
                    
                # Stack
                new_logprobs = torch.stack(new_logprobs_list) # [T, mb]
                new_values = torch.stack(new_values_list) # [T, mb]
                entropies = torch.stack(entropy_list) # [T, mb]
                
                # PPO Loss
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy
                entropy_loss = -entropies.mean()
                
                loss = pg_loss + self.value_loss_coef * v_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss_sum += loss.item()
                pg_loss_sum += pg_loss.item()
                val_loss_sum += v_loss.item()
                entropy_sum += entropies.mean().item()
                num_updates += 1
                
        return {
            "loss": total_loss_sum / num_updates,
            "pg_loss": pg_loss_sum / num_updates,
            "val_loss": val_loss_sum / num_updates,
            "entropy": entropy_sum / num_updates
        }

    def _slice_state(self, state: Any, indices: torch.Tensor) -> Any:
        """
        Helper to slice the state object by batch indices.
        Handles Tensor, Tuple of Tensors, etc.
        """
        if isinstance(state, torch.Tensor):
            return state[indices]
        elif isinstance(state, tuple):
            return tuple(self._slice_state(s, indices) for s in state)
        elif isinstance(state, list):
            return [self._slice_state(s, indices) for s in state]
        elif state is None:
            return None
        else:
            # Try to handle dataclass or object
            # For now assume tensor/tuple structure
            raise ValueError(f"Unsupported state type for slicing: {type(state)}")
