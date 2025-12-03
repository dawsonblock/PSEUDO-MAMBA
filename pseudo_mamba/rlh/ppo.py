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
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # LR Scheduler
        self.scheduler = None
        
    def set_scheduler(self, total_updates: int):
        """
        Initialize a linear decay scheduler.
        """
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        batch = buffer.get_batch()
        
        # [T, B, ...]
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        old_values = batch["values"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        dones = batch["dones"]
        
        # Initial state of the rollout (at t=0)
        initial_state = batch["initial_state"]
        
        T, B = obs.shape[:2]
        batch_size = B
        minibatch_size = batch_size // self.num_minibatches
        
        total_loss_sum = 0
        pg_loss_sum = 0
        val_loss_sum = 0
        entropy_sum = 0
        grad_norm_sum = 0
        explained_var_sum = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Shuffle env indices
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size
                mb_indices = indices[start_idx:end_idx]
                
                # Get minibatch data
                mb_obs = obs[:, mb_indices]
                mb_actions = actions[:, mb_indices]
                mb_old_logprobs = old_logprobs[:, mb_indices]
                mb_advantages = advantages[:, mb_indices]
                mb_returns = returns[:, mb_indices]
                mb_dones = dones[:, mb_indices]
                
                mb_state = self._slice_state(initial_state, mb_indices)
                
                # Re-run forward pass (BPTT)
                new_logprobs_list = []
                new_values_list = []
                entropy_list = []
                
                current_state = mb_state
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    for t in range(T):
                        current_state = self.actor_critic.reset_mask(current_state, mb_dones[t])
                        
                        logits, val, current_state = self.actor_critic.forward_step(mb_obs[t], current_state)
                        
                        dist = Categorical(logits=logits)
                        new_logprob = dist.log_prob(mb_actions[t])
                        entropy = dist.entropy()
                        
                        new_logprobs_list.append(new_logprob)
                        new_values_list.append(val.squeeze(-1))
                        entropy_list.append(entropy)
                        
                    # Stack
                    new_logprobs = torch.stack(new_logprobs_list)
                    new_values = torch.stack(new_values_list)
                    entropies = torch.stack(entropy_list)
                    
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
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Metrics
                y_pred = new_values.flatten()
                y_true = mb_returns.flatten()
                var_y = torch.var(y_true)
                explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                
                total_loss_sum += loss.item()
                pg_loss_sum += pg_loss.item()
                val_loss_sum += v_loss.item()
                entropy_sum += entropies.mean().item()
                grad_norm_sum += grad_norm.item()
                explained_var_sum += explained_var.item()
                num_updates += 1
        
        # Step LR Scheduler
        if self.scheduler:
            self.scheduler.step()
                
        return {
            "loss": total_loss_sum / num_updates,
            "pg_loss": pg_loss_sum / num_updates,
            "val_loss": val_loss_sum / num_updates,
            "entropy": entropy_sum / num_updates,
            "grad_norm": grad_norm_sum / num_updates,
            "explained_var": explained_var_sum / num_updates,
            "lr": self.optimizer.param_groups[0]["lr"]
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
