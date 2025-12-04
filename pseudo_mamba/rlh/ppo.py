import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any
from .rollout import RolloutBuffer
from .actor_critic import ActorCritic

class PPO:
    """
    Proximal Policy Optimization (PPO) with 3-level Recurrent Support.

    Modes:
        - "full": Full-sequence BPTT (re-run entire trajectory with gradients)
        - "truncated": Truncated BPTT with K-step windows
        - "cached": No BPTT in update (use stored values/logprobs only)
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
                 num_minibatches: int = 4,
                 recurrent_mode: str = "full",
                 burn_in_steps: int = 32):
        
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
        self.recurrent_mode = recurrent_mode
        self.burn_in_steps = burn_in_steps
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Validate recurrent mode
        if recurrent_mode not in ["full", "truncated", "cached"]:
            raise ValueError(f"Invalid recurrent_mode: {recurrent_mode}. Must be 'full', 'truncated', or 'cached'.")
        
        # LR Scheduler
        self.scheduler = None
        
    def set_scheduler(self, total_updates: int):
        """
        Initialize a linear decay scheduler.
        """
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Dispatch to appropriate update method based on recurrent_mode.
        """
        if self.recurrent_mode == "cached":
            return self._update_cached(buffer)
        elif self.recurrent_mode == "truncated":
            return self._update_truncated(buffer)
        else:  # "full"
            return self._update_full_bptt(buffer)

    def _update_full_bptt(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Level 3: Full-sequence BPTT (existing behavior).
        Re-runs entire trajectory with gradients.
        """
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
        minibatch_size = max(1, batch_size // self.num_minibatches)

        total_loss_sum = 0
        pg_loss_sum = 0
        val_loss_sum = 0
        entropy_sum = 0
        grad_norm_sum = 0
        explained_var_sum = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Shuffle env indices
            indices = torch.randperm(batch_size, device=obs.device)

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
                with torch.no_grad():
                    var_y = torch.var(y_true)
                    var_y = torch.clamp(var_y, min=1e-8)
                    num = torch.var(y_true - y_pred)
                    explained_var = 1 - (num / var_y)

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

    def _update_cached(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Level 1: Cached mode (K=0).
        No BPTT in update - uses stored logprobs and values from rollout.
        Fast but approximate (ignores recurrent dependencies in gradient).
        """
        batch = buffer.get_batch()

        # [T, B, ...]
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        T, B = obs.shape[:2]

        obs_flat = obs.contiguous().reshape(T * B, -1)
        actions_flat = actions.contiguous().reshape(T * B)
        old_logprobs_flat = old_logprobs.contiguous().reshape(T * B)
        advantages_flat = advantages.contiguous().reshape(T * B)
        returns_flat = returns.contiguous().reshape(T * B)

        # Normalize advantages (single, stable pass)
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std()
        advantages_flat = (advantages_flat - adv_mean) / torch.clamp(adv_std, min=1e-8)

        dataset_size = T * B
        minibatch_size = max(1, dataset_size // self.num_minibatches)

        total_loss_sum = 0
        pg_loss_sum = 0
        val_loss_sum = 0
        entropy_sum = 0
        grad_norm_sum = 0
        explained_var_sum = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size, device=obs.device)

            for start_idx in range(0, dataset_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, dataset_size)
                mb_indices = indices[start_idx:end_idx]

                mb_obs = obs_flat[mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_old_logprobs = old_logprobs_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Forward without recurrent state (treat as stateless)
                    # Initialize fresh state for this batch
                    dummy_state = self.actor_critic.init_state(len(mb_indices), mb_obs.device)
                    logits, values, _ = self.actor_critic.forward_step(mb_obs, dummy_state)

                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()

                    # PPO Loss
                    ratio = torch.exp(new_logprobs - mb_old_logprobs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                    pg_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((values.squeeze(-1) - mb_returns) ** 2).mean()

                    # Entropy
                    entropy_loss = -entropy.mean()

                    loss = pg_loss + self.value_loss_coef * v_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Metrics
                y_pred = values.squeeze(-1).detach()
                y_true = mb_returns
                var_y = torch.var(y_true)
                explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

                total_loss_sum += loss.item()
                pg_loss_sum += pg_loss.item()
                val_loss_sum += v_loss.item()
                entropy_sum += entropy.mean().item()
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

    def _update_truncated(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Level 2: Truncated BPTT with K-step windows.
        Replays K-step windows with gradients for honest recurrent updates.
        """
        batch = buffer.get_batch()

        # [T, B, ...]
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        dones = batch["dones"]
        initial_state = batch["initial_state"]

        T, B = obs.shape[:2]
        if T == 0:
            return {"loss": 0.0, "pg_loss": 0.0, "val_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0, "explained_var": 0.0, "lr": self.optimizer.param_groups[0]["lr"]}
        K = max(1, self.burn_in_steps)

        total_loss_sum = 0
        pg_loss_sum = 0
        val_loss_sum = 0
        entropy_sum = 0
        grad_norm_sum = 0
        explained_var_sum = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Process sliding windows of size K
            for t0 in range(0, T, K):
                t1 = min(t0 + K, T)

                # Get initial state for this window
                if t0 == 0:
                    window_state = initial_state
                else:
                    # Use per-timestep states from buffer for proper truncation
                    # Detach to prevent gradient flow beyond window
                    if "states" in batch:
                        window_state = batch["states"][t0].detach()
                    else:
                        raise ValueError("Truncated BPTT requires per-timestep states in batch['states']")

                # Extract window data
                window_obs = obs[t0:t1]  # [K', B, obs_dim]
                window_actions = actions[t0:t1]
                window_old_logprobs = old_logprobs[t0:t1]
                window_advantages = advantages[t0:t1]
                window_returns = returns[t0:t1]
                window_dones = dones[t0:t1]

                # Re-run forward pass through window
                new_logprobs_list = []
                new_values_list = []
                entropy_list = []

                current_state = window_state

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    for t_rel in range(t1 - t0):
                        t_abs = t0 + t_rel
                        if t_abs > 0:
                            current_state = self.actor_critic.reset_mask(current_state, window_dones[t_rel])

                        logits, val, current_state = self.actor_critic.forward_step(
                            window_obs[t_rel], current_state
                        )

                        dist = Categorical(logits=logits)
                        new_logprob = dist.log_prob(window_actions[t_rel])
                        entropy = dist.entropy()

                        new_logprobs_list.append(new_logprob)
                        new_values_list.append(val.squeeze(-1))
                        entropy_list.append(entropy)

                    # Stack
                    new_logprobs = torch.stack(new_logprobs_list)
                    new_values = torch.stack(new_values_list)
                    entropies = torch.stack(entropy_list)

                    # PPO Loss
                    ratio = torch.exp(new_logprobs - window_old_logprobs)
                    surr1 = ratio * window_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * window_advantages
                    pg_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((new_values - window_returns) ** 2).mean()

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
                y_true = window_returns.flatten()
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
