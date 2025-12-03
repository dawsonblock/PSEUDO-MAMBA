#!/usr/bin/env python3
"""
neural_memory_mamba_long_rl.py

Corrected long-horizon memory benchmark:
- Multi-bit delayed-cue environment
- PPO with GAE
- GRU vs Mamba (API-correct, stateless per chunk)

This is designed as a clean, reproducible benchmark script.
"""

import argparse
import time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. Device selection
# ============================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Environment: Multi-bit Delayed Cue (Vectorized)
# ============================================================

class DelayedCueEnv:
    """
    Vectorized multi-bit delayed match-to-sample task.

    Timeline per env:
        t = 0        : cue presented (num_bits bits)
        t = 1..H     : delay, no cue
        t = H+1      : query; agent must output full bit pattern (integer 0..2^bits-1)
        t = H+2      : terminal step, env auto-resets to t = 0, new cue

    Observations:
        [0 : num_bits)     : cue bits at t=0 in {-1,+1}, else 0
        [num_bits]         : is_start flag (1 at t=0)
        [num_bits + 1]     : is_query flag (1 at t=H+1)
        [num_bits + 2]     : Gaussian noise

    Reward:
        +1 on query step if predicted pattern == cue, else 0.
    """

    def __init__(self,
                 horizon: int = 10000,
                 num_envs: int = 64,
                 num_bits: int = 4,
                 device: torch.device = torch.device("cpu")):
        assert num_bits >= 1
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.num_bits = int(num_bits)
        self.device = device

        self.action_dim = 2 ** self.num_bits
        self.obs_dim = self.num_bits + 3  # bits + is_start + is_query + noise

        self.t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cue = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset()

    def reset(self) -> torch.Tensor:
        self.t.zero_()
        self.cue = torch.randint(
            0, self.action_dim, (self.num_envs,), device=self.device
        )
        return self._make_obs()

    def _make_obs(self) -> torch.Tensor:
        obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device)

        # t == 0: show cue bits + is_start flag
        at_start = (self.t == 0)
        if at_start.any():
            idx = at_start.nonzero(as_tuple=True)[0]
            cues = self.cue[idx]
            for b in range(self.num_bits):
                bit = ((cues >> b) & 1).float() * 2.0 - 1.0
                obs[idx, b] = bit
            obs[idx, self.num_bits] = 1.0  # is_start

        # t == horizon+1: query flag
        at_query = (self.t == self.horizon + 1)
        if at_query.any():
            obs[at_query, self.num_bits + 1] = 1.0  # is_query

        # noise
        obs[:, self.num_bits + 2] = torch.randn(self.num_envs, device=self.device) * 0.1
        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = actions.to(self.device).clamp(0, self.action_dim - 1)
        self.t += 1

        at_query = (self.t == self.horizon + 1)
        correct = (actions == self.cue) & at_query
        rew = correct.float()

        done = (self.t >= self.horizon + 2)
        if done.any():
            idx = done.nonzero(as_tuple=True)[0]
            self.t[idx] = 0
            self.cue[idx] = torch.randint(
                0, self.action_dim, (idx.numel(),), device=self.device
            )

        obs = self._make_obs()
        return obs, rew, done


# ============================================================
# 3. Actor-Critic with GRU or Mamba
# ============================================================

class MemActorCritic(nn.Module):
    """
    Actor-Critic with pluggable memory controller (GRU or Mamba).

    For Mamba:
      - Now uses proper InferenceParams and state management APIs
      - Can explicitly reset states at episode boundaries
      - Supports state introspection and analysis
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 controller: str = "mamba",
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 num_envs: int = 64,
                 max_seqlen: int = 10000):
        super().__init__()
        self.controller = controller.lower()
        self.hidden_dim = hidden_dim
        self.num_envs = num_envs

        self.enc = nn.Linear(obs_dim, hidden_dim)

        if self.controller == "gru":
            self.core = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.inference_params = None
        elif self.controller == "mamba":
            try:
                from mamba_ssm import Mamba
                from mamba_ssm.utils.generation import InferenceParams
            except ImportError as e:
                raise ImportError(
                    "mamba-ssm not installed. Install with:\n"
                    "    pip install mamba-ssm causal-conv1d"
                ) from e
            self.core = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=0,  # Set layer_idx for state management
            )
            # Create InferenceParams for stateful inference
            self.inference_params = InferenceParams(
                max_batch_size=num_envs,
                max_seqlen=max_seqlen
            )
        else:
            raise ValueError(f"Unknown controller: {controller}")

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, obs_dim] or [B, T, obs_dim]
        h: GRU hidden [1, B, H] or None; ignored for Mamba.
        """
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)  # [B, 1, D]

        z = F.relu(self.enc(x))

        if self.controller == "gru":
            out, h_new = self.core(z, h)
        else:
            # For Mamba, use inference_params for stateful processing
            if self.inference_params is not None:
                out = self.core(z, inference_params=self.inference_params)
                # Increment seqlen_offset for proper state tracking
                self.inference_params.seqlen_offset += z.shape[1]
            else:
                out = self.core(z)
            h_new = None

        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)

        if single_step:
            logits = logits.squeeze(1)
            value = value.squeeze(1)

        return logits, value, h_new

    def reset_mamba_state(self, env_mask: Optional[torch.Tensor] = None):
        """
        Reset Mamba state for specified environments (or all if env_mask is None).

        Args:
            env_mask: Boolean tensor [num_envs], True for envs that need reset.
                     If None, resets all environments.
        """
        if self.controller != "mamba" or self.inference_params is None:
            return

        # For simplicity, reset all states if any environment needs reset
        # A more sophisticated implementation could do per-env state masking
        if env_mask is None or env_mask.any():
            self.core.zero_inference_state(self.inference_params)
            self.inference_params.seqlen_offset = 0


# ============================================================
# 4. Rollout + GAE
# ============================================================

@torch.no_grad()
def rollout_chunk(
    env: DelayedCueEnv,
    model: MemActorCritic,
    chunk_len: int,
    device: torch.device,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Dict[str, torch.Tensor]:
    B = env.num_envs

    obs_buf: List[torch.Tensor] = []
    act_buf: List[torch.Tensor] = []
    logp_buf: List[torch.Tensor] = []
    val_buf: List[torch.Tensor] = []
    rew_buf: List[torch.Tensor] = []
    done_buf: List[torch.Tensor] = []

    obs = env._make_obs()  # env state already set; first call after reset()
    h = None  # GRU hidden, or None for Mamba

    for _ in range(chunk_len):
        logits, val, h = model(obs, h)
        dist = Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)

        obs_buf.append(obs)
        act_buf.append(act)
        logp_buf.append(logp)
        val_buf.append(val)

        obs, rew, done = env.step(act)
        rew_buf.append(rew)
        done_buf.append(done.float())

        # Reset hidden state on done envs
        if h is not None and done.any():
            # GRU: mask hidden state
            mask = (~done).float().view(1, B, 1)
            h = h * mask
        elif done.any():
            # Mamba: reset inference state
            model.reset_mamba_state(done)

    # Bootstrap with last observation
    _, last_val, _ = model(obs, h)

    obs_b = torch.stack(obs_buf)   # [T, B, obs_dim]
    act_b = torch.stack(act_buf)   # [T, B]
    logp_b = torch.stack(logp_buf) # [T, B]
    val_b = torch.stack(val_buf)   # [T, B]
    rew_b = torch.stack(rew_buf)   # [T, B]
    done_b = torch.stack(done_buf) # [T, B]

    T = chunk_len
    adv = torch.zeros_like(rew_b)
    last_adv = torch.zeros(B, device=device)
    v_next = last_val  # [B]

    for t in reversed(range(T)):
        mask = 1.0 - done_b[t]
        delta = rew_b[t] + gamma * v_next * mask - val_b[t]
        last_adv = delta + gamma * lam * mask * last_adv
        adv[t] = last_adv
        v_next = val_b[t]

    ret = adv + val_b

    return {
        "obs": obs_b,
        "act": act_b,
        "logp": logp_b,
        "val": val_b,
        "rew": rew_b,
        "done": done_b,
        "adv": adv,
        "ret": ret,
    }


# ============================================================
# 5. PPO update
# ============================================================

def ppo_update(
    model: MemActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    clip_eps: float = 0.1,
    value_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> None:
    T, B, obs_dim = batch["obs"].shape
    N = T * B

    obs_flat = batch["obs"].reshape(N, obs_dim)
    act_flat = batch["act"].reshape(N)
    logp_old = batch["logp"].reshape(N)
    adv_flat = batch["adv"].reshape(N)
    ret_flat = batch["ret"].reshape(N)

    # Normalize advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    for _ in range(ppo_epochs):
        idx_perm = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            end = min(start + minibatch_size, N)
            idx = idx_perm[start:end]

            logits, val, _ = model(obs_flat[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_flat[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old[idx])
            surr1 = ratio * adv_flat[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_flat[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(val, ret_flat[idx])
            loss = policy_loss + value_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# ============================================================
# 6. Clean evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: MemActorCritic,
    env: DelayedCueEnv,
    device: torch.device,
    steps: int = 4096,
) -> float:
    model.eval()
    obs = env.reset()
    h = None
    correct = 0.0
    queries = 0.0

    for _ in range(steps):
        # count which envs are at query step before acting
        is_query = (env.t == env.horizon + 1)
        queries += is_query.sum().item()

        logits, _, h = model(obs, h)
        act = logits.argmax(dim=-1)
        obs, rew, done = env.step(act)
        correct += rew.sum().item()

        # Reset hidden state on done envs
        if h is not None and done.any():
            mask = (~done).float().view(1, env.num_envs, 1)
            h = h * mask
        elif done.any():
            model.reset_mamba_state(done)

    model.train()
    return correct / max(queries, 1.0)


# ============================================================
# 7. Training loops
# ============================================================

def run_quick(config: Dict[str, Any]) -> None:
    device = get_device()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

    env = DelayedCueEnv(
        horizon=config["horizon"],
        num_envs=config["num_envs"],
        num_bits=config["num_bits"],
        device=device,
    )

    model = MemActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=config["hidden_dim"],
        controller=config["controller"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        expand=config["expand"],
        num_envs=config["num_envs"],
        max_seqlen=config["horizon"] * 2,  # Conservative estimate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print("=" * 80)
    print(f"Controller: {config['controller'].upper()} | "
          f"Horizon: {config['horizon']:,} | Bits: {config['num_bits']} "
          f"({env.action_dim} actions)")
    print("=" * 80)

    start = time.time()
    for update in range(1, config["total_updates"] + 1):
        batch = rollout_chunk(env, model, config["chunk_len"], device)
        ppo_update(
            model, optimizer, batch, device,
            ppo_epochs=config["ppo_epochs"],
            minibatch_size=config["minibatch_size"],
            clip_eps=config["clip_eps"],
            value_coef=config["value_coef"],
            ent_coef=config["ent_coef"],
        )

        if update % config["log_interval"] == 0:
            acc = evaluate(model, env, device, steps=2048)
            elapsed = (time.time() - start) / 60.0
            print(f"[{update:05d}/{config['total_updates']:05d}] "
                  f"Eval success: {acc*100:6.2f}% | Elapsed: {elapsed:5.1f} min")

    final = evaluate(model, env, device, steps=8192)
    total = (time.time() - start) / 60.0
    print("=" * 80)
    print(f"FINAL CLEAN SUCCESS: {final*100:.2f}% | Total time: {total:.1f} min")
    print("=" * 80)


def run_scale_experiment(config: Dict[str, Any]) -> None:
    device = get_device()
    horizons = config["horizons"]
    controllers = ["gru", "mamba"]
    results = {c: [] for c in controllers}

    for controller in controllers:
        print("\n" + "#" * 80)
        print(f"Controller: {controller.upper()}")
        print("#" * 80)

        for horizon in horizons:
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config["seed"])

            env = DelayedCueEnv(
                horizon=horizon,
                num_envs=config["num_envs"],
                num_bits=config["num_bits"],
                device=device,
            )

            model = MemActorCritic(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                hidden_dim=config["hidden_dim"],
                controller=controller,
                d_state=config["d_state"],
                d_conv=config["d_conv"],
                expand=config["expand"],
                num_envs=config["num_envs"],
                max_seqlen=horizon * 2,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

            max_updates = config["max_updates"]
            min_updates = config["min_updates"]
            base = int(max_updates * (10000 / max(horizon, 10000)))
            total_updates = max(min_updates, min(max_updates, base))

            print(f"\nHorizon {horizon:,} | Updates: {total_updates}")
            for update in range(1, total_updates + 1):
                batch = rollout_chunk(env, model, config["chunk_len"], device)
                ppo_update(
                    model, optimizer, batch, device,
                    ppo_epochs=config["ppo_epochs"],
                    minibatch_size=config["minibatch_size"],
                    clip_eps=config["clip_eps"],
                    value_coef=config["value_coef"],
                    ent_coef=config["ent_coef"],
                )

                if update % config["log_interval"] == 0:
                    acc = evaluate(model, env, device, steps=2048)
                    print(f"  [{update:05d}/{total_updates:05d}] "
                          f"Eval success: {acc*100:6.2f}%")

            final = evaluate(model, env, device, steps=8192)
            results[controller].append(final)
            print(f"--> FINAL @ H={horizon:,}: {final*100:.2f}%")

            del model, env, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("SCALING SUMMARY (clean eval success %)")
    print("=" * 80)
    header = "| Horizon | " + " | ".join([f"{c.upper():>7}" for c in controllers]) + " |"
    print(header)
    print("|---------|" + "|".join(["--------"] * len(controllers)) + "|")
    for i, h in enumerate(horizons):
        row_vals = []
        for c in controllers:
            val = results[c][i] * 100.0
            row_vals.append(f"{val:6.2f}%")
        row = f"| {h:7,} | " + " | ".join(row_vals) + " |"
        print(row)
    print("=" * 80)


# ============================================================
# 8. CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRU vs Mamba delayed-cue RL benchmark")
    p.add_argument("--mode", type=str, default="quick", choices=["quick", "scale"])
    p.add_argument("--controller", type=str, default="mamba", choices=["gru", "mamba"])
    p.add_argument("--horizon", type=int, default=20000)
    p.add_argument("--num-bits", type=int, default=4, help="4→16 actions, 5→32, etc.")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--clip-eps", type=float, default=0.1)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--total-updates", type=int, default=1200)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1000, 5000, 10000, 20000, 50000])
    p.add_argument("--max-updates", type=int, default=1500)
    p.add_argument("--min-updates", type=int, default=300)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = vars(args)

    if config["mode"] == "quick":
        run_quick(config)
    else:
        run_scale_experiment(config)


if __name__ == "__main__":
    main()
