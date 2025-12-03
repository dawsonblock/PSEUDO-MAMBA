#!/usr/bin/env python3
"""
Pseudo-Mamba Benchmark Runner

Runs the vectorized memory environments from pseudo_mamba_memory_suite.py
across multiple controllers and writes a JSON summary.

Example:

    python pseudo_mamba_benchmark.py \\
        --envs delayed_cue copy_memory assoc_recall \\
        --controllers gru mamba_stub real_mamba pseudo_mamba_ext \\
        --horizon 1000 \\
        --num_envs 64 \\
        --hidden_dim 128 \\
        --total_updates 5000 \\
        --out results/benchmark_summary.json
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any

import torch
import numpy as np

# Import from our package
from pseudo_mamba.controllers.gru import GRUController
from pseudo_mamba.controllers.mamba import MambaController
from pseudo_mamba.controllers.pseudo_mamba import PseudoMambaController
from pseudo_mamba.rlh.actor_critic import ActorCritic
from pseudo_mamba.rlh.rollout import RolloutBuffer
from pseudo_mamba.rlh.ppo import PPO

from pseudo_mamba.envs.delayed_cue import DelayedCueEnv
from pseudo_mamba.envs.copy_memory import CopyMemoryEnv
from pseudo_mamba.envs.assoc_recall import AssocRecallEnv
from pseudo_mamba.envs.n_back import NBackEnv
from pseudo_mamba.envs.multi_cue_delay import MultiCueDelayEnv
from pseudo_mamba.envs.permuted_copy import PermutedCopyEnv
from pseudo_mamba.envs.pattern_binding import PatternBindingEnv
from pseudo_mamba.envs.distractor_nav import DistractorNavEnv

TASK_MAP = {
    "delayed_cue": DelayedCueEnv,
    "copy_memory": CopyMemoryEnv,
    "assoc_recall": AssocRecallEnv,
    "n_back": NBackEnv,
    "multi_cue_delay": MultiCueDelayEnv,
    "permuted_copy": PermutedCopyEnv,
    "pattern_binding": PatternBindingEnv,
    "distractor_nav": DistractorNavEnv
}

CONTROLLER_MAP = {
    "gru": GRUController,
    "mamba": MambaController,
    "pseudo_mamba": PseudoMambaController
}

def train(
    env_name: str,
    controller: str,
    device: torch.device,
    horizon: int,
    num_envs: int,
    hidden_dim: int,
    total_updates: int,
    gamma: float,
    gae_lambda: float,
    lr: float,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, Any]:
    
    # Init Env
    env_cls = TASK_MAP[env_name]
    # Assuming envs take horizon as arg if applicable, or we might need to adjust envs
    # For now, we use default args or kwargs if env supports them
    # Most of our envs have hardcoded horizons or defaults, let's assume defaults for now
    # or update envs to accept horizon. 
    # The user's prompt implies envs are configurable.
    # Let's pass horizon if the env accepts it, otherwise ignore.
    try:
        env = env_cls(batch_size=num_envs, device=device, sequence_length=horizon)
    except TypeError:
        env = env_cls(batch_size=num_envs, device=device)
    
    # Init Controller
    controller_cls = CONTROLLER_MAP[controller]
    model_core = controller_cls(
        input_dim=env.obs_dim,
        hidden_dim=hidden_dim,
        feature_dim=hidden_dim
    )
    
    # Init ActorCritic
    model = ActorCritic(model_core, act_dim=env.act_dim).to(device)
    
    # Init PPO
    ppo = PPO(
        model, 
        lr=lr, 
        gamma=gamma, 
        gae_lambda=gae_lambda, 
        value_loss_coef=value_coef, 
        entropy_coef=entropy_coef
    )
    ppo.set_scheduler(total_updates)
    
    # Init Buffer
    # Rollout steps usually = horizon or a fixed chunk
    rollout_steps = min(horizon, 128) 
    buffer = RolloutBuffer(
        num_steps=rollout_steps,
        num_envs=num_envs,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        device=device
    )
    
    obs = env.reset()
    state = model.init_state(num_envs, device)
    
    completed_returns = []
    
    start_time = time.time()
    
    for update in range(total_updates):
        for step in range(rollout_steps):
            with torch.no_grad():
                logits, value, new_state = model.forward_step(obs, state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            
            next_obs, reward, done, info = env.step(action)
            
            if step == 0:
                buffer.states[0] = state
            
            buffer.insert(obs, action, reward, done, value.squeeze(-1), logprob, new_state)
            
            obs = next_obs
            state = new_state
            
            if done.any():
                state = model.reset_mask(state, done)
                # Track returns if available in info? 
                # Our envs don't currently return episode return in info.
                # We can approximate with buffer rewards.
        
        with torch.no_grad():
            _, last_value, _ = model.forward_step(obs, state)
            buffer.compute_gae(last_value.squeeze(-1))
            
        metrics = ppo.update(buffer)
        
        avg_reward = buffer.rewards.sum() / num_envs
        completed_returns.append(avg_reward.item())
        
        if update % 50 == 0:
             print(f"[{update:05d}] env={env_name} ctrl={controller} "
                  f"loss={metrics['loss']:.3f} "
                  f"return={avg_reward.item():.3f} "
                  f"lr={metrics['lr']:.6f}")

    final_avg_return = float(np.mean(completed_returns[-100:])) if len(completed_returns) >= 100 else float(np.mean(completed_returns))
    
    return {
        "env": env_name,
        "controller": controller,
        "final_avg_return": final_avg_return,
        "total_updates": total_updates,
        "horizon": horizon,
        "num_envs": num_envs,
        "hidden_dim": hidden_dim,
        "wall_time": time.time() - start_time
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        default=list(TASK_MAP.keys()),
        help="List of environments to run.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=list(CONTROLLER_MAP.keys()),
        help="List of controllers to benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda).",
    )
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default="results/benchmark_summary.json",
        help="Path to write JSON summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results: List[dict] = []

    for env_name in args.envs:
        if env_name not in TASK_MAP:
            print(f"Skipping unknown env: {env_name}")
            continue
            
        for controller in args.controllers:
            if controller not in CONTROLLER_MAP:
                print(f"Skipping unknown controller: {controller}")
                continue
                
            print(
                f"\n=== Running env={env_name} | controller={controller} "
                f"| horizon={args.horizon} | updates={args.total_updates} ==="
            )
            try:
                summary = train(
                    env_name=env_name,
                    controller=controller,
                    device=device,
                    horizon=args.horizon,
                    num_envs=args.num_envs,
                    hidden_dim=args.hidden_dim,
                    total_updates=args.total_updates,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    lr=args.lr,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                )
                results.append(summary)
            except Exception as e:
                print(f"Failed to run {env_name} with {controller}: {e}")
                import traceback
                traceback.print_exc()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote benchmark summary to: {args.out}")
    print("Sample entries:")
    for row in results[:5]:
        print(row)


if __name__ == "__main__":
    main()
