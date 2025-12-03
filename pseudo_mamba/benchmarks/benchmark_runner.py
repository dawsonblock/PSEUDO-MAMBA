import torch
import argparse
import os
import json
import time
from typing import Dict, Any

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

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Init Env
    env_cls = TASK_MAP[args.task]
    # We pass kwargs if needed, for now defaults
    env = env_cls(batch_size=args.num_envs, device=device)
    
    # Init Controller
    controller_cls = CONTROLLER_MAP[args.controller]
    controller = controller_cls(
        input_dim=env.obs_dim,
        hidden_dim=args.hidden_dim,
        feature_dim=args.hidden_dim
    )
    
    # Init ActorCritic
    model = ActorCritic(controller, act_dim=env.act_dim).to(device)
    
    # Init PPO
    ppo = PPO(model, lr=args.lr)
    ppo.set_scheduler(args.total_updates)
    
    # Torch Compile
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        # Re-assign compiled model to ppo
        ppo.actor_critic = model
    
    # Init Buffer
    buffer = RolloutBuffer(
        num_steps=args.rollout_steps,
        num_envs=args.num_envs,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        device=device
    )
    
    # Training Loop
    obs = env.reset()
    state = model.init_state(args.num_envs, device)
    
    results = []
    start_time = time.time()
    
    for update in range(args.total_updates):
        # Rollout
        for step in range(args.rollout_steps):
            with torch.no_grad():
                logits, value, new_state = model.forward_step(obs, state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                
            next_obs, reward, done, info = env.step(action)
            
            # Insert into buffer
            # Note: state stored is the one BEFORE the step? Or AFTER?
            # RolloutBuffer stores state at t+1.
            # We need to store state at t=0 initially.
            if step == 0:
                buffer.states[0] = state
                
            buffer.insert(obs, action, reward, done, value.squeeze(-1), logprob, new_state)
            
            obs = next_obs
            state = new_state
            
            # Reset state for done envs is handled in PPO update via mask, 
            # but for rollout continuity we should reset state here too?
            # Our Controller.reset_mask handles it?
            # No, we need to call reset_mask on state if done.
            if done.any():
                state = model.reset_mask(state, done)
                
        # Compute GAE
        with torch.no_grad():
            _, last_value, _ = model.forward_step(obs, state)
            buffer.compute_gae(last_value.squeeze(-1))
            
        # Update
        metrics = ppo.update(buffer)
        
        # Log
        avg_reward = buffer.rewards.sum() / args.num_envs # Sum over rollout
        metrics["avg_reward"] = avg_reward.item()
        metrics["update"] = update
        metrics["time"] = time.time() - start_time
        
        results.append(metrics)
        
        if update % 10 == 0:
            print(f"Update {update}: Reward={avg_reward.item():.4f}, Loss={metrics['loss']:.4f}, LR={metrics['lr']:.6f}, GradNorm={metrics['grad_norm']:.4f}")
            
    # Save results
    os.makedirs("results", exist_ok=True)
    filename = f"results/{args.task}_{args.controller}_{args.seed}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=TASK_MAP.keys())
    parser.add_argument("--controller", type=str, required=True, choices=CONTROLLER_MAP.keys())
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--total_updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run_benchmark(args)
