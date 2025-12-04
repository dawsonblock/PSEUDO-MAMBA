import os
import math
import random
import importlib
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from pseudo_mamba.rlh.actor_critic import ActorCritic
from pseudo_mamba.rlh.ppo import PPO
from pseudo_mamba.rlh.rollout import RolloutBuffer


# ============================================================
# Helpers
# ============================================================

def select_device(spec: Optional[str]) -> torch.device:
    """
    spec: None | "auto" | "cpu" | "cuda" | "cuda:0" | "cuda:1" | ...
    """
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ============================================================
# Builders
# ============================================================

def build_env_from_config(cfg: Dict[str, Any],
                          job: Dict[str, Any],
                          device: torch.device):
    """
    Build a vectorized environment instance from config + job.

    Expects in cfg["tasks"][task_name]:
        module: "pseudo_mamba.envs.delayed_cue"
        env_class: "DelayedCueEnv"
        kwargs: {...}

    Horizon handling:
        - For most tasks, the env has its own length; we keep job["horizon"]
          purely as a training/benchmark label.
        - If you want horizon to control delay length etc, you can encode that
          in task-specific kwargs or add extra logic here.
    """
    task_name = job["task"]
    task_cfg = cfg["tasks"][task_name]

    EnvCls = load_class(task_cfg["module"], task_cfg["env_class"])

    env_kwargs = dict(task_cfg.get("kwargs", {}))
    
    # If job specifies horizon, pass it as sequence_length if the env supports it
    # Most of our envs take sequence_length in __init__, but DelayedCueEnv takes delay_len
    if "horizon" in job:
        if task_name == "delayed_cue":
            # For DelayedCue, horizon ~= total length. total = delay + 2.
            # So delay = horizon - 2
            env_kwargs["delay_len"] = max(1, job["horizon"] - 2)
        else:
            env_kwargs["sequence_length"] = job["horizon"]

    env = EnvCls(
        batch_size=job["num_envs"],
        device=device,
        **env_kwargs,
    )

    obs_dim = getattr(env, "obs_dim")
    act_dim = getattr(env, "act_dim")

    # Some envs expose total_len; if present, we use it as episode length
    episode_len = getattr(env, "total_len", None)

    return env, obs_dim, act_dim, episode_len


def build_controller_from_config(cfg: Dict[str, Any],
                                 job: Dict[str, Any],
                                 obs_dim: int):
    """
    Build controller from cfg["controllers"][controller_name].

    Expects:
        module: "pseudo_mamba.controllers.gru"
        class: "GRUController" / "PseudoMambaController" / "MambaController"
        feature_dim / hidden_dim / d_model, etc. in cfg.
    """
    ctrl_name = job["controller"]
    ctrl_cfg = cfg["controllers"][ctrl_name]

    CtrlCls = load_class(ctrl_cfg["module"], ctrl_cfg["class"])

    # Common pattern in this repo:
    #   GRUController(obs_dim, hidden_dim, feature_dim)
    #   PseudoMambaController(obs_dim, hidden_dim, feature_dim, use_cuda_ext=...)
    #   MambaController(obs_dim, d_model, d_state, d_conv, feature_dim, ...)
    kwargs = dict(ctrl_cfg.get("kwargs", {}))

    # Try to be as generic as possible. You can tighten this if you want.
    if ctrl_name == "gru":
        controller = CtrlCls(
            input_dim=obs_dim,
            hidden_dim=ctrl_cfg["hidden_dim"],
            feature_dim=ctrl_cfg["feature_dim"],
            **kwargs,
        )
    elif ctrl_name == "pseudo_mamba":
        controller = CtrlCls(
            input_dim=obs_dim,
            hidden_dim=ctrl_cfg["hidden_dim"],
            feature_dim=ctrl_cfg["feature_dim"],
            use_cuda_ext=ctrl_cfg.get("use_cuda_ext", True),
            **kwargs,
        )
    elif ctrl_name == "mamba":
        # MambaController usually takes these Mamba-specific dims
        controller = CtrlCls(
            input_dim=obs_dim,
            hidden_dim=ctrl_cfg["d_model"], # MambaController uses hidden_dim as d_model
            feature_dim=ctrl_cfg["feature_dim"],
            d_state=ctrl_cfg["d_state"],
            d_conv=ctrl_cfg["d_conv"],
            require_patched_mamba=ctrl_cfg.get("require_patched_mamba", False),
            **kwargs,
        )
    elif ctrl_name == "transformer":
        controller = CtrlCls(
            input_dim=obs_dim,
            hidden_dim=ctrl_cfg["hidden_dim"],
            feature_dim=ctrl_cfg["feature_dim"],
            **kwargs,
        )
    else:
        # Fallback: pass input_dim plus any extra numeric fields
        controller = CtrlCls(
            input_dim=obs_dim,
            **kwargs,
        )

    return controller


# ============================================================
# Core training loop
# ============================================================

def train_single_run(
    cfg: Dict[str, Any],
    job: Dict[str, Any],
    run_dir: str,
    tags: List[str],
    resume_path: Optional[str] = None,
) -> None:
    """
    One full training run:
        - Build env + controller
        - Wrap in ActorCritic
        - Run recurrent PPO with full-sequence BPTT
        - Log & checkpoint to run_dir

    Required job fields (populated by CLI + config merger):
        job["id"]             e.g. "dc_h200/gru"
        job["task"]           task name (cfg.tasks key)
        job["controller"]     controller name (cfg.controllers key)
        job["num_envs"]
        job["steps"] or job["episodes"]
        job["seed"]
        Optional:
        job["device"], job["precision"], job["rollout_length"], lr, gamma, gae_lambda,
        clip_coef, vf_coef, ent_coef, max_grad_norm, ppo_epochs, num_minibatches
    """
    os.makedirs(run_dir, exist_ok=True)

    # ---------- Device & seed ----------
    device = select_device(job.get("device"))
    set_seed(job.get("seed", 1))

    # ---------- Build env ----------
    env, obs_dim, act_dim, episode_len = build_env_from_config(cfg, job, device)

    # Decide rollout length:
    #   Prefer explicit job["rollout_length"], else env.total_len if present, else defaults.
    defaults = cfg.get("defaults", {})
    rollout_length = job.get(
        "rollout_length",
        episode_len if episode_len is not None else defaults.get("rollout_length", 128),
    )

    num_envs = job["num_envs"]

    # ---------- Build controller + ActorCritic ----------
    controller = build_controller_from_config(cfg, job, obs_dim)
    actor_critic = ActorCritic(controller, act_dim).to(device)

    # ---------- PPO + buffer hyperparams ----------
    lr = job.get("lr", defaults.get("lr", 3e-4))
    gamma = job.get("gamma", defaults.get("gamma", 0.99))
    gae_lambda = job.get("gae_lambda", defaults.get("gae_lambda", 0.95))
    clip_coef = job.get("clip_coef", defaults.get("clip_coef", 0.2))
    vf_coef = job.get("vf_coef", defaults.get("vf_coef", 0.5))
    ent_coef = job.get("ent_coef", defaults.get("ent_coef", 0.0))
    max_grad_norm = job.get("max_grad_norm", defaults.get("max_grad_norm", 0.5))
    ppo_epochs = job.get("ppo_epochs", defaults.get("ppo_epochs", 4))
    num_minibatches = job.get("num_minibatches", defaults.get("num_minibatches", 4))

    ppo = PPO(
        actor_critic=actor_critic,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_param=clip_coef,
        entropy_coef=ent_coef,
        value_loss_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        num_minibatches=num_minibatches,
    )

    buffer = RolloutBuffer(
        num_steps=rollout_length,
        num_envs=num_envs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    # ---------- Scheduler setup ----------
    target_steps = job.get("steps")
    target_episodes = job.get("episodes")
    if target_steps is None and target_episodes is None:
        raise ValueError("Either job['steps'] or job['episodes'] must be set.")

    steps_per_update = rollout_length * num_envs

    if target_steps is None:
        target_steps = target_episodes * steps_per_update

    total_updates = math.ceil(target_steps / steps_per_update)
    ppo.set_scheduler(total_updates)

    # ---------- Optional resume ----------
    start_update = 0
    steps_done = 0
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")

    if resume_path is None and os.path.exists(ckpt_path):
        resume_path = ckpt_path

    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        actor_critic.load_state_dict(ckpt["actor_critic"])
        ppo.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ppo.scheduler is not None:
            ppo.scheduler.load_state_dict(ckpt["scheduler"])
        start_update = ckpt.get("update_idx", 0)
        steps_done = ckpt.get("steps_done", 0)
        print(f"[RESUME] Loaded checkpoint from {resume_path} at update {start_update}, steps {steps_done}")

    # ---------- WandB Setup ----------
    wandb_cfg = job.get("wandb", defaults.get("wandb", {}))
    use_wandb = False
    if wandb_cfg.get("mode", "disabled") != "disabled":
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "pseudo-mamba"),
                entity=wandb_cfg.get("entity"),
                config=job,
                name=job["id"],
                tags=tags,
                mode=wandb_cfg.get("mode"),
                reinit=True
            )
            use_wandb = True
        except ImportError:
            print("WandB not installed, skipping logging.")

    # ---------- Main training loop ----------
    obs = env.reset()  # [B, obs_dim]
    state = controller.init_state(batch_size=num_envs, device=device)

    for update_idx in range(start_update, total_updates):
        # Prepare rollout buffer for new episode/rollout
        buffer.step = 0
        buffer.obs[0].copy_(obs)
        buffer.states[0] = state
        buffer.dones[0].zero_()

        # Collect one rollout of length = rollout_length
        for t in range(rollout_length):
            with torch.no_grad():
                logits, value, next_state = actor_critic.forward_step(obs, state)
                dist = torch.distributions.Categorical(logits=logits)

                action = dist.sample()              # [B]
                logprob = dist.log_prob(action)     # [B]

            next_obs, reward, done, info = env.step(action)

            # Store step
            buffer.insert(
                obs=next_obs,                       # obs at t+1
                action=action,
                reward=reward,
                done=done.float(),
                value=value.squeeze(-1),            # [B]
                logprob=logprob,
                state=next_state,                   # state at t+1
            )

            # Recurrent state masking on done
            state = actor_critic.reset_mask(next_state, done)

            obs = next_obs
            steps_done += num_envs

        # Last value for GAE
        with torch.no_grad():
            _, last_value, _ = actor_critic.forward_step(obs, state)
            last_value = last_value.squeeze(-1)    # [B]

        buffer.compute_gae(last_value, gamma=gamma, gae_lambda=gae_lambda)

        # PPO update
        stats = ppo.update(buffer)

        # Scheduler step
        if ppo.scheduler is not None:
            ppo.scheduler.step()

        # ----- Logging -----
        avg_reward = buffer.rewards.sum() / num_envs
        
        if update_idx % 10 == 0:
            log_line = (
                f"[{job['id']}] update {update_idx+1}/{total_updates} "
                f"steps_done={steps_done} "
                f"return={avg_reward.item():.3f} "
                f"loss={stats.get('loss', 0):.4f} "
                f"pg_loss={stats.get('pg_loss', 0):.4f} "
                f"val_loss={stats.get('val_loss', 0):.4f} "
                f"entropy={stats.get('entropy', 0):.4f}"
            )
            print(log_line)

        if use_wandb:
            wandb.log({
                "reward": avg_reward.item(),
                "loss": stats.get("loss", 0),
                "pg_loss": stats.get("pg_loss", 0),
                "val_loss": stats.get("val_loss", 0),
                "entropy": stats.get("entropy", 0),
                "lr": stats.get("lr", 0),
                "grad_norm": stats.get("grad_norm", 0),
                "explained_var": stats.get("explained_var", 0),
                "steps": steps_done
            }, step=update_idx)

        # ----- Checkpoint -----
        if update_idx % 50 == 0 or update_idx == total_updates - 1:
            torch.save(
                {
                    "actor_critic": actor_critic.state_dict(),
                    "optimizer": ppo.optimizer.state_dict(),
                    "scheduler": ppo.scheduler.state_dict() if ppo.scheduler is not None else None,
                    "update_idx": update_idx + 1,
                    "steps_done": steps_done,
                    "job": job,
                    "tags": tags,
                },
                ckpt_path,
            )

    print(f"[DONE] {job['id']} finished at {steps_done} steps; checkpoint at {ckpt_path}")
    
    if use_wandb:
        wandb.finish()
