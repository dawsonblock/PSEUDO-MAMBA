#!/usr/bin/env python3
import argparse
import os
import yaml
import itertools
from typing import Dict, Any, List

import torch

from pseudo_mamba_rl_core import train_single_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pseudo-Mamba Memory Benchmark Suite"
    )
    p.add_argument("--config", type=str, default="pseudo_mamba_memory_suite.yaml")

    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--suite", type=str, help="Suite name from config.suites")
    mode.add_argument("--task", type=str, help="Single task id (config.tasks key)")

    p.add_argument("--controller", type=str, help="Controller id (config.controllers key)")

    # Env / difficulty
    p.add_argument("--horizon", type=int)
    p.add_argument("--num-envs", type=int)

    # Training
    p.add_argument("--steps", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--seed", type=int, default=1)

    # Logging
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str)
    p.add_argument("--tag", action="append", default=[])

    # Compute
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--precision", type=str, choices=["fp32", "amp"])

    # Behaviour
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=str)
    p.add_argument("--overwrite", action="store_true")

    # WandB
    p.add_argument("--wandb-mode", type=str, choices=["disabled", "online", "offline"])
    p.add_argument("--wandb-project", type=str)

    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        # Fallback to looking in package dir if installed?
        # For now, assume local or relative path
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, "r") as f:
        return yaml.safe_load(f)


def expand_suite(cfg: Dict[str, Any], suite_name: str) -> List[Dict[str, Any]]:
    if suite_name not in cfg["suites"]:
        raise ValueError(f"Suite '{suite_name}' not found in config. Available: {list(cfg['suites'].keys())}")
        
    suite = cfg["suites"][suite_name]
    defaults = cfg.get("defaults", {})

    jobs = []
    for run_def in suite["runs"]:
        task_name = run_def["task"]
        # If run_def has 'controllers', expand over them. Else use all defined in config?
        # The user's spec says: "controllers: [gru, pseudo_mamba, mamba]" in run_def
        controllers = run_def.get("controllers", list(cfg["controllers"].keys()))
        
        for ctrl_name in controllers:
            job = {
                "id": f"{run_def['id']}/{ctrl_name}",
                "task": task_name,
                "controller": ctrl_name,
            }
            job.update(defaults)
            job.update(run_def)  # horizon, steps, etc. overrides defaults
            
            # Remove 'controllers' list from job dict as it's now resolved
            if "controllers" in job:
                del job["controllers"]
                
            jobs.append(job)
    return jobs


def build_single_job(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if not args.task or not args.controller:
        raise ValueError("Single-run mode requires --task and --controller.")

    defaults = cfg.get("defaults", {})
    job = {
        "id": f"{args.task}/{args.controller}",
        "task": args.task,
        "controller": args.controller,
    }
    job.update(defaults)

    # CLI overrides
    if args.horizon is not None:
        job["horizon"] = args.horizon
    if args.steps is not None:
        job["steps"] = args.steps
    if args.episodes is not None:
        job["episodes"] = args.episodes
    if args.num_envs is not None:
        job["num_envs"] = args.num_envs

    return job


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.suite:
        jobs = expand_suite(cfg, args.suite)
    else:
        jobs = [build_single_job(cfg, args)]

    # Global overrides from CLI
    for job in jobs:
        if args.device:
            job["device"] = args.device
        if args.precision:
            job["precision"] = args.precision
        if args.wandb_mode:
            job.setdefault("wandb", {})
            job["wandb"]["mode"] = args.wandb_mode
        if args.wandb_project:
            job.setdefault("wandb", {})
            job["wandb"]["project"] = args.wandb_project

    if args.dry_run:
        print(f"Loaded config from {args.config}")
        print(f"Found {len(jobs)} jobs:")
        for j in jobs:
            print(f"  - {j['id']}: task={j['task']}, ctrl={j['controller']}, "
                  f"horizon={j.get('horizon')}, steps={j.get('steps')}")
        return

    for idx, job in enumerate(jobs):
        seed = (args.seed or 1) + idx
        job["seed"] = seed

        run_name = args.run_name or job["id"]
        run_dir = os.path.join(args.log_dir, run_name)
        
        if os.path.exists(run_dir) and not args.overwrite and not args.resume:
             # If resume is set, we allow existing dir. If overwrite is set, we allow it.
             # If neither, we error to prevent accidental loss.
             raise FileExistsError(f"Run dir {run_dir} exists (use --overwrite or --resume).")

        print(f"\n[{idx+1}/{len(jobs)}] Running {job['id']} → {run_dir}")

        try:
            train_single_run(
                cfg=cfg,
                job=job,
                run_dir=run_dir,
                tags=args.tag,
                resume_path=args.resume,
            )
        except Exception as e:
            print(f"FAILED job {job['id']}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
