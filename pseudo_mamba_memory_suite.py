#!/usr/bin/env python3
"""
pseudo_mamba_memory_suite.py

Canonical config-driven benchmark suite for long-horizon memory tasks.

Features:
    - Runs GRU / Mamba / Pseudo-Mamba across multiple memory tasks.
    - Uses a YAML/JSON config to define experiments, horizons, and controllers.
    - Wraps the existing `train()` function in pseudo_mamba.benchmarks.pseudo_mamba_benchmark.
    - Produces a single JSON summary file with all runs.

Usage:
    python pseudo_mamba_memory_suite.py --config configs/memory_suite.yaml
"""

import argparse
import json
import os
import time
import random
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from pseudo_mamba.benchmarks.pseudo_mamba_benchmark import train as benchmark_train


def load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in [".yml", ".yaml"]:
            if not HAS_YAML:
                raise ImportError("pyyaml is required for YAML configs. Install with `pip install pyyaml`.")
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {ext}. Use .yaml/.yml or .json.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-Mamba Memory Suite")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/memory_suite.yaml",
        help="Path to YAML/JSON config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output JSON path from config.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional subset of experiment IDs to run.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=None,
        help="Override controllers for all experiments.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds list for all experiments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved run plan and exit without training.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Force-disable Weights & Biases logging.",
    )
    return parser.parse_args()


def make_run_plan(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    defaults = cfg.get("defaults", {})
    suite_name = cfg.get("suite_name", "pseudo_mamba_memory_suite")
    base_device = args.device or cfg.get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    base_out = args.out or cfg.get("out") or f"results/{suite_name}.json"

    # Global controllers + seeds from config, overridden by CLI
    default_controllers = cfg.get("default_controllers", ["gru", "mamba", "pseudo_mamba"])
    global_controllers = args.controllers or default_controllers
    global_seeds = args.seeds or cfg.get("seeds", [0])

    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False)) and not args.no_wandb

    all_experiments = cfg.get("experiments", [])
    if not all_experiments:
        raise ValueError("Config must define at least one experiment under 'experiments'.")

    # Filter experiments by ID if requested
    if args.experiments is not None:
        exp_ids = set(args.experiments)
        experiments = [e for e in all_experiments if e.get("id") in exp_ids]
        if not experiments:
            raise ValueError(f"No experiments found matching IDs: {sorted(exp_ids)}")
    else:
        experiments = all_experiments

    run_entries: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_id = exp.get("id")
        if not exp_id:
            raise ValueError("Each experiment must have an 'id' field.")

        env_name = exp["task"]
        horizon = int(exp.get("horizon", 0))
        exp_controllers = exp.get("controllers", global_controllers)
        exp_overrides = exp.get("overrides", {})
        env_kwargs = exp.get("env_kwargs", {})

        for seed in global_seeds:
            for controller in exp_controllers:
                entry = {
                    "suite": suite_name,
                    "experiment_id": exp_id,
                    "task": env_name,
                    "controller": controller,
                    "horizon": horizon,
                    "seed": seed,
                    "device": base_device,
                    "env_kwargs": env_kwargs,
                    # Hyperparams resolved lazily at run time
                    "hyperparams": {
                        **defaults,
                        **exp_overrides,
                    },
                    "wandb": {
                        "enabled": wandb_enabled,
                        **wandb_cfg,
                    },
                }
                run_entries.append(entry)

    return {
        "suite_name": suite_name,
        "device": base_device,
        "out": base_out,
        "runs": run_entries,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    plan = make_run_plan(cfg, args)

    runs = plan["runs"]
    out_path = plan["out"]
    device = torch.device(plan["device"])

    print(f"[SUITE] {plan['suite_name']}")
    print(f"[DEVICE] {device}")
    print(f"[OUT] {out_path}")
    print(f"[NUM RUNS] {len(runs)}")

    if args.dry_run:
        for r in runs:
            print(
                f"  - id={r['experiment_id']} task={r['task']} "
                f"ctrl={r['controller']} horizon={r['horizon']} seed={r['seed']}"
            )
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results: List[Dict[str, Any]] = []
    start_suite = time.time()

    for idx, run in enumerate(runs, start=1):
        exp_id = run["experiment_id"]
        task = run["task"]
        controller = run["controller"]
        horizon = run["horizon"]
        seed = int(run["seed"])
        hyper = run["hyperparams"]
        wandb_cfg = run["wandb"]

        print(
            f"\n[RUN {idx}/{len(runs)}] "
            f"exp={exp_id} task={task} ctrl={controller} "
            f"hor={horizon} seed={seed}"
        )

        set_seed(seed)

        # Resolve hyperparams with sane defaults
        num_envs = int(hyper.get("num_envs", 64))
        hidden_dim = int(hyper.get("hidden_dim", 128))
        total_updates = int(hyper.get("total_updates", 5000))
        gamma = float(hyper.get("gamma", 0.99))
        gae_lambda = float(hyper.get("gae_lambda", 0.95))
        lr = float(hyper.get("lr", 3e-4))
        value_coef = float(hyper.get("value_coef", 0.5))
        entropy_coef = float(hyper.get("entropy_coef", 0.01))
        mamba_d_state = int(hyper.get("mamba_d_state", 16))
        mamba_d_conv = int(hyper.get("mamba_d_conv", 4))
        mamba_expand = int(hyper.get("mamba_expand", 2))
        transformer_n_head = int(hyper.get("transformer_n_head", 4))
        transformer_n_layer = int(hyper.get("transformer_n_layer", 2))

        use_wandb = bool(wandb_cfg.get("enabled", False))

        t0 = time.time()
        try:
            train_result = benchmark_train(
                env_name=task,
                controller=controller,
                device=device,
                horizon=horizon,
                num_envs=num_envs,
                hidden_dim=hidden_dim,
                total_updates=total_updates,
                gamma=gamma,
                gae_lambda=gae_lambda,
                lr=lr,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                use_wandb=use_wandb,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                transformer_n_head=transformer_n_head,
                transformer_n_layer=transformer_n_layer,
                env_kwargs=run["env_kwargs"],
            )
        except Exception as e:
            duration = time.time() - t0
            print(f"[ERROR] run failed: {e}")
            import traceback
            traceback.print_exc()

            results.append(
                {
                    "status": "error",
                    "error": str(e),
                    "experiment_id": exp_id,
                    "task": task,
                    "controller": controller,
                    "horizon": horizon,
                    "seed": seed,
                    "duration_sec": duration,
                    "hyperparams": hyper,
                    "env_kwargs": run["env_kwargs"],
                }
            )
            continue

        duration = time.time() - t0
        merged = {
            "status": "ok",
            "experiment_id": exp_id,
            "task": task,
            "controller": controller,
            "horizon": horizon,
            "seed": seed,
            "duration_sec": duration,
            "device": str(device),
            "hyperparams": hyper,
            "env_kwargs": run["env_kwargs"],
        }
        # Merge whatever train() returned (final_avg_return, etc.)
        if isinstance(train_result, dict):
            merged.update(train_result)

        results.append(merged)

        # Incremental write to avoid losing everything on crash
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[RUN DONE] final_avg_return={merged.get('final_avg_return', 'NA')}")

    total_duration = time.time() - start_suite
    print(f"\n[SUITE COMPLETE] runs={len(results)} total_time_sec={total_duration:.1f}")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
