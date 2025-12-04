"""
Benchmark Registry: Standard task configurations and difficulty levels.

This module defines standardized configurations for all memory tasks,
organized by difficulty level to enable reproducible benchmarking.
"""

from typing import Dict, Any, List

# Task difficulty levels
DIFFICULTY_LEVELS = {
    "easy": {
        "horizon_range": (50, 200),
        "description": "Quick sanity checks / CI tests",
        "target_updates": 100,
    },
    "medium": {
        "horizon_range": (200, 1000),
        "description": "Standard benchmarking",
        "target_updates": 1000,
    },
    "hard": {
        "horizon_range": (1000, 5000),
        "description": "Long-horizon stress tests",
        "target_updates": 2000,
    },
    "extreme": {
        "horizon_range": (5000, 20000),
        "description": "Ultimate memory stress",
        "target_updates": 5000,
    }
}

# Standard task configs
TASK_REGISTRY = {
    "delayed_cue": {
        "description": "Present a cue, long delay, then query. Agent must recall the cue.",
        "difficulty_configs": {
            "easy": {"horizon": 50, "delay_len": 48, "num_cues": 4, "target_success": 0.95},
            "medium": {"horizon": 200, "delay_len": 198, "num_cues": 8, "target_success": 0.90},
            "hard": {"horizon": 1000, "delay_len": 998, "num_cues": 8, "target_success": 0.80},
            "extreme": {"horizon": 5000, "delay_len": 4998, "num_cues": 16, "target_success": 0.70}
        }
    },
    "copy_memory": {
        "description": "Copy a sequence of tokens after a delay.",
        "difficulty_configs": {
            "easy": {"horizon": 50, "seq_len": 10, "target_success": 0.90},
            "medium": {"horizon": 200, "seq_len": 20, "target_success": 0.85},
            "hard": {"horizon": 500, "seq_len": 30, "target_success": 0.75},
            "extreme": {"horizon": 2000, "seq_len": 50, "target_success": 0.60}
        }
    },
    "assoc_recall": {
        "description": "Key-value associative recall with distractors.",
        "difficulty_configs": {
            "easy": {"horizon": 50, "num_pairs": 4, "target_success": 0.90},
            "medium": {"horizon": 200, "num_pairs": 8, "target_success": 0.85},
            "hard": {"horizon": 800, "num_pairs": 12, "target_success": 0.75},
            "extreme": {"horizon": 3000, "num_pairs": 20, "target_success": 0.65}
        }
    },
    "n_back": {
        "description": "Continuous N-back working memory task.",
        "difficulty_configs": {
            "easy": {"horizon": 100, "n": 2, "target_success": 0.85},
            "medium": {"horizon": 300, "n": 3, "target_success": 0.80},
            "hard": {"horizon": 1000, "n": 4, "target_success": 0.70},
            "extreme": {"horizon": 5000, "n": 5, "target_success": 0.60}
        }
    },
    "multi_cue_delay": {
        "description": "Remember multiple cues presented at random times.",
        "difficulty_configs": {
            "easy": {"horizon": 100, "num_cues": 2, "target_success": 0.85},
            "medium": {"horizon": 300, "num_cues": 4, "target_success": 0.80},
            "hard": {"horizon": 1000, "num_cues": 6, "target_success": 0.70},
            "extreme": {"horizon": 5000, "num_cues": 10, "target_success": 0.60}
        }
    },
    "permuted_copy": {
        "description": "Reproduce a sequence in a permuted order.",
        "difficulty_configs": {
            "easy": {"horizon": 50, "seq_len": 8, "target_success": 0.85},
            "medium": {"horizon": 200, "seq_len": 15, "target_success": 0.75},
            "hard": {"horizon": 500, "seq_len": 25, "target_success": 0.65},
            "extreme": {"horizon": 2000, "seq_len": 40, "target_success": 0.50}
        }
    },
    "pattern_binding": {
        "description": "Bind and recall pattern sequences.",
        "difficulty_configs": {
            "easy": {"horizon": 100, "num_patterns": 3, "target_success": 0.85},
            "medium": {"horizon": 300, "num_patterns": 5, "target_success": 0.75},
            "hard": {"horizon": 1000, "num_patterns": 8, "target_success": 0.65},
            "extreme": {"horizon": 5000, "num_patterns": 12, "target_success": 0.55}
        }
    },
    "distractor_nav": {
        "description": "Navigate to a target while ignoring distractors.",
        "difficulty_configs": {
            "easy": {"horizon": 50, "num_distractors": 2, "target_success": 0.90},
            "medium": {"horizon": 200, "num_distractors": 5, "target_success": 0.80},
            "hard": {"horizon": 500, "num_distractors": 10, "target_success": 0.70},
            "extreme": {"horizon": 2000, "num_distractors": 20, "target_success": 0.60}
        }
    }
}


def get_task_config(task_name: str, difficulty: str = "medium") -> Dict[str, Any]:
    """
    Get standardized config for a task at a difficulty level.
    
    Args:
        task_name: Name of the task (e.g., "delayed_cue")
        difficulty: Difficulty level ("easy", "medium", "hard", "extreme")
    
    Returns:
        Dictionary with task configuration including horizon and target metrics
    
    Raises:
        ValueError: If task_name or difficulty is invalid
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"Unknown difficulty: {difficulty}. "
            f"Available difficulties: {list(DIFFICULTY_LEVELS.keys())}"
        )
    
    config = TASK_REGISTRY[task_name]["difficulty_configs"][difficulty].copy()
    config["task"] = task_name
    config["difficulty"] = difficulty
    config["description"] = TASK_REGISTRY[task_name]["description"]
    
    # Add standard training params from difficulty level
    config["total_updates"] = DIFFICULTY_LEVELS[difficulty]["target_updates"]
    
    return config


def get_benchmark_suite(difficulty: str = "medium", tasks: List[str] = None) -> List[Dict[str, Any]]:
    """
    Get a full benchmark suite at a given difficulty.
    
    Args:
        difficulty: Difficulty level for all tasks
        tasks: Optional list of specific tasks to include. If None, includes all tasks.
    
    Returns:
        List of task configurations
    """
    if tasks is None:
        tasks = list(TASK_REGISTRY.keys())
    
    return [
        get_task_config(task_name, difficulty)
        for task_name in tasks
        if task_name in TASK_REGISTRY
    ]


def list_tasks() -> List[str]:
    """Return list of all available task names."""
    return list(TASK_REGISTRY.keys())


def list_difficulties() -> List[str]:
    """Return list of all difficulty levels."""
    return list(DIFFICULTY_LEVELS.keys())


def print_registry_summary():
    """Print a formatted summary of the benchmark registry."""
    print("=" * 80)
    print("PSEUDO-MAMBA BENCHMARK REGISTRY")
    print("=" * 80)
    
    print("\nDifficulty Levels:")
    for difficulty, info in DIFFICULTY_LEVELS.items():
        horizon_min, horizon_max = info["horizon_range"]
        print(f"  {difficulty:10s} - {info['description']:40s} H={horizon_min}-{horizon_max}")
    
    print("\nRegistered Tasks:")
    for task_name, task_info in TASK_REGISTRY.items():
        print(f"\n  {task_name}:")
        print(f"    {task_info['description']}")
        print(f"    Difficulty configs: {list(task_info['difficulty_configs'].keys())}")


if __name__ == "__main__":
    print_registry_summary()
