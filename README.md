# Pseudo-Mamba v2.0: The Universal Memory Benchmark Suite

**Production-grade RL testbed for long-horizon memory in Recurrent Neural Networks and State Space Models.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://github.com/dawsonblock/PSEUDO-MAMBA/actions/workflows/tests.yml/badge.svg)](https://github.com/dawsonblock/PSEUDO-MAMBA/actions)
[![Status](https://img.shields.io/badge/status-production-green)](https://github.com/dawsonblock/PSEUDO-MAMBA)

---

## 🚀 Overview

Pseudo-Mamba v2.0 is a **complete production system** for benchmarking memory capabilities in sequence models. It provides a unified CLI, standardized task registry, and comprehensive testing infrastructure to compare **GRU**, **Mamba**, **Transformer**, and **Pseudo-Mamba** controllers across 8 vectorized memory environments.

### What Makes This Different?

✅ **Unified CLI**: Single `pseudo-mamba-memory-suite` command for all benchmarks  
✅ **Standardized Registry**: 8 tasks × 4 difficulty levels with target success rates  
✅ **Production PPO**: Mathematically correct recurrent RL with full-sequence BPTT  
✅ **CI/CD Pipeline**: Automated testing on Python 3.9, 3.10, 3.11  
✅ **Reference Baselines**: Pre-computed metrics for validation  
✅ **YAML Configuration**: Reproducible experiment definitions  

---

## 📦 Installation

```bash
git clone https://github.com/dawsonblock/PSEUDO-MAMBA.git
cd PSEUDO-MAMBA
pip install -e .
```

**Optional Dependencies:**
```bash
# For Mamba controller (requires patched mamba-ssm)
pip install -e .[mamba]

# For development and testing
pip install -e .[dev]
```

> **Note**: Mac/CPU systems will automatically use PyTorch fallbacks. CUDA kernels are optional.

---

## ⚡ Quick Start

### 1. Verify Installation (Golden Run)

Run this 5-minute reference benchmark to validate your setup:

```bash
python pseudo_mamba_memory_suite.py \
  --task delayed_cue \
  --controller gru \
  --horizon 200 \
  --steps 200000
```

**Expected Results** (CPU):
- **Final Return**: `0.85 ± 0.05`
- **Wall Time**: `~300s`

### 2. Run a Full Benchmark Suite

Use the new YAML-based CLI to run standardized benchmarks:

```bash
# Dry-run to see what will execute
python pseudo_mamba_memory_suite.py --suite basic_v1 --dry-run

# Run the full suite (12 experiments: 3 tasks × 3 controllers × 2 horizons)
python pseudo_mamba_memory_suite.py --suite basic_v1 --log-dir runs/
```

### 3. Compare to Reference Baseline

```bash
python scripts/compare_to_baseline.py --results results/benchmark_summary.json
```

### 4. Visualize Results

```bash
python scripts/plot_results.py --json results/benchmark_summary.json --out plots/
```

---

## 🧠 Supported Controllers

| Controller | Description | State Management |
|:-----------|:------------|:----------------|
| **GRU** | Standard Gated Recurrent Unit | Native PyTorch `GRUCell` |
| **Mamba** | Selective State Space Model from `mamba_ssm` | Patched for explicit state extraction |
| **Pseudo-Mamba** | Toy SSM (`y = tanh(x + h)`) | Custom CUDA kernel + PyTorch fallback |
| **Transformer** | Causal Self-Attention with KV-cache | Full history buffer |

All controllers implement the **unified `BaseController` interface**:
```python
controller.init_state(batch_size, device)  # → state
controller.forward_step(obs, state)         # → (features, new_state)
controller.reset_mask(state, done_mask)    # → reset_state
```

---

## 🧪 Neural Memory Suite

### 8 Vectorized Tasks

| Task | Description | Key Challenge |
|:-----|:------------|:--------------|
| **Delayed Cue** | Remember a one-hot cue after delay | Recall over long horizon |
| **Copy Memory** | Reproduce a token sequence | Faithful sequence reproduction |
| **Associative Recall** | Key-value retrieval with distractors | Binding & selective recall |
| **N-Back** | Continuous working memory | Rolling buffer management |
| **Multi-Cue Delay** | Multiple cues at random times | Order & context tracking |
| **Permuted Copy** | Reproduce sequence in permuted order | Order-insensitive recall |
| **Pattern Binding** | Bind and recall pattern sequences | Compositional memory |
| **Distractor Navigation** | Navigate while ignoring noise | Selective attention |

### 4 Difficulty Levels

Use the **Benchmark Registry** to access standardized configs:

```python
from pseudo_mamba.benchmarks.registry import get_task_config

# Get "medium" difficulty config for delayed_cue
config = get_task_config('delayed_cue', difficulty='medium')
# → {'horizon': 200, 'num_cues': 8, 'target_success': 0.90, ...}
```

| Difficulty | Horizon Range | Description | Updates |
|:-----------|:--------------|:------------|:--------|
| **Easy** | 50-200 | Quick sanity checks / CI | 100 |
| **Medium** | 200-1K | Standard benchmarking | 1000 |
| **Hard** | 1K-5K | Long-horizon stress tests | 2000 |
| **Extreme** | 5K-20K | Ultimate memory challenge | 5000 |

---

## 🛠️ CLI Reference

### Single-Task Mode

```bash
python pseudo_mamba_memory_suite.py \
  --task delayed_cue \
  --controller gru \
  --horizon 200 \
  --steps 200000 \
  --num-envs 64 \
  --log-dir runs/my_experiment
```

### Suite Mode (YAML-Driven)

Define custom suites in `pseudo_mamba_memory_suite.yaml`:

```yaml
suites:
  my_custom_suite:
    runs:
      - id: short_test
        task: delayed_cue
        controllers: [gru, transformer]
        horizon: 100
        steps: 50000
```

Then run:
```bash
python pseudo_mamba_memory_suite.py --suite my_custom_suite
```

### Advanced Options

```bash
--wandb-mode online              # Enable WandB logging
--device cuda:0                  # Specify GPU
--resume runs/checkpoint.pt      # Resume from checkpoint
--mamba_d_state 32               # Configure Mamba SSM dimension
--transformer_n_head 8           # Configure Transformer heads
```

---

## 📊 Benchmark Registry

All tasks are registered with **standardized configurations** in `pseudo_mamba/benchmarks/registry.py`:

```python
from pseudo_mamba.benchmarks.registry import get_benchmark_suite

# Get all "medium" difficulty tasks
suite = get_benchmark_suite(difficulty='medium')
# → [{'task': 'delayed_cue', 'horizon': 200, ...}, ...]
```

Print the full registry:
```bash
python -m pseudo_mamba.benchmarks.registry
```

---

## ✅ Testing & CI

### Run Tests Locally

```bash
# CPU tests (no CUDA required)
pytest tests/test_autograd_equivalence.py::test_pseudo_mamba_cpu_fallback -v

# Smoke test
python pseudo_mamba_memory_suite.py --suite tiny_sanity --overwrite
```

### GitHub Actions CI

On every push/PR:
- Syntax checks across Python 3.9, 3.10, 3.11
- Unit tests for autograd equivalence
- Smoke test via `tiny_sanity` suite

See [`.github/workflows/tests.yml`](.github/workflows/tests.yml) for details.

---

## 📚 Documentation

- **[System Overview](docs/system_overview.md)** – Architecture & design philosophy
- **[State Management Guide](docs/state_management.md)** – Mamba patches & RL state handling
- **[Benchmark Details](docs/benchmarks.md)** – Task specs & evaluation metrics
- **[Benchmark Registry](pseudo_mamba/benchmarks/registry.py)** – Standardized configs

---

## 🏗️ Project Structure

```
pseudo_mamba/
├── controllers/          # BaseController + GRU/Mamba/Pseudo-Mamba/Transformer
├── envs/                 # 8 vectorized memory tasks
├── rlh/                  # PPO engine, rollout buffers, ActorCritic
├── kernels/              # CUDA extensions (Pseudo-Mamba)
└── benchmarks/           # Registry, runner, analysis tools

scripts/
├── plot_results.py       # Visualization
├── compare_to_baseline.py # Baseline comparison
└── test_all.sh           # Integration tests

results/
└── reference_metrics.json # Reference baseline for validation

tests/
└── test_autograd_equivalence.py # CUDA kernel correctness tests
```

---

## 🔬 Advanced Usage

### Custom Controllers

Implement the `BaseController` interface:

```python
from pseudo_mamba.controllers.base import BaseController

class MyController(BaseController):
    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def forward_step(self, obs, state):
        # Your recurrent logic here
        features = self.my_rnn(obs, state)
        return features, new_state
    
    def reset_mask(self, state, done_mask):
        return state * (1 - done_mask.float()).view(-1, 1)
```

Register in `pseudo_mamba_memory_suite.yaml` and run.

### WandB Integration

```bash
python pseudo_mamba_memory_suite.py \
  --suite basic_v1 \
  --wandb-mode online \
  --wandb-project my-memory-experiments
```

---

## 📈 Reference Metrics

See [`results/reference_metrics.json`](results/reference_metrics.json) for baseline results.

Example (CPU, medium difficulty):
- `delayed_cue` @ H=200, GRU: **0.87 ± 0.03**
- `copy_memory` @ H=200, GRU: **0.82 ± 0.04**

GPU results will be faster and may achieve higher returns.

---

## 🤝 Contributing

We welcome contributions! Areas of interest:
1. **New Controllers** (e.g., RWKV, Hyena)
2. **New Memory Tasks** (e.g., graph recall, multi-hop QA)
3. **Performance Optimizations** (kernel improvements, distributed training)
4. **Reference Results** (submit your benchmark metrics via PR)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

Apache 2.0 – See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built on [Mamba](https://github.com/state-spaces/mamba) by Tri Dao and Albert Gu
- PPO implementation inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl)
- Memory tasks adapted from classic cognitive neuroscience paradigms

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/dawsonblock/PSEUDO-MAMBA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dawsonblock/PSEUDO-MAMBA/discussions)
- **Author**: Dawson Block

---

**Star ⭐ this repo if you find it useful for your research!**
