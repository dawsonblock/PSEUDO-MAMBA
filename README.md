# Pseudo-Mamba v2.0: Neural Memory Benchmark Suite

**A unified, production-grade system for benchmarking long-horizon memory in Recurrent Neural Networks and State Space Models.**

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-stable-green)

## 🚀 Overview

Pseudo-Mamba v2.0 is a complete overhaul of the Mamba repository, transforming it into a rigorous testbed for memory capabilities. It provides a unified interface for comparing **GRU**, **Mamba**, and **Pseudo-Mamba** (custom CUDA kernel) controllers across a suite of 8 vectorized memory tasks.

### Key Features

*   **Unified Controller API**: Plug-and-play interface for GRU, Mamba, and custom kernels.
*   **Neural Memory Suite v1.1**: 8 vectorized, GPU-accelerated tasks designed to stress-test recall, pattern binding, and navigation.
*   **Honest PPO Engine**: Mathematically correct Proximal Policy Optimization with full-sequence Backpropagation Through Time (BPTT) and proper recurrent state handling.
*   **CUDA Kernels**: Custom C++/CUDA extensions for efficient SSM operations (with CPU fallbacks).
*   **Reproducible Benchmarks**: Single-command runner to generate comparable results across models.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/dawsonblock/PSEUDO-MAMBA.git
cd PSEUDO-MAMBA

# Install in editable mode
pip install -e .
```

*Note: For Mac users or systems without CUDA, the installation will automatically skip the CUDA kernels and use PyTorch fallbacks.*

## ⚡ Quick Start

Run a quick benchmark to verify your installation:

```bash
# Run a short integration test
bash scripts/test_all.sh
```

Run a full training session on the **Copy Memory** task with a **Mamba** controller:

```bash
python -m pseudo_mamba.benchmarks.benchmark_runner \
    --task copy_memory \
    --controller mamba \
    --num_envs 64 \
    --total_updates 1000
```

## 🧠 Supported Controllers

| Controller | Description |
| :--- | :--- |
| **GRU** | Standard Gated Recurrent Unit (Baseline). |
| **Mamba** | Official `mamba_ssm` implementation with patched state management. |
| **Pseudo-Mamba** | Custom "Toy" SSM (`y = tanh(x + h)`) using a dedicated CUDA kernel. |

## 🧪 Memory Tasks

1.  **Delayed Cue**: Recall a cue after a long delay.
2.  **Copy Memory**: Reproduce a sequence of tokens.
3.  **Associative Recall**: Key-Value retrieval.
4.  **N-Back**: Continuous working memory task.
5.  **Multi-Cue Delay**: Remember multiple cues presented at random times.
6.  **Permuted Copy**: Reproduce a sequence in a permuted order.
7.  **Pattern Binding**: Bind and recall sequences.
8.  **Distractor Navigation**: Navigate to a target while ignoring distractors.

## 📚 Documentation

*   [System Overview](docs/system_overview.md)
*   [State Management Guide](docs/state_management.md)
*   [Benchmark Details](docs/benchmarks.md)

## 🛠️ Project Structure

```
pseudo_mamba/
├── controllers/       # Unified Controller API (GRU, Mamba, Pseudo-Mamba)
├── envs/              # Vectorized Memory Environments
├── rlh/               # PPO Engine & Rollout Buffers
├── kernels/           # CUDA Extensions
└── benchmarks/        # Runner & Analysis Tools
```

## 📄 License

Apache 2.0
