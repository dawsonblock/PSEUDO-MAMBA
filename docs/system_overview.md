# Pseudo-Mamba v2.0 System Overview

## Architecture

The Pseudo-Mamba system is designed for rigorous evaluation of long-horizon memory capabilities in recurrent neural networks and state space models.

### System Architecture

```mermaid
graph TD
    subgraph "Environments (envs/)"
        E[VectorizedEnv] --> T1[CopyMemory]
        E --> T2[AssocRecall]
        E --> T3[DelayedCue]
        E --> T4[...5 others]
    end

    subgraph "Controllers (controllers/)"
        C[BaseController] --> G[GRU]
        C --> M[Mamba]
        C --> P[PseudoMamba]
        P -.->|Optional| K[CUDA Kernel]
    end

    subgraph "RL Engine (rlh/)"
        AC[ActorCritic] --> C
        PPO[PPO Engine] --> AC
        PPO --> RB[RolloutBuffer]
    end

    PPO <-->|Step & Train| E
```

### Core Components

1.  **Controllers (`pseudo_mamba.controllers`)**:
    *   **Unified API**: All controllers inherit from `BaseController` and expose a consistent `forward_step(x, state) -> (features, new_state)` interface.
    *   **Implementations**:
        *   `GRUController`: Standard Gated Recurrent Unit.
        *   `MambaController`: Wrapper around `mamba_ssm` with explicit state management.
        *   `PseudoMambaController`: Minimal SSM implementation using a custom CUDA kernel (`y = tanh(x + h)`).

2.  **Environments (`pseudo_mamba.envs`)**:
    *   **Vectorized Suite**: 8 memory-intensive tasks implemented in PyTorch for massive parallelism.
    *   **Tasks**: `DelayedCue`, `CopyMemory`, `AssocRecall`, `NBack`, `MultiCueDelay`, `PermutedCopy`, `PatternBinding`, `DistractorNav`.

3.  **RL Engine (`pseudo_mamba.rlh`)**:
    *   **Honest PPO**: Implements Proximal Policy Optimization with full-sequence Backpropagation Through Time (BPTT).
    *   **Recurrent Support**: Correctly handles hidden state preservation and masking across rollout chunks.

4.  **Kernels (`pseudo_mamba.kernels`)**:
    *   **CUDA Extension**: Custom C++/CUDA kernels for efficient SSM operations.
    *   **Scaffold**: Robust setup scripts and correctness tests.

## Usage

### Running Benchmarks

```bash
python -m pseudo_mamba.benchmarks.benchmark_runner \
    --task copy_memory \
    --controller mamba \
    --num_envs 64 \
    --total_updates 1000
```

### Running Tests

```bash
bash scripts/test_all.sh
```
