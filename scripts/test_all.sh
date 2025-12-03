#!/bin/bash
set -e

echo "Running Pseudo-Mamba v2.0 Test Suite"
echo "===================================="

# 1. Install package
echo "Installing package..."
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -e . --no-build-isolation

# 2. Run CUDA Correctness Test (if GPU available)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "Running CUDA Correctness Test..."
    python pseudo_mamba/kernels/pseudo_mamba/test_correctness.py
else
    echo "Skipping CUDA tests (no GPU detected)"
fi

# 3. Run Short Benchmark (Integration Test)
echo "Running Integration Test (Short Benchmark)..."
python -m pseudo_mamba.benchmarks.benchmark_runner \
    --task delayed_cue \
    --controller gru \
    --num_envs 16 \
    --rollout_steps 32 \
    --total_updates 2 \
    --hidden_dim 64

echo "All tests passed!"
