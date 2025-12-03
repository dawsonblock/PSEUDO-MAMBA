#!/bin/bash
# GPU Build and Verification Script for Mamba State Management
# Run this on a Linux machine with CUDA GPU

set -e  # Exit on error

echo "============================================================="
echo "Mamba State Management - GPU Build & Verification"
echo "============================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check requirements
echo "Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗ CUDA not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}✓ CUDA ${CUDA_VERSION} found${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# Check PyTorch with CUDA
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓ PyTorch ${TORCH_VERSION} with CUDA support${NC}"
else
    echo -e "${RED}✗ PyTorch with CUDA required. Install with:${NC}"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

echo ""
echo "============================================================="
echo "Phase 1: Clean Build"
echo "============================================================="

# Set build parallelism
export MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Setting MAX_JOBS=${MAX_JOBS} for faster compilation"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
rm -f pseudo_mamba_ext/*.so

# Install dependencies
echo "Installing dependencies..."
pip install -q packaging ninja einops transformers

# Build and install
echo "Building mamba-ssm with state management patches..."
# Use --verbose to see compilation errors if they happen
pip install -e . --no-build-isolation -v

echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Build pseudo_mamba_ext
echo "Building pseudo_mamba_ext..."
cd pseudo_mamba_ext
pip install -e .
cd ..
echo -e "${GREEN}✓ Extension build complete${NC}"
echo ""

echo "============================================================="
echo "Phase 2: API Verification"
echo "============================================================="

# Test imports
echo "Testing imports..."
python -c "
from mamba_ssm.modules.mamba_simple import Mamba, MambaInferenceState
print('  ✓ Imports successful')
"

# Test state management API
echo "Testing state management API..."
python test_state_management.py

echo ""
echo "============================================================="
echo "Phase 3: RL Benchmark Suite Test"
echo "============================================================="

echo "Running unified benchmark suite (Real Mamba)..."
python pseudo_mamba_memory_suite.py \
    --env delayed_cue \
    --controller real_mamba \
    --horizon 200 \
    --num_envs 16 \
    --total_updates 50

echo ""
echo "Running unified benchmark suite (Pseudo-Mamba Ext)..."
python pseudo_mamba_memory_suite.py \
    --env delayed_cue \
    --controller pseudo_mamba_ext \
    --horizon 200 \
    --num_envs 16 \
    --total_updates 50

echo ""
echo "============================================================="
echo "Phase 4: GRU Baseline Comparison"
echo "============================================================="

echo "Running GRU baseline..."
python pseudo_mamba_memory_suite.py \
    --env delayed_cue \
    --controller gru \
    --horizon 200 \
    --num_envs 16 \
    --total_updates 50

echo ""
echo "============================================================="
echo "Phase 5: Debug Features Verification"
echo "============================================================="

echo "Running debug features suite..."
python debug_mamba_features.py

echo ""
echo "============================================================="
echo "Build & Verification Complete!"
echo "============================================================="
echo ""
echo -e "${GREEN}All tests passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run full benchmark:"
echo "     python pseudo_mamba_memory_suite.py --env copy_memory --controller real_mamba --horizon 2000"
echo ""
echo "  2. Debug Mamba models:"
echo "     python debug_mamba_features.py"
echo ""
echo "  3. Integrate into your project"
echo ""
echo "  3. See STATE_MANAGEMENT_README.md for API docs"
echo ""
