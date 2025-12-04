#!/bin/bash
set -e

echo "Running syntax check..."
find . -name "*.py" -not -path "./build/*" -not -path "./dist/*" -not -path "./.git/*" -exec python -m py_compile {} +

echo "Running unit tests..."
pytest tests/test_autograd_equivalence.py

echo "Running benchmark dry-run..."
python pseudo_mamba_memory_suite.py --config configs/memory_suite.yaml --dry-run

echo "All checks passed!"
