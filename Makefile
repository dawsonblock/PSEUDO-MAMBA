.PHONY: install test lint format clean benchmark

install:
	pip install -e .

test:
	pytest tests/

lint:
	flake8 pseudo_mamba tests
	isort --check-only pseudo_mamba tests
	black --check pseudo_mamba tests

format:
	isort pseudo_mamba tests
	black pseudo_mamba tests

clean:
	rm -rf build dist *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

benchmark:
	python pseudo_mamba_memory_suite.py --config configs/memory_suite.yaml
