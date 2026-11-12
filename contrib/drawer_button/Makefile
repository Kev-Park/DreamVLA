.PHONY: test lint check

test:
	CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests

lint:
	ruff check .
	ruff format --check .

check: lint test
