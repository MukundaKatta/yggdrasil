.PHONY: test lint clean install dev

test:
	PYTHONPATH=src python3 -m pytest tests/ -v --tb=short

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info src/*.egg-info

lint:
	python3 -m py_compile src/yggdrasil/core.py
	python3 -m py_compile src/yggdrasil/graph.py
	python3 -m py_compile src/yggdrasil/memory.py
	python3 -m py_compile src/yggdrasil/cli.py
