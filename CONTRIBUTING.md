# Contributing to Yggdrasil

Thank you for your interest in contributing to Yggdrasil.

## Development Setup

```bash
git clone https://github.com/MukundaKatta/yggdrasil.git
cd yggdrasil
pip install -e ".[dev]"
```

## Running Tests

```bash
make test
# or
PYTHONPATH=src python3 -m pytest tests/ -v --tb=short
```

## Code Style

- Use type hints for all function signatures
- Write docstrings for public classes and methods
- Keep dependencies at zero (pure Python only)
- Target Python 3.9+ compatibility

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Project Structure

```
src/yggdrasil/
    __init__.py    — Package exports
    core.py        — Vector store and distance metrics
    graph.py       — Knowledge graph
    memory.py      — Unified memory layer
    config.py      — Configuration management
    cli.py         — Command-line interface
    __main__.py    — Module entry point
tests/
    test_core.py   — Vector store tests
    test_graph.py  — Graph tests
    test_memory.py — Memory layer tests
```
