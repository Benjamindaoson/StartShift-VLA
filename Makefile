PYTHON ?= python

.PHONY: install test lint format-check splits m0

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check startshift tests

format-check:
	$(PYTHON) -m compileall -q startshift tests

splits:
	startshift make-splits --output-dir splits --seed 42

m0:
	bash scripts/run_m0.sh
