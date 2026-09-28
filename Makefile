PYTHON ?= python

.PHONY: install test lint compile ci verify splits m0 plan

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check startshift tests scripts/run_full_matrix.py scripts/verify_environment.py

compile:
	$(PYTHON) -m compileall -q startshift tests scripts

ci: lint compile test

verify:
	$(PYTHON) scripts/verify_environment.py

splits:
	startshift make-splits --output-dir splits --seed 42

m0:
	bash scripts/run_m0.sh

plan:
	$(PYTHON) scripts/run_full_matrix.py
