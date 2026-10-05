PYTHON ?= python

.PHONY: verify test lint compile ci
verify:
	$(PYTHON) scripts/verify_archive.py
test:
	$(PYTHON) -m pytest -ra
lint:
	$(PYTHON) -m ruff check startshift tests scripts
compile:
	$(PYTHON) -m compileall -q startshift tests scripts
ci: lint compile test verify
