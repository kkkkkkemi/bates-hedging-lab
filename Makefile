PYTHON ?= python

.PHONY: report quick test verify
report:
	$(PYTHON) -m quantproj --config experiments/full.yaml --output output/full
quick:
	$(PYTHON) -m quantproj --config experiments/quick.yaml --output output/quick
test:
	$(PYTHON) -m pytest -m "not slow"
verify:
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy
