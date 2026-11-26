PYTHON ?= python

.PHONY: report quick test verify rates-quick rates-report report-all
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

rates-quick:
	$(PYTHON) -m quantproj.rates --config experiments/rates/quick.yaml --output output/rates/quick
rates-report:
	$(PYTHON) -m quantproj.rates --config experiments/rates/full.yaml --output output/rates/full
report-all: report rates-report
