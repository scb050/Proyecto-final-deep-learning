# Makefile — pipeline reproducible end-to-end
# Uso:
#   make setup
#   make ingest
#   make process
#   make eda
#   make train MODEL=lstm
#   make benchmark
#   make book
#   make all

PYTHON ?= python
PIP    ?= pip
CONFIG ?= config/config.yaml
MODEL  ?= lstm
SEEDS  ?= 2

.PHONY: help setup ingest process eda train benchmark book test lint format clean all

help:
	@echo "Targets: setup ingest process eda train benchmark book test lint format clean all"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install || true

ingest:
	$(PYTHON) -m src.data.ingest_inmet --config $(CONFIG)

process:
	$(PYTHON) -m src.data.process --config $(CONFIG)

eda:
	$(PYTHON) -m src.eda.general    --config $(CONFIG)
	$(PYTHON) -m src.eda.timeseries --config $(CONFIG)

train:
	$(PYTHON) -m src.training.runner --config $(CONFIG) --model $(MODEL) --seeds $(SEEDS)

benchmark:
	$(PYTHON) -m src.benchmark.compare    --config $(CONFIG)
	$(PYTHON) -m src.benchmark.stats_tests --config $(CONFIG)
	$(PYTHON) -m src.benchmark.report      --config $(CONFIG)

book:
	jupyter-book build jupyter_book

test:
	pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

clean:
	rm -rf data/interim/* data/processed/* experiments/* results/figures/* results/tables/* results/stats/*
	rm -rf jupyter_book/_build .pytest_cache .ruff_cache .mypy_cache

all: setup ingest process eda benchmark book
