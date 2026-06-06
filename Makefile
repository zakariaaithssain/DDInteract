.PHONY: train api test lint format clean export install hooks

install:
	uv sync

hooks:
	uv run pre-commit install

train:
	uv run python -m src.train

api:
	uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	uv run pytest -v tests/

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

clean:
	rm -rf models/ data/features.npy data/labels.npy results.json mlruns/ mlflow.db

export:
	uv run python -m src.export_model

dvc-repro:
	dvc repro

all: install train test
