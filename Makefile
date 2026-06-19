.PHONY: train build-features search-hyperparams api test lint format clean export install hooks report report-watch

install:
	uv sync

hooks:
	uv run pre-commit install

build-features:
	uv run python -m src.build_features

search-hyperparams:
	uv run python -m src.search_hyperparams

train: build-features search-hyperparams

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

report:
	latexmk -pdf -shell-escape -cd report/main.tex

report-watch:
	latexmk -pvc -pdf -shell-escape -cd report/main.tex

dvc-repro:
	dvc repro

all: install train test
