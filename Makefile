.PHONY: train api test lint format clean export install

install:
	pip install -r requirements.txt

train:
	python src/train.py

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -v tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf models/ data/features.npy data/labels.npy results.json mlruns/ mlflow.db

export:
	python src/export_model.py

dvc-repro:
	dvc repro

all: install train test
