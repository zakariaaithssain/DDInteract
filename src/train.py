"""Pipeline orchestrator — runs both steps sequentially.

Usage:
    python -m src.train               # run full pipeline
    python -m src.build_features       # step 1 only
    python -m src.search_hyperparams   # step 2 only
"""

from src.build_features import main as build_features
from src.search_hyperparams import main as search_hyperparams


def main() -> None:
    build_features()
    search_hyperparams()


if __name__ == "__main__":
    main()
