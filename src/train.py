"""Pipeline orchestrator — runs all three steps sequentially.

Usage:
    python -m src.train           # run full pipeline
    python -m src.build_features   # step 1 only
    python -m src.select_features  # step 2 only
    python -m src.search_hyperparams  # step 3 only
"""

from src.build_features import main as build_features
from src.search_hyperparams import main as search_hyperparams
from src.select_features import main as select_features


def main() -> None:
    build_features()
    select_features()
    search_hyperparams()


if __name__ == "__main__":
    main()
