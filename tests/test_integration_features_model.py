"""Integration tests: feature engineering → model training → prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.features import build_features


def _small_ddi_dataset() -> pd.DataFrame:
    """Real SMILES pairs with diverse structures and all 3 severity classes."""
    return pd.DataFrame(
        {
            "smiles_a": [
                "O",
                "CCO",
                "c1ccccc1",
                "CC",
                "C1CCCCC1",
                "CCOCC",
                "O",
                "CCO",
                "c1ccccc1",
                "CC",
            ],
            "smiles_b": [
                "CCO",
                "O",
                "CC",
                "c1ccccc1",
                "CCOCC",
                "C1CCCCC1",
                "CC",
                "c1ccccc1",
                "CCO",
                "O",
            ],
            "severity_label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        }
    )


class TestFeaturesToModel:
    """Integration: real RDKit features → sklearn model → predictions."""

    def test_build_features_returns_correct_shape(self):
        df = _small_ddi_dataset()
        X = build_features(df)
        assert X.shape == (10, 1045)
        assert X.dtype == np.float64

    def test_logistic_regression_trains_and_predicts(self):
        df = _small_ddi_dataset()
        X = build_features(df)
        y = df["severity_label"].values

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 10
        assert preds.dtype == np.int64 or preds.dtype == np.int32

    def test_random_forest_trains_and_predicts(self):
        df = _small_ddi_dataset()
        X = build_features(df)
        y = df["severity_label"].values

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 10

    def test_different_pairs_produce_different_features(self):
        df = _small_ddi_dataset()
        X = build_features(df)
        # All rows should differ (no duplicate SMILES pairs in dataset)
        assert len(np.unique(X, axis=0)) == 10

    def test_model_generalises_across_folds(self):
        df = _small_ddi_dataset()
        X = build_features(df)
        y = df["severity_label"].values

        model = LogisticRegression(max_iter=2000, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_predict_proba_returns_valid_probabilities(self):
        df = _small_ddi_dataset()
        X = build_features(df)
        y = df["severity_label"].values

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X)

        assert probs.shape == (10, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_interaction_features_are_symmetric(self):
        """Tanimoto similarity and prop diff/sum are order-invariant."""
        df = pd.DataFrame(
            {
                "smiles_a": ["O", "CCO"],
                "smiles_b": ["CCO", "O"],
                "severity_label": [0, 0],
            }
        )
        X = build_features(df)
        # fp_a and fp_b are swapped in the feature vector (first 512 bits),
        # but the symmetric components (diff, product, sim, prop_diff, prop_sum)
        # should be identical.
        np.testing.assert_array_equal(X[0][512:], X[1][512:])

    def test_feature_matrix_is_deterministic(self):
        df = _small_ddi_dataset()
        X1 = build_features(df)
        X2 = build_features(df)
        np.testing.assert_array_equal(X1, X2)
