"""Tests for PubChem SMILES resolution."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.fetch_smiles import CACHE, SEVERITY_MAP, get_smiles, main


class TestGetSmiles:
    def setup_method(self):
        CACHE.clear()

    def test_returns_none_when_no_compounds_found(self):
        with patch("src.fetch_smiles.pcp.get_compounds", return_value=[]):
            result = get_smiles("unknown_drug_xyz123")
        assert result == ("unknown_drug_xyz123", None)

    def test_returns_smiles_when_compound_found(self):
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "CCO"
        with patch("src.fetch_smiles.pcp.get_compounds", return_value=[mock_compound]):
            name, smi = get_smiles("Ethanol")
        assert name == "Ethanol"
        assert smi == "CCO"

    def test_caches_result(self):
        CACHE.clear()
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "O"
        with patch("src.fetch_smiles.pcp.get_compounds", return_value=[mock_compound]):
            get_smiles("Water")
        assert CACHE.get("Water") == "O"

    def test_returns_cached_result(self):
        CACHE.clear()
        CACHE["Water"] = "O"
        with patch("src.fetch_smiles.pcp.get_compounds") as mock_get:
            name, smi = get_smiles("Water")
        mock_get.assert_not_called()
        assert smi == "O"

    def test_strips_whitespace(self):
        CACHE.clear()
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "CCO"
        with patch("src.fetch_smiles.pcp.get_compounds", return_value=[mock_compound]):
            name, smi = get_smiles("  Ethanol  ")
        assert name == "Ethanol"
        assert smi == "CCO"

    def test_uses_fallback_name(self):
        CACHE.clear()
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "CC"

        def get_compounds_side_effect(name, **kwargs):
            if name == "St. John's Wort":
                return []
            return [mock_compound]

        with patch("src.fetch_smiles.pcp.get_compounds", side_effect=get_compounds_side_effect) as mock_get:
            get_smiles("St. John's Wort")
        mock_get.assert_any_call("hypericum perforatum", namespace="name", timeout=15)

    def test_handles_exception_during_lookup(self):
        CACHE.clear()
        with patch("src.fetch_smiles.pcp.get_compounds", side_effect=Exception("timeout")):
            name, smi = get_smiles("SomeDrug")
        assert smi is None


class TestMain:
    def test_load_csv_success(self):
        CACHE.clear()
        mock_raw = pd.DataFrame(
            {
                "drug_a": ["Water", "Ethanol"],
                "drug_b": ["Ethanol", "Water"],
                "severity": ["Minor", "Moderate"],
                "mechanism": ["x", "y"],
                "effect": ["a", "b"],
            }
        )
        mock_compound_a = MagicMock()
        mock_compound_a.connectivity_smiles = "O"
        mock_compound_b = MagicMock()
        mock_compound_b.connectivity_smiles = "CCO"

        def mock_get_compounds(name, **kwargs):
            if "Water" in name or "O" == name:
                return [mock_compound_a]
            return [mock_compound_b]

        with (
            patch("src.fetch_smiles.pd.read_csv", return_value=mock_raw),
            patch("src.fetch_smiles.pcp.get_compounds", side_effect=mock_get_compounds),
            patch("src.fetch_smiles.pd.DataFrame.to_csv") as mock_to_csv,
        ):
            main()

        args, _ = mock_to_csv.call_args
        saved_path = args[0]
        assert saved_path == "data/chemical_ddi.csv"

    def test_drops_unresolved_drugs(self):
        CACHE.clear()
        mock_raw = pd.DataFrame(
            {
                "drug_a": ["UnknownDrug"],
                "drug_b": ["Water"],
                "severity": ["Major"],
                "mechanism": [""],
                "effect": [""],
            }
        )
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "O"

        def mock_get_compounds(name, **kwargs):
            if name.strip() == "UnknownDrug":
                return []
            return [mock_compound]

        with (
            patch("src.fetch_smiles.pd.read_csv", return_value=mock_raw),
            patch("src.fetch_smiles.pcp.get_compounds", side_effect=mock_get_compounds),
            patch("src.fetch_smiles.pd.DataFrame.to_csv"),
        ):
            main()

    def test_severity_mapping(self):
        assert SEVERITY_MAP == {"Minor": 0, "Moderate": 1, "Major": 2}
