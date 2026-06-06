from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pubchempy as pcp

from src.config import DATA_PATH, RAW_DATA_PATH
from src.logger import logger

RAW_PATH: str = RAW_DATA_PATH
OUT_PATH: str = DATA_PATH
SEVERITY_MAP: dict[str, int] = {"Minor": 0, "Moderate": 1, "Major": 2}
CACHE: dict[str, str | None] = {}
MAX_WORKERS: int = 5

FALLBACK_NAMES: dict[str, str] = {
    "St. John's Wort": "hypericum perforatum",
    "Ginseng": "panax ginseng",
    "Ginkgo Biloba": "ginkgo biloba",
    "Sucralfate": "sucralfate",
    "Cholestyramine": "cholestyramine",
    "Colestipol": "colestipol",
    "Barbiturates": "phenobarbital",
    "Carbonic Anhydrase Inhibitors": "acetazolamide",
    "Electrolyte Supplements": "potassium chloride",
    "Zinc Supplements": "zinc",
    "Iodinated Contrast Dye": "iohexol",
    "Avocado": "avocado",
    "ACE Inhibitors (e.g., Lisinopril)": "lisinopril",
    "Sulfamethoxazole/Trimethoprim": "trimethoprim",
}


def get_smiles(drug_name: str) -> tuple[str, str | None]:
    """Resolve a drug name to its SMILES string via PubChem.

    Args:
        drug_name: Common or scientific drug name.

    Returns:
        Tuple of (cleaned drug name, SMILES string or None if unresolved).
    """
    clean = drug_name.strip()
    if clean in CACHE:
        return clean, CACHE[clean]
    candidates = [clean]
    if clean in FALLBACK_NAMES:
        candidates.append(FALLBACK_NAMES[clean])
    for name in candidates:
        try:
            compounds = pcp.get_compounds(name, namespace="name", timeout=15)
            if compounds:
                smi = compounds[0].connectivity_smiles
                CACHE[clean] = smi
                return clean, smi
        except Exception:
            pass
    CACHE[clean] = None
    return clean, None


def main() -> None:
    """Resolve all unique drug names to SMILES and save the enriched dataset.

    Reads raw_ddi.csv, resolves drug names via PubChem (with fallbacks
    for herbal/complex names), maps severity strings to ordinal labels,
    drops unresolved rows, and writes the result to chemical_ddi.csv.
    """
    df = pd.read_csv(RAW_PATH)
    unique_drugs = sorted(set(df["drug_a"].unique()) | set(df["drug_b"].unique()))
    logger.info("Found %d unique drug names", len(unique_drugs))

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_smiles, d): d for d in unique_drugs}
        for f in as_completed(futures):
            done += 1
            if done % 100 == 0:
                resolved = sum(1 for v in CACHE.values() if v is not None)
                logger.info("  %d/%d done, %d resolved", done, len(unique_drugs), resolved)

    df["smiles_a"] = df["drug_a"].map(CACHE)
    df["smiles_b"] = df["drug_b"].map(CACHE)
    df["severity_label"] = df["severity"].map(SEVERITY_MAP)

    before = len(df)
    df = df.dropna(subset=["smiles_a", "smiles_b", "severity_label"])
    logger.info("Dropped %d rows with unresolved drugs", before - len(df))

    df = df.drop(columns=["mechanism", "effect"], errors="ignore")
    df.to_csv(OUT_PATH, index=False)
    logger.info("Saved chemical DDI data to %s (%d rows)", OUT_PATH, len(df))

    resolved = sum(1 for v in CACHE.values() if v is not None)
    logger.info("Cache stats: %d unique, %d resolved, %d failed", len(CACHE), resolved, len(CACHE) - resolved)


if __name__ == "__main__":
    main()
