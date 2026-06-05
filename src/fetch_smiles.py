import pandas as pd
import pubchempy as pcp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_PATH = "data/raw_ddi.csv"
OUT_PATH = "data/chemical_ddi.csv"
SEVERITY_MAP = {"Minor": 0, "Moderate": 1, "Major": 2}
CACHE = {}
MAX_WORKERS = 5

FALLBACK_NAMES = {
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


def main():
    df = pd.read_csv(RAW_PATH)
    unique_drugs = sorted(set(df["drug_a"].unique()) | set(df["drug_b"].unique()))
    print(f"Found {len(unique_drugs)} unique drug names", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_smiles, d): d for d in unique_drugs}
        for f in as_completed(futures):
            done += 1
            if done % 100 == 0:
                resolved = sum(1 for v in CACHE.values() if v is not None)
                print(f"  {done}/{len(unique_drugs)} done, {resolved} resolved", flush=True)

    df["smiles_a"] = df["drug_a"].map(CACHE)
    df["smiles_b"] = df["drug_b"].map(CACHE)
    df["severity_label"] = df["severity"].map(SEVERITY_MAP)

    before = len(df)
    df = df.dropna(subset=["smiles_a", "smiles_b", "severity_label"])
    print(f"Dropped {before - len(df)} rows with unresolved drugs", flush=True)

    df = df.drop(columns=["mechanism", "effect"], errors="ignore")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved chemical DDI data to {OUT_PATH} ({len(df)} rows)", flush=True)

    resolved = sum(1 for v in CACHE.values() if v is not None)
    print(f"Cache stats: {len(CACHE)} unique, {resolved} resolved, {len(CACHE)-resolved} failed")


if __name__ == "__main__":
    main()
