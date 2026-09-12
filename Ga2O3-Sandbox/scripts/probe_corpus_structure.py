import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

NET = Path(__file__).resolve().parents[1].parent / "Ga2O3-Net"
df = pd.read_csv(NET / "data/raw/experimental/ga2o3_exp.csv")
df = df[df["usable_flag"].fillna(1) != 0] if "usable_flag" in df.columns else df

# within-paper doped-vs-undoped pairs for PDR / dark / V_O
print("=== ga2o3_exp.csv device corpus ===")
print("rows:", len(df), "| unique doi:", df["doi"].nunique())
# papers with both a doped and an undoped row
grp = df.groupby("doi")
pairs_pdr = pairs_dark = pairs_vo = 0
multi_conc = 0
for doi, g in grp:
    has_und = (g["element"].astype(str).str.lower() == "undoped").any()
    has_dop = (g["element"].astype(str).str.lower() != "undoped").any()
    if has_und and has_dop:
        if g["photo_dark_ratio"].notna().sum() >= 2:
            pairs_pdr += 1
        if g["dark_current_pA"].notna().sum() >= 2:
            pairs_dark += 1
        if g["vacancy_concentration"].notna().sum() >= 2:
            pairs_vo += 1
    # concentration series (same element, >=2 concentrations)
    for el, ge in g.groupby("element"):
        if str(el).lower() != "undoped" and ge["concentration_at%"].nunique() >= 2:
            multi_conc += 1
print(f"papers with doped+undoped & >=2 PDR: {pairs_pdr}")
print(f"papers with doped+undoped & >=2 dark: {pairs_dark}")
print(f"papers with doped+undoped & >=2 V_O: {pairs_vo}")
print(f"(element, doi) concentration series (>=2 conc): {multi_conc}")

# distinct dopant elements in corpus
els = sorted(set(str(e) for e in df["element"].dropna() if str(e).lower() != "undoped"))
print(f"\ndistinct dopant elements ({len(els)}):", els)

# transport corpus
tp = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
print("\n=== transport corpus ===")
print("rows:", len(tp), "cols:", list(tp.columns))
