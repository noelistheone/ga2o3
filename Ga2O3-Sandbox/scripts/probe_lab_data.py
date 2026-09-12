import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd

NET = Path(__file__).resolve().parents[1].parent / "Ga2O3-Net"
df = pd.read_csv(NET / "data/raw/experimental/ga2o3_exp.csv")
print("total rows:", len(df))
print("columns:", list(df.columns))
# identify lab same-lab-anchor rows (Phase-105 corrected: doi == 'Original CSV' family)
doi = df["doi"].astype(str) if "doi" in df.columns else pd.Series([""] * len(df))
mask = doi.str.contains("Original", case=False, na=False)
print("\nlab-anchor rows (doi~Original):", int(mask.sum()))
keep = [c for c in ["element", "dopant_spec", "concentration_at%", "method", "atmosphere",
                    "temperature_C", "photo_dark_ratio", "dark_current", "responsivity",
                    "detectivity", "eqe_percent", "vacancy_concentration", "doi"] if c in df.columns]
lab = df[mask][keep]
for _, r in lab.iterrows():
    print({k: r[k] for k in keep})
# also carrier/mobility columns if present
print("\ncarrier/mobility columns present:",
      [c for c in df.columns if any(t in c.lower() for t in ["carrier", "hall", "mobility", "mu_", "sigma", "conduct"])])
