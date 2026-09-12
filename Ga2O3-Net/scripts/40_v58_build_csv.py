#!/usr/bin/env python
"""
Phase 58 — build the V58 enhanced training CSV (design doc §6.4).

Reads the most recent V55/V57 experimental CSV and adds four V58 columns WITHOUT
dropping or altering any existing row or column:

  paper_id          : stable string id derived from the `doi` column
                      (falls back to a hash of identifier columns when doi missing)
  dft_features_idx  : cache key "{element}|{atm}|{T_C}" matching the keys in
                      data/processed/dft_features_cache_v58.npz; "NONE" when the
                      row's first-cation dopant is not among the 19 cache dopants
                      (or undoped / unparseable) → model falls back to z_DFT=0.
  mu_O_eff_eV       : Reuter-Scheffler mu_O(T_K, p_O2) bilinear lookup from
                      data/processed/mu_O_table_v58.npz.
  mu_dopant_max_eV  : competing-phase delta_mu_M_max for the row's dopant at the
                      nearest (atm, T) bin from data/processed/dft_chemical_potentials.json;
                      NaN when dopant not tabulated.

No training. Pure data assembly. Output:
  data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v58.csv
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dopant_spec import DopantSpec
from src.data.mu_o_table import lookup as mu_o_lookup

PROJ = Path(__file__).resolve().parents[1]
EXP_DIR = PROJ / "data" / "raw" / "experimental"
PROC = PROJ / "data" / "processed"

DFT_CACHE = PROC / "dft_features_cache_v58.npz"
MU_O_TABLE = PROC / "mu_O_table_v58.npz"
CHEM_POT_JSON = PROC / "dft_chemical_potentials.json"
OUT_CSV = EXP_DIR / "ga2o3_exp_aug3x_vcinterp_v58.csv"

# DFT cache temperature bins (degC) and atmosphere bins.
T_BINS = np.array([500, 700, 900, 1100], dtype=float)
ATM_BINS = ["Ar", "Ar_O2_4_1", "Ar_O2_1_1", "O2"]

# p_O2 (atm) per atmosphere bin — from the dft cache meta `atm_p_O2`.
ATM_P_O2 = {"Ar": 1e-5, "Ar_O2_4_1": 0.05, "Ar_O2_1_1": 0.2, "O2": 1.0}

# Map the raw CSV `atmosphere` strings onto one of the four DFT atmosphere bins.
#   reducing / inert / no-O2  -> Ar          (lowest p_O2)
#   Ar:O2=3:1 (mostly Ar)      -> Ar_O2_4_1  (most-Ar O2 mix in the cache)
#   Ar:O2=1:1                  -> Ar_O2_1_1
#   O2-rich / oxidizing        -> O2          (highest p_O2)
ATM_MAP = {
    "Ar": "Ar",
    "vacuum": "Ar",
    "inert": "Ar",
    "Ar+H2": "Ar",
    "Ar+N2": "Ar",
    "N2": "Ar",
    "N2_plasma": "Ar",
    "Ar:O2=3:1": "Ar_O2_4_1",
    "Ar:O2=1:1": "Ar_O2_1_1",
    "O2": "O2",
    "O2_plasma": "O2",
    "air": "O2",
    "N2+O2": "O2",
    "N2O": "O2",
    "wet": "O2",
}
DEFAULT_ATM_BIN = "Ar"  # used when atmosphere is missing/unrecognized


def pick_source_csv() -> Path:
    for name in ("ga2o3_exp_aug3x_vcinterp_v57.csv",
                 "ga2o3_exp_aug3x_vcinterp_v55.csv"):
        p = EXP_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("No V57 or V55 source CSV found in " + str(EXP_DIR))


def first_cation(spec_str) -> str | None:
    """First-component cation symbol, or None for undoped / NaN / unparseable."""
    if pd.isna(spec_str):
        return None
    try:
        sp = DopantSpec.parse(str(spec_str))
    except Exception:
        return None
    if sp.is_undoped:
        return None
    return sp.components[0].cation


def atm_bin(atm) -> str:
    if pd.isna(atm):
        return DEFAULT_ATM_BIN
    return ATM_MAP.get(str(atm).strip(), DEFAULT_ATM_BIN)


def nearest_T_bin(T_C) -> int:
    """Nearest of {500,700,900,1100}C; default 700C when temperature missing."""
    if pd.isna(T_C):
        return 700
    return int(T_BINS[int(np.argmin(np.abs(T_BINS - float(T_C))))])


def make_paper_id(row, idx) -> str:
    doi = row.get("doi", None)
    if not pd.isna(doi) and str(doi).strip():
        token = str(doi).strip()
        # strip a leading "DOI:" prefix for cleaner keys; keep free-text refs as-is
        if token.lower().startswith("doi:"):
            token = token[4:].strip()
        return token
    # fallback: stable hash of available identifier columns
    parts = [str(row.get(c, "")) for c in
             ("dopant_spec", "method", "atmosphere", "temperature_C", "notes")]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"NODOI_{h}"


def main() -> None:
    src = pick_source_csv()
    df = pd.read_csv(src)
    n0 = len(df)
    cols0 = list(df.columns)
    print(f"[load] source = {src.name}  rows = {n0}  cols = {len(cols0)}")

    # --- DFT cache key set (19 dopants × 4 atm × 4 T = 304 keys) ----------------
    dft = np.load(DFT_CACHE, allow_pickle=True)
    dft_keys = set(str(k) for k in dft["keys"])
    dft_meta = json.loads(str(dft["meta_json"]))
    dft_dopants = set(dft_meta["dopants_covered"])
    print(f"[dft]  cache keys = {len(dft_keys)}  dopants = {sorted(dft_dopants)}")

    # --- chemical-potential table (delta_mu_M_max) ------------------------------
    chem = json.loads(CHEM_POT_JSON.read_text())["entries"]
    chem_elems = set(chem.keys())
    print(f"[chem] mu_dopant entries = {sorted(chem_elems)}")

    paper_id = []
    dft_idx = []
    mu_o_eff = []
    mu_dop_max = []

    n_valid_dft = 0
    n_none_dft = 0
    n_mu_dop = 0

    for idx, row in df.iterrows():
        paper_id.append(make_paper_id(row, idx))

        el = first_cation(row.get("dopant_spec"))
        ab = atm_bin(row.get("atmosphere"))
        tb = nearest_T_bin(row.get("temperature_C"))

        # dft_features_idx -------------------------------------------------------
        key = f"{el}|{ab}|{tb}" if el is not None else None
        if key is not None and key in dft_keys:
            dft_idx.append(key)
            n_valid_dft += 1
        else:
            dft_idx.append("NONE")
            n_none_dft += 1

        # mu_O_eff_eV (continuous physical lookup; uses actual T, not binned) ----
        T_C = row.get("temperature_C")
        T_K = (700.0 + 273.15) if pd.isna(T_C) else float(T_C) + 273.15
        p_o2 = ATM_P_O2[ab]
        mu_o_eff.append(round(mu_o_lookup(MU_O_TABLE, T_K, p_o2), 6))

        # mu_dopant_max_eV (nearest atm/T bin) -----------------------------------
        if el is not None and el in chem_elems:
            bin_key = f"{ab}|{tb}C"
            entry = chem[el]["by_atm_T"]
            val = entry.get(bin_key)
            if val is not None:
                mu_dop_max.append(round(float(val["delta_mu_M_max_eV"]), 6))
                n_mu_dop += 1
            else:
                mu_dop_max.append(np.nan)
        else:
            mu_dop_max.append(np.nan)

    df["paper_id"] = paper_id
    df["dft_features_idx"] = dft_idx
    df["mu_O_eff_eV"] = mu_o_eff
    df["mu_dopant_max_eV"] = mu_dop_max

    # --- invariants: no rows dropped, original columns preserved & unmodified ---
    assert len(df) == n0, f"row count changed {n0} -> {len(df)}"
    for c in cols0:
        assert c in df.columns, f"original column lost: {c}"

    df.to_csv(OUT_CSV, index=False)

    print(f"\n[write] {OUT_CSV.name}  rows = {len(df)}  cols = {len(df.columns)} "
          f"(+{len(df.columns) - len(cols0)} new)")
    print(f"[stat] dft_features_idx valid = {n_valid_dft}  NONE = {n_none_dft} "
          f"(coverage {100.0 * n_valid_dft / n0:.1f}%)")
    print(f"[stat] mu_dopant_max_eV non-NaN = {n_mu_dop}  NaN = {n0 - n_mu_dop}")
    print(f"[stat] mu_O_eff_eV  min={min(mu_o_eff):.4f}  max={max(mu_o_eff):.4f}")
    print("[stat] new columns:", [c for c in df.columns if c not in cols0])


if __name__ == "__main__":
    main()
