"""
Phase 5E: within-DOI concentration-response augmentation.

For each (DOI, dopant element) group with >=2 labeled rows, linearly interpolate
N synthetic points between the min and max concentration. Interpolated PDR is
the linear fit on log-scale concentration, jittered with small Gaussian noise
to avoid collinearity. All other columns (temperature, method, atmosphere,
substrate, notes, voltage, wavelength) are copied from the nearest observed
row by concentration.

Design rationale (docs/experiment_log.md §13.30/§13.33):
  Phase 5B showed within-DOI rho(conc, pred) = -0.228 for Mg despite
  rho(conc, true) = +0.725. Even with an architectural fix (5C/5D), 190 labeled
  rows may be too thin for the model to learn within-DOI slopes. This script
  densifies the slope information without introducing cross-paper leakage —
  every synthetic row inherits its source DOI so GroupKFold keeps it in the
  same fold.

Usage:
    python scripts/make_phase5e_augmentation.py \
        --in data/raw/experimental/ga2o3_exp_augmented_3x.csv \
        --out data/raw/experimental/ga2o3_exp_aug3x_interp.csv \
        --n-interp 3
"""

import argparse
import numpy as np
import pandas as pd


def interpolate_group(
    group: pd.DataFrame,
    n_interp: int,
    rng: np.random.Generator,
    target_col: str = "photo_dark_ratio",
    label_noise: float = 0.05,
    interp_tag: str = "interp_phase5e",
) -> list[dict]:
    """Interpolate n_interp synthetic rows between min and max concentration."""
    g = group.sort_values("concentration_at%").reset_index(drop=True)
    c_vals = g["concentration_at%"].astype(float).values
    y_vals = g[target_col].astype(float).values
    c_min, c_max = c_vals.min(), c_vals.max()
    if c_max - c_min < 1e-6:
        return []

    c_new = np.linspace(c_min, c_max, n_interp + 2)[1:-1]

    log_c = np.log10(np.clip(c_vals, 1e-4, None))
    slope, intercept = np.polyfit(log_c, y_vals, 1)
    y_new = slope * np.log10(np.clip(c_new, 1e-4, None)) + intercept
    y_new = y_new + rng.normal(0.0, label_noise, size=y_new.shape)

    # Columns that must not be carried over from the nearest real row on a
    # synthetic sample — they are either the target itself (already set) or
    # other measurement scalars that would be inconsistent with the jittered
    # concentration.
    _unrelated_measurements = {
        "photo_dark_ratio", "vacancy_concentration",
        "photocurrent_uA", "dark_current_pA", "sensitivity",
    }

    out = []
    for c_syn, y_syn in zip(c_new, y_new):
        nearest_idx = int(np.argmin(np.abs(c_vals - c_syn)))
        base_row = g.iloc[nearest_idx].to_dict()
        base_row["concentration_at%"] = float(c_syn)
        # Blank out every measurement column, then set the one we interpolated.
        for col in _unrelated_measurements:
            if col in base_row:
                base_row[col] = np.nan
        base_row[target_col] = float(y_syn)
        element = base_row.get("element", "")
        if pd.notna(element) and element and element != "—":
            base_row["dopant_spec"] = f"{element}:{float(c_syn)/100.0:.6f}"
        notes_existing = str(base_row.get("notes", "") or "")
        base_row["notes"] = (notes_existing + f" [{interp_tag}]").strip()
        base_row["usable_flag"] = "ok"
        out.append(base_row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--n-interp", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-col", default="photo_dark_ratio",
                    help="Column to interpolate (e.g. photo_dark_ratio or "
                         "vacancy_concentration). Default: photo_dark_ratio.")
    ap.add_argument("--label-noise", type=float, default=0.05,
                    help="Gaussian noise σ added to interpolated labels.")
    ap.add_argument("--interp-tag", default="interp_phase5e",
                    help="Tag written into the notes column on synthetic rows.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    rng = np.random.default_rng(args.seed)

    usable = df[
        df[args.target_col].notna()
        & df["concentration_at%"].notna()
        & df["element"].notna()
        & (df["element"] != "—")
        & (df.get("usable_flag", "ok") == "ok")
    ].copy()

    new_rows: list[dict] = []
    for (doi, element), group in usable.groupby(["doi", "element"]):
        if len(group) < 2:
            continue
        new_rows.extend(
            interpolate_group(
                group, args.n_interp, rng,
                target_col=args.target_col,
                label_noise=args.label_noise,
                interp_tag=args.interp_tag,
            )
        )

    print(f"Interpolated {len(new_rows)} synthetic rows from "
          f"{len(usable.groupby(['doi', 'element']))} (doi, element) groups.")

    if not new_rows:
        print("No interpolated rows produced; writing input unchanged.")
        df.to_csv(args.out_path, index=False)
        return

    new_df = pd.DataFrame(new_rows)
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[df.columns]

    merged = pd.concat([df, new_df], ignore_index=True)
    merged.to_csv(args.out_path, index=False)
    print(f"Wrote {len(merged)} rows (was {len(df)}) to {args.out_path}")


if __name__ == "__main__":
    main()
