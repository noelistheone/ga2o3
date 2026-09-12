"""Phase 58 — parse the 90 gap-dopant V_O DFT calcs into REFERENCE-INDEPENDENT quantities.

Inputs:  dft/qe_hse06/outputs/PBEU_v58_{dopant}_{O_I|O_II|O_III}_q{0,1,2}.out
         = an oxygen vacancy (V_O) in a {dopant}-DOPED 79-atom supercell.

⚠ CRITICAL physics constraint (design-doc §7 "DFT anchor 数据源更正" callout +
docs/phase58_lite_dft_finding.md):
  There is NO "doped-no-vacancy" reference cell on disk (only V_O-containing relaxed
  structures exist). Therefore the ABSOLUTE per-dopant V_O formation energy ΔE_f is NOT
  extractable here — referencing the doped V_O cell to PRISTINE Ga2O3 (+μ_O only) is
  exactly the bug that produced the ~1490 eV garbage in `v_o_ef.csv` (it conflates the
  Ga→dopant substitution energy +μ_Ga−μ_M, ≈1200–1500 eV, which was never added).
  This script DOES NOT compute ΔE_f. It extracts only the two quantities for which the
  doped-no-vacancy reference and the μ_O term CANCEL between the compared total energies:

  (a) CHARGE TRANSITION LEVELS ε(q/q') per (dopant, site).
      For two charge states of the SAME (dopant, site) V_O defect, the formation energies
        E_f^q(εF)  = E_tot[q] − E_ref(doped,no-V_O) + μ_O + q·(ε_VBM + εF) + E_corr[q]
        E_f^q'(εF) = E_tot[q'] − E_ref(doped,no-V_O) + μ_O + q'·(ε_VBM + εF) + E_corr[q']
      Setting E_f^q = E_f^q' and solving for the Fermi level (measured above VBM, ε_VBM=0):
        ε(q/q') = [ (E_tot[q]+E_corr[q]) − (E_tot[q']+E_corr[q']) ] / (q' − q)
      Both E_ref and μ_O cancel identically → reference-INDEPENDENT, valid to extract.
      E_corr[q] is the Freysoldt charged-supercell correction (also a charge-state
      difference quantity → reference-independent), reused from dft/qe_hse06/freysoldt.py.

  (b) SITE PREFERENCE: lowest total energy among O_I / O_II / O_III for the same
      (dopant, charge). All three share the SAME supercell composition (one O removed
      from the doped cell, just a different O sublattice site), so their total energies
      are directly comparable; the lowest-E site is the dominant V_O site for that dopant.

Energy line: `!    total energy   = ... Ry` (the converged SCF result), Ry→eV ×13.605693.
SCF-non-converged calcs (200-iteration cutoff, no `!` line) are flagged and skipped.

Outputs (results-in-workdir rule — every parsed number on disk):
  dft/qe_hse06/results/v58_gap_vo_transitions.csv
      dopant, site, charge, total_energy_eV, scf_converged, e_corr_FS_eV,
      cell_volume_A3, force_total, status
  dft/qe_hse06/results/v58_gap_dopant_summary.csv
      dopant, dominant_site, eps_0_1_eV, eps_1_2_eV, n_charges_done, dominant_charge_eF4.0

GPU-over-CPU note: pure text parse + scalar arithmetic; no GPU path applies. CPU is correct.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from dft.qe_hse06.freysoldt import freysoldt_correction

PROJ = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJ / "dft" / "qe_hse06" / "outputs"
RESULTS_DIR = PROJ / "dft" / "qe_hse06" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RY_TO_EV = 13.605693
EG_GA2O3 = 4.85
# n-type host Fermi level used for the dominant-charge call (same as build_dft_cache
# ANCHOR_FERMI_EV). At εF=4.0 above VBM, the dominant charge of a V_O is whichever state q
# minimizes E_f^q(εF) relative to a neighbouring state, decided directly by the transition
# levels: q=+2 stable above ε(+1/+2), q=+1 between ε(+1/+2) and ε(0/+1), q=0 below ε(0/+1).
ANCHOR_FERMI_EV = 4.0

GAP_DOPANTS = ["Al", "B", "Cr", "Cu", "Fe", "Ni", "Ti", "V", "W", "Zn"]
SITES = ["O_I", "O_II", "O_III"]
CHARGES = [0, 1, 2]

PREFIX_RE = re.compile(r"^PBEU_v58_(?P<dop>[A-Za-z]+)_(?P<site>O_I{1,3}|O_II|O_III)_q(?P<q>\d)$")


def parse_one(out_file: Path) -> dict:
    """Parse a single QE .out file for the converged total energy + diagnostics.

    Returns dict: dopant, site, charge, total_energy_eV, scf_converged, force_total,
                  cell_volume_A3, status.
    status ∈ {"OK","SCF_NOT_CONVERGED","NO_ENERGY","NOT_DONE","MISSING","READ_ERROR"}.
    """
    stem = out_file.stem
    m = PREFIX_RE.match(stem)
    if m:
        dopant, site, charge = m.group("dop"), m.group("site"), int(m.group("q"))
    else:  # fall back to manual split (robust to site spelling)
        parts = stem.replace("PBEU_v58_", "").split("_")
        dopant = parts[0]
        site = "_".join(parts[1:-1])
        charge = int(parts[-1][1:])

    base = {"dopant": dopant, "site": site, "charge": charge,
            "total_energy_eV": np.nan, "scf_converged": False,
            "e_corr_FS_eV": np.nan, "cell_volume_A3": np.nan,
            "force_total": np.nan, "status": "UNKNOWN"}

    if not out_file.exists():
        base["status"] = "MISSING"
        return base
    try:
        text = out_file.read_text()
    except Exception as e:  # pragma: no cover
        base["status"] = "READ_ERROR"
        base["_msg"] = str(e)[:80]
        return base

    job_done = "JOB DONE" in text
    scf_converged = bool(re.search(r"convergence has been achieved", text))
    scf_failed = bool(re.search(r"convergence NOT achieved", text))
    base["scf_converged"] = scf_converged

    # Cell volume (a.u.^3 → Å^3); needed for the Freysoldt correction.
    vol_m = re.search(r"unit-cell volume\s*=\s*([\d.]+)\s*\(a\.u\.\)\^3", text)
    vol_A3 = float(vol_m.group(1)) * 0.14818 if vol_m else 0.0
    base["cell_volume_A3"] = vol_A3

    # Final reported total force (last occurrence).
    forces = re.findall(r"Total force\s*=\s*([\d.eE+-]+)", text)
    if forces:
        base["force_total"] = float(forces[-1])

    # Converged total energy: line beginning with "!".
    e_m = re.search(r"^!\s+total energy\s*=\s*(-?\d+\.\d+)\s*Ry", text, re.MULTILINE)
    if e_m:
        base["total_energy_eV"] = float(e_m.group(1)) * RY_TO_EV
        # Freysoldt charged-supercell correction (reference-independent charge term).
        base["e_corr_FS_eV"] = freysoldt_correction(charge, vol_A3) if vol_A3 > 0 else 0.0
        base["status"] = "OK" if scf_converged else "OK_NO_CONV_FLAG"
    else:
        if scf_failed:
            base["status"] = "SCF_NOT_CONVERGED"
        elif not job_done:
            base["status"] = "NOT_DONE"
        else:
            base["status"] = "NO_ENERGY"
    return base


def compute_transitions(df_ok: pd.DataFrame) -> pd.DataFrame:
    """Per-(dopant,site) charge transition levels ε(0/+1), ε(+1/+2), reference-independent.

    ε(q/q') = [ (E_tot[q]+E_corr[q]) − (E_tot[q']+E_corr[q']) ] / (q' − q),  εF above VBM.
    Requires both charge states present for that (dopant, site).
    """
    rows = []
    for (dop, site), g in df_ok.groupby(["dopant", "site"]):
        e = {int(r.charge): r.total_energy_eV + r.e_corr_FS_eV for r in g.itertuples()}
        eps01 = (e[0] - e[1]) / (1 - 0) if (0 in e and 1 in e) else np.nan
        eps12 = (e[1] - e[2]) / (2 - 1) if (1 in e and 2 in e) else np.nan
        rows.append({"dopant": dop, "site": site,
                     "eps_0_1_eV": eps01, "eps_1_2_eV": eps12,
                     "n_charges": len(e)})
    return pd.DataFrame(rows)


def dominant_charge_at_eF(eps01: float, eps12: float, eF: float = ANCHOR_FERMI_EV) -> int:
    """Dominant V_O charge state at Fermi level eF (above VBM) from the transition levels.

    Donor V_O ladder: +2 below ε(+1/+2); +1 between ε(+1/+2) and ε(0/+1); 0 above ε(0/+1).
    (Lower formation energy of the more-positive state when εF is below the transition.)
    """
    if not np.isnan(eps12) and eF < eps12:
        return 2
    if not np.isnan(eps01) and eF < eps01:
        return 1
    return 0


def build_dopant_summary(df_ok: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
    """Per-dopant: dominant V_O site (lowest-E among sites, preferring q=0), the
    transition levels at that dominant site, charges done, and dominant charge at εF=4.0."""
    rows = []
    for dop in sorted(df_ok["dopant"].unique()):
        gd = df_ok[df_ok["dopant"] == dop]
        # Dominant site: lowest total energy at q=0 (neutral) if available, else lowest at
        # the lowest available charge — site total energies are comparable within a charge
        # state (same supercell composition, different O sublattice site).
        site_e = {}
        for site, gs in gd.groupby("site"):
            q0 = gs[gs["charge"] == 0]
            if len(q0):
                site_e[site] = float(q0["total_energy_eV"].iloc[0])
        if not site_e:  # no neutral; fall back to lowest-charge comparison
            qmin = int(gd["charge"].min())
            for site, gs in gd[gd["charge"] == qmin].groupby("site"):
                site_e[site] = float(gs["total_energy_eV"].iloc[0])
        dominant_site = min(site_e, key=site_e.get) if site_e else "—"

        trow = df_trans[(df_trans["dopant"] == dop) & (df_trans["site"] == dominant_site)]
        if len(trow):
            eps01 = float(trow["eps_0_1_eV"].iloc[0])
            eps12 = float(trow["eps_1_2_eV"].iloc[0])
        else:
            eps01 = eps12 = np.nan
        rows.append({
            "dopant": dop,
            "dominant_site": dominant_site,
            "eps_0_1_eV": round(eps01, 4) if not np.isnan(eps01) else np.nan,
            "eps_1_2_eV": round(eps12, 4) if not np.isnan(eps12) else np.nan,
            "n_charges_done": int(gd["charge"].nunique()),
            "dominant_charge_eF4.0": dominant_charge_at_eF(eps01, eps12),
        })
    return pd.DataFrame(rows)


def main():
    out_files = sorted(OUTPUT_DIR.glob("PBEU_v58_*.out"))
    print(f"Parsing {len(out_files)} gap-dopant V_O .out files in {OUTPUT_DIR}/\n")

    rows = [parse_one(f) for f in out_files]
    df = pd.DataFrame(rows)

    # ----- per-file transitions CSV (all 90 rows, OK + flagged) -----
    cols = ["dopant", "site", "charge", "total_energy_eV", "scf_converged",
            "e_corr_FS_eV", "cell_volume_A3", "force_total", "status"]
    trans_csv = RESULTS_DIR / "v58_gap_vo_transitions.csv"
    df[cols].sort_values(["dopant", "site", "charge"]).to_csv(trans_csv, index=False)

    n_total = len(df)
    n_ok = int((df["total_energy_eV"].notna()).sum())
    n_bad = n_total - n_ok
    print(f"[parse] {n_total} files: {n_ok} with a clean '! total energy' line, "
          f"{n_bad} without.")
    print(f"[status] {df['status'].value_counts().to_dict()}")
    if n_bad:
        print("\n[skipped — no clean converged energy] reason = SCF hit 200-iter cutoff:")
        for r in df[df["total_energy_eV"].isna()].itertuples():
            print(f"    {r.dopant:3s} {r.site:6s} q{r.charge}  -> {r.status}")

    df_ok = df[df["total_energy_eV"].notna()].copy()

    # ----- transition levels + dominant site -----
    df_trans = compute_transitions(df_ok)
    df_summary = build_dopant_summary(df_ok, df_trans)
    summ_csv = RESULTS_DIR / "v58_gap_dopant_summary.csv"
    df_summary.to_csv(summ_csv, index=False)

    print(f"\n[saved] {trans_csv}  ({n_total} rows)")
    print(f"[saved] {summ_csv}  ({len(df_summary)} dopants)")

    # ----- summary table -----
    print("\n=== PER-DOPANT V_O TRANSITION SUMMARY (ε above VBM=0; reference-independent) ===")
    print(f"{'dop':4s} {'dom_site':9s} {'eps(0/+1)':>10s} {'eps(+1/+2)':>11s} "
          f"{'n_q':>4s} {'q@eF4.0':>8s}")
    for r in df_summary.itertuples():
        e01 = f"{r.eps_0_1_eV:10.3f}" if not (isinstance(r.eps_0_1_eV, float) and np.isnan(r.eps_0_1_eV)) else f"{'NaN':>10s}"
        e12 = f"{r.eps_1_2_eV:11.3f}" if not (isinstance(r.eps_1_2_eV, float) and np.isnan(r.eps_1_2_eV)) else f"{'NaN':>11s}"
        qcol = getattr(r, "_6")  # dominant_charge_eF4.0
        print(f"{r.dopant:4s} {r.dominant_site:9s} {e01} {e12} "
              f"{r.n_charges_done:4d} {qcol:8d}")

    n_q2 = int((df_summary["dominant_charge_eF4.0"] == 2).sum())
    n_q1 = int((df_summary["dominant_charge_eF4.0"] == 1).sum())
    n_q0 = int((df_summary["dominant_charge_eF4.0"] == 0).sum())
    print(f"\n[dominant charge @ εF=4.0 eV]  V_O^+2: {n_q2}   V_O^+1: {n_q1}   V_O^0: {n_q0}")
    print("\n[NOT extractable] absolute per-dopant ΔE_f(V_O) — requires E[doped, no-vacancy]")
    print("                  reference cell which is NOT on disk (documented trap).")
    return df, df_trans, df_summary


if __name__ == "__main__":
    main()
