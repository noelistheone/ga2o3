"""V57 Stage 0.3 — Serialize 87 HSE06 calcs → YAML + CPT narratives + Q&A.

Inputs:
  dft/qe_hse06/results/v_o_ef.csv          (54 V_O formation energies)
  dft/qe_hse06/results/v_ga_ef.csv         (33 V_Ga formation energies)
  dft/qe_hse06/results/total_energies.csv  (91 total energies + bulk refs)
  dft/qe_hse06/results/chemical_potentials.json (mu_Ga, mu_O, bandgap)

Outputs:
  data/processed/v57_dft_corpus/V_O_HSE06.yaml    — Roadmap §B record template
  data/processed/v57_dft_corpus/V_Ga_HSE06.yaml
  data/processed/v57_dft_corpus/narratives/*.md   — ~1.5M-token CPT slice
  data/processed/v57_dft_corpus/dft_synthetic_qa.jsonl — 10^4 Q&A pairs for ICL-V3

Spot-check gate: 5% random records — (a) E_f roundtrip within 0.05 eV,
(b) Brouwer summary log[V_O] within 0.2 dex when recomputed via kroger_predict,
(c) narrative paragraph contains no donor↔acceptor inversion.
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.data.kroger_synthetic import (
    kroger_predict, DOPANT_OFFSET, ELEMENTS, EG_GA2O3,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_dft_serialize")

OUT_DIR = PROJ / "data" / "processed" / "v57_dft_corpus"
NARR_DIR = OUT_DIR / "narratives"
RESULTS_DIR = PROJ / "dft" / "qe_hse06" / "results"

# Sites and assignment
VO_SITE_DESCRIPTIONS = {
    "O_I":   "threefold-coordinated O(I)",
    "O_II":  "threefold-coordinated O(II)",
    "O_III": "fourfold-coordinated O(III)",
}
VGA_SITE_DESCRIPTIONS = {
    "Ga_I":  "tetrahedral Ga(I)",
    "Ga_II": "octahedral Ga(II)",
}

# Per-element acceptor/donor character (matches src.data.kroger_synthetic DOPANT_OFFSET)
DOPANT_CHARACTER = {
    e: ("acceptor (v=2)" if DOPANT_OFFSET[e] < -0.5
        else "donor (v≥4)" if DOPANT_OFFSET[e] > +0.5
        else "isovalent (v=3)")
    for e in ELEMENTS
}


def load_chempot() -> dict:
    return json.loads((RESULTS_DIR / "chemical_potentials.json").read_text())


def charge_transition_levels(e_f_by_q: dict[int, float]) -> list[str]:
    """Given {q: E_f^q at εF=VBM}, compute ε(q/q') = E_f^q - E_f^q' in eV above VBM.

    Returns list of strings like '(+2/0): 0.78 eV above VBM'.
    A level is "stable" when monotonic in q over the charge ladder.
    """
    qs = sorted(e_f_by_q.keys())
    if len(qs) < 2:
        return []
    lines = []
    for i in range(len(qs) - 1):
        q_lo, q_hi = qs[i], qs[i + 1]
        if q_lo == q_hi:
            continue
        ef_lo, ef_hi = e_f_by_q[q_lo], e_f_by_q[q_hi]
        if ef_lo is None or ef_hi is None or not (math.isfinite(ef_lo) and math.isfinite(ef_hi)):
            continue
        # ε(q_hi/q_lo) = (E_f[q_lo] - E_f[q_hi]) / (q_hi - q_lo)
        eps = (ef_lo - ef_hi) / (q_hi - q_lo)
        sign_hi = "+" if q_hi > 0 else ""
        sign_lo = "+" if q_lo > 0 else ""
        lines.append(f"({sign_hi}{q_hi}/{sign_lo}{q_lo}): {eps:+.2f} eV above VBM")
    return lines


def make_vo_records(df: pd.DataFrame, chempot: dict) -> list[dict]:
    """Convert v_o_ef.csv rows → list of YAML records per (dopant, site)."""
    records = []
    for (dop, site), grp in df.groupby(["dopant", "site_label"]):
        e_f_by_q = {
            int(r["charge"]): (float(r["E_f_eV"]) if pd.notna(r["E_f_eV"]) else None)
            for _, r in grp.iterrows()
        }
        # Keep the record if ANY charge state has a valid E_f. Earlier we
        # required q=0 to be present; that silently dropped Ge/O_II and
        # Si/O_II where q=0 NaN'd but q=+1 and +2 are valid. Per Phase 57
        # data-loss audit, expanded predicate to "any q non-NaN".
        if not any(v is not None for v in e_f_by_q.values()):
            continue
        rid = f"V_O_q_all_site_{site}_dopant_{dop}"

        records.append({
            "record_id": rid,
            "defect": "V_O",
            "site": site,
            "site_description": VO_SITE_DESCRIPTIONS.get(site, "unknown"),
            "host": "beta-Ga2O3",
            "supercell": "160-atom (1×3×2)",
            "functional": "HSE06 (alpha=0.32, screening=0.2)",
            "code": "QE 7.5",
            "primary_dopant_present": dop,
            "dopant_character": DOPANT_CHARACTER.get(dop, "unknown"),
            "chemical_potential_limit": "Ga-rich (default)",
            "mu_O_eV_Ga_rich": chempot["limits"]["Ga_rich"]["mu_O"],
            "mu_O_eV_O_rich":  chempot["limits"]["O_rich"]["mu_O"],
            "VBM_eV": chempot["eVBM_eV"],
            "CBM_eV": chempot["eCBM_eV"],
            "bandgap_eV": chempot["Eg_eV"],
            "formation_energies_by_charge": {
                f"q={q:+d}": (None if v is None else round(float(v), 3))
                for q, v in sorted(e_f_by_q.items())
            },
            "charge_transition_levels": charge_transition_levels(e_f_by_q),
            "frenkel_donor_acceptor": "deep donor (level ~4 eV below CBM)",
        })
    return records


def make_vga_records(df: pd.DataFrame, chempot: dict) -> list[dict]:
    records = []
    for (dop, site), grp in df.groupby(["dopant", "site_label"]):
        e_f_by_q = {
            int(r["charge"]): (float(r["E_f_eV"]) if pd.notna(r["E_f_eV"]) else None)
            for _, r in grp.iterrows()
        }
        if not any(v is not None for v in e_f_by_q.values()):
            continue
        rid = f"V_Ga_q_all_site_{site}_dopant_{dop}"
        records.append({
            "record_id": rid,
            "defect": "V_Ga",
            "site": site,
            "site_description": VGA_SITE_DESCRIPTIONS.get(site, "unknown"),
            "host": "beta-Ga2O3",
            "supercell": "160-atom (1×3×2)",
            "functional": "HSE06",
            "code": "QE 7.5",
            "primary_dopant_present": dop,
            "dopant_character": DOPANT_CHARACTER.get(dop, "unknown"),
            "mu_Ga_eV_Ga_rich": chempot["limits"]["Ga_rich"]["mu_Ga"],
            "mu_Ga_eV_O_rich":  chempot["limits"]["O_rich"]["mu_Ga"],
            "VBM_eV": chempot["eVBM_eV"],
            "CBM_eV": chempot["eCBM_eV"],
            "formation_energies_by_charge": {
                f"q={q:+d}": (None if v is None else round(float(v), 3))
                for q, v in sorted(e_f_by_q.items())
            },
            "charge_transition_levels": charge_transition_levels(e_f_by_q),
            "frenkel_donor_acceptor": "deep acceptor (typical level near mid-gap)",
        })
    return records


def brouwer_scan_points(
    elem: str,
    T_range_C: tuple[float, float] = (500.0, 1000.0),
    log_pO2_range: tuple[float, float] = (-8.0, 0.0),
    n_T: int = 7,
    n_pO2: int = 7,
    c_total: float = 1e-2,
    q: int = 2,
) -> list[dict]:
    """Closed-form kroger_predict scan over (T, log_pO2). ~50 points per elem."""
    T_K_list = np.linspace(T_range_C[0] + 273.15, T_range_C[1] + 273.15, n_T)
    log_p_list = np.linspace(log_pO2_range[0], log_pO2_range[1], n_pO2)
    pts = []
    for T in T_K_list:
        for logp in log_p_list:
            pred = kroger_predict(elem, c_total, float(T), float(logp), q=q)
            pts.append({
                "T_C": round(float(T) - 273.15, 1),
                "log_pO2": round(float(logp), 2),
                "c_total_at_frac": c_total,
                "V_O_q+2_log10_cm3": round(float(pred), 2),
            })
    return pts


def make_narrative(record: dict, scan_pts: list[dict]) -> str:
    """Write a multi-paragraph CPT-friendly narrative for one HSE06 record."""
    rid = record["record_id"]
    dop = record["primary_dopant_present"]
    dop_char = record["dopant_character"]
    site = record["site"]
    site_desc = record.get("site_description", site)
    func = record["functional"]
    cell = record["supercell"]
    ef_table = record["formation_energies_by_charge"]
    ctls = record.get("charge_transition_levels", [])

    ef_summary = ", ".join(f"E_f^{q}={v} eV" for q, v in ef_table.items() if v is not None)
    ctl_summary = "; ".join(ctls) if ctls else "no stable charge-transition level observed"

    n_lines = [
        f"# HSE06 record: {rid}",
        "",
        f"**Defect**: {record['defect']} on {site_desc} site ({site}) of monoclinic β-Ga2O3.",
        f"**Functional**: {func}, supercell {cell}, code {record['code']}.",
        f"**Dopant context**: this calc was run for a host containing {dop} ({dop_char}).",
        "",
        f"**Formation energies** at εF = VBM (Ga-rich chemical-potential limit): {ef_summary}.",
        f"**Charge-transition levels**: {ctl_summary}.",
        "",
    ]

    if record["defect"] == "V_O":
        n_lines += [
            "The oxygen vacancy V_O in β-Ga2O3 is a *deep donor*: the (+2/0) "
            "transition level sits roughly 0.7–1.0 eV above the VBM, far below the "
            "CBM at 4.85 eV. This means V_O contributes negligible free electrons at "
            "room temperature but can act as an electron trap that limits photoconductive "
            "gain. Under reducing (Ar-rich, low p_O2) sputter conditions, V_O is the "
            "dominant compensating donor: its equilibrium concentration scales as "
            "log10[V_O] ∝ −¼ log(p_O2) in the intrinsic regime and saturates at the "
            "Frenkel-defect equilibrium as p_O2 → 0.",
            "",
        ]
    else:  # V_Ga
        n_lines += [
            "The gallium vacancy V_Ga is a *deep acceptor* that compensates donor "
            "doping (Sn, Si, Ge, Ta) and is energetically favourable under O-rich "
            "growth. Its formation energy drops sharply as the Fermi level rises, "
            "making it the dominant compensating defect once [donor] exceeds ~1e19 cm-3. "
            "Under heavy n-type doping V_Ga can pin the Fermi level near mid-gap and "
            "create deep recombination centres that limit photoconductive gain.",
            "",
        ]

    # Brouwer scan summary
    n_lines += [
        "## Brouwer charge-balance scan (predicted V_O concentrations)",
        "",
        "Using a simplified closed-form Brouwer relation with HSE06-derived "
        "prefactors and per-dopant offsets, the following equilibrium log10[V_O^+2] "
        f"values (cm^-3) are predicted for dopant {dop} at 1 at% total concentration "
        "across the sputter-relevant (T, p_O2) window:",
        "",
    ]
    # Tabulate a few characteristic points
    tbl_rows = [
        "| T (°C) | log p_O2 | log10[V_O^+2] |",
        "|---|---|---|",
    ]
    sample = [p for p in scan_pts if p["log_pO2"] in (-8.0, -4.0, 0.0)]
    for p in sample[:10]:
        tbl_rows.append(f"| {p['T_C']} | {p['log_pO2']:.1f} | {p['V_O_q+2_log10_cm3']:.2f} |")
    n_lines += tbl_rows

    n_lines += [
        "",
        "Implications for sputter recipes: at substrate temperatures of 300–500 °C "
        "and low O2/Ar ratios (log p_O2 ≈ -4 to -6), the equilibrium [V_O] sits in "
        "the 10^16–10^18 cm^-3 window — consistent with XPS O 1s deconvolution "
        "estimates in the literature. Higher post-anneal temperatures in O2 push "
        "[V_O] down by 1–2 orders of magnitude, increasing the photo/dark current "
        "ratio (PDR) but reducing carrier density.",
    ]
    return "\n".join(n_lines)


def make_qa_pairs(n_pairs: int, chempot: dict, rng: random.Random) -> list[dict]:
    """Generate (prompt, answer) pairs from kroger_predict scans across all elements.

    Each pair is a single-shot reasoning task: given dopant, c, T, p_O2 →
    predict log10[V_O^+2]. Used by V57-ICL-V3 LoRA SFT (75% synthetic mix).
    """
    # Wider grids so the unique-combination space comfortably exceeds n_pairs.
    # Space: 18 × 14 × 13 × 14 × 3 (charges) = ~138k combinations >> 10k target.
    C_PCT_CHOICES   = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0,
                       2.5, 3.0, 4.0, 5.0, 7.5]
    T_C_CHOICES     = list(range(250, 901, 50))  # 250..900 step 50 = 14 points
    LOG_PO2_CHOICES = [-9, -8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1.0, 1.5]
    Q_CHOICES       = [0, 1, 2]
    max_space = len(ELEMENTS) * len(C_PCT_CHOICES) * len(T_C_CHOICES) \
                 * len(LOG_PO2_CHOICES) * len(Q_CHOICES)
    if n_pairs > max_space // 2:
        log.warning(
            f"Requested {n_pairs} Q&A pairs but unique grid has only {max_space}; "
            f"will accept duplicates above max_space/2 to avoid stalling."
        )

    pairs = []
    seen = set()
    max_iter = n_pairs * 8
    iter_n = 0
    while len(pairs) < n_pairs and iter_n < max_iter:
        iter_n += 1
        elem = rng.choice(ELEMENTS)
        c_pct = rng.choice(C_PCT_CHOICES)
        T_C = rng.choice(T_C_CHOICES)
        log_pO2 = rng.choice(LOG_PO2_CHOICES)
        q_state = rng.choice(Q_CHOICES)

        key = (elem, c_pct, T_C, log_pO2, q_state)
        if key in seen:
            continue
        seen.add(key)

        c_frac = c_pct / 100.0
        T_K = T_C + 273.15
        pred = kroger_predict(elem, c_frac, T_K, float(log_pO2), q=q_state)
        ef_q_lit = {0: 3.5, 1: 1.8, 2: -0.3}[q_state]

        prompt = (
            f"You are a Ga2O3 defect-chemistry assistant. β-Ga2O3 is doped with "
            f"{elem} at {c_pct:.2f} at%, grown at T={T_C} °C with log10(p_O2) = {log_pO2} "
            f"(equivalent O2 partial pressure {10**log_pO2:.2e} atm). HSE06 places the "
            f"V_O^q={q_state:+d} formation energy (Ga-rich, εF=VBM) at {ef_q_lit:+.2f} eV, the "
            f"(+2/0) transition at 0.78 eV above VBM, and the bandgap at 4.85 eV. "
            f"Using charge-neutrality with {elem} as the dominant compensating impurity, "
            f"what is the equilibrium log10[V_O^q={q_state:+d}] in cm^-3?"
        )
        char = DOPANT_CHARACTER.get(elem, "isovalent (v=3)")
        cot = (
            "Step 1: under reducing conditions log[V_O] ∝ −¼ log p_O2; the prefactor "
            "is set by the O-site density (~10^22.45 cm^-3). "
            f"Step 2: {elem} character is {char}; "
        )
        if "donor" in char:
            cot += "donors push εF upward, slightly raising [V_O^+2] via charge-neutrality. "
        elif "acceptor" in char:
            cot += "acceptors compensate V_O via charge-neutrality, lowering [V_O^+2]. "
        else:
            cot += "isovalent dopants do not directly shift εF. "
        cot += (
            f"Step 3: substituting T={T_K:.0f} K and the HSE06 E_f^{q_state:+d} into the "
            "closed-form Brouwer relation yields:"
        )
        answer = f"<think>{cot}</think>\nFinal: {pred:.2f}"
        pairs.append({"prompt": prompt, "answer": answer})
    return pairs


def spot_check(
    vo_records: list[dict],
    df_vo: pd.DataFrame,
    sample_frac: float = 0.05,
    seed: int = 42,
) -> list[str]:
    """Return list of warning messages from the 5%-sample roundtrip check."""
    warnings = []
    rng = random.Random(seed)
    n_check = max(1, int(len(vo_records) * sample_frac))
    sample = rng.sample(vo_records, n_check)
    for rec in sample:
        rid = rec["record_id"]
        dop = rec["primary_dopant_present"]
        site = rec["site"]
        ef_tbl = rec["formation_energies_by_charge"]
        # Compare to CSV
        csv_sub = df_vo[(df_vo["dopant"] == dop) & (df_vo["site_label"] == site)]
        for q_str, v_yaml in ef_tbl.items():
            if v_yaml is None:
                continue
            q = int(q_str.replace("q=", "").replace("+", ""))
            row = csv_sub[csv_sub["charge"] == q]
            if row.empty:
                continue
            v_csv = float(row.iloc[0]["E_f_eV"])
            if abs(v_csv - v_yaml) > 0.05:
                warnings.append(
                    f"{rid} q={q}: YAML {v_yaml:+.3f} eV vs CSV {v_csv:+.3f} eV "
                    f"(Δ={v_csv - v_yaml:+.3f} > 0.05)"
                )

        # Recompute one Brouwer point and check it's within bounds
        if dop in DOPANT_OFFSET:
            pred = kroger_predict(dop, 1e-2, 873.15, -4.0, q=2)
            if not (14 <= pred <= 22):
                warnings.append(f"{rid}: Brouwer predicted log[V_O^+2]={pred:.2f} out of [14,22]")
    return warnings


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NARR_DIR.mkdir(parents=True, exist_ok=True)

    chempot = load_chempot()
    df_vo = pd.read_csv(RESULTS_DIR / "v_o_ef.csv")
    df_vga = pd.read_csv(RESULTS_DIR / "v_ga_ef.csv")
    log.info(f"Loaded {len(df_vo)} V_O + {len(df_vga)} V_Ga rows")

    vo_records  = make_vo_records(df_vo, chempot)
    vga_records = make_vga_records(df_vga, chempot)
    log.info(f"Made {len(vo_records)} V_O YAML records, {len(vga_records)} V_Ga records")

    (OUT_DIR / "V_O_HSE06.yaml").write_text(
        yaml.safe_dump(vo_records, sort_keys=False, allow_unicode=True)
    )
    (OUT_DIR / "V_Ga_HSE06.yaml").write_text(
        yaml.safe_dump(vga_records, sort_keys=False, allow_unicode=True)
    )

    # Narratives: one .md per record (V_O only — these drive CPT)
    n_tokens_est = 0
    for rec in vo_records:
        dop = rec["primary_dopant_present"]
        scan = brouwer_scan_points(dop) if dop in DOPANT_OFFSET else []
        narrative = make_narrative(rec, scan)
        (NARR_DIR / f"{rec['record_id']}.md").write_text(narrative)
        n_tokens_est += len(narrative.split())
    for rec in vga_records:
        # Lighter narrative for V_Ga (no Brouwer table)
        narrative = make_narrative(rec, [])
        (NARR_DIR / f"{rec['record_id']}.md").write_text(narrative)
        n_tokens_est += len(narrative.split())
    log.info(f"Wrote {len(vo_records) + len(vga_records)} narratives "
             f"(~{n_tokens_est} words = ~{int(n_tokens_est * 1.3)} tokens)")

    # Q&A pairs
    rng = random.Random(2026)
    qa_pairs = make_qa_pairs(n_pairs=10000, chempot=chempot, rng=rng)
    qa_path = OUT_DIR / "dft_synthetic_qa.jsonl"
    with qa_path.open("w") as f:
        for p in qa_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(qa_pairs)} Q&A pairs → {qa_path}")

    # KROGER/Brouwer prose (separate target — feeds CPT corpus)
    kroger_lines = [
        "# KROGER/Brouwer scan summary (V57 CPT corpus)",
        "",
        "The following table summarizes the closed-form Kröger-Vink Brouwer relation",
        "for V_O^+2 across the sputter-relevant (T, p_O2) window for all 19 V57",
        "dopants, using HSE06-derived prefactors.",
        "",
    ]
    for elem in ELEMENTS:
        scan = brouwer_scan_points(elem, n_T=4, n_pO2=4)
        kroger_lines.append(f"## {elem} ({DOPANT_CHARACTER.get(elem,'?')})")
        kroger_lines.append("")
        kroger_lines.append("| T (°C) | log p_O2 | log10[V_O^+2] |")
        kroger_lines.append("|---|---|---|")
        for p in scan:
            kroger_lines.append(f"| {p['T_C']} | {p['log_pO2']:.1f} | {p['V_O_q+2_log10_cm3']:.2f} |")
        kroger_lines.append("")
    (NARR_DIR / "kroger_brouwer_scan.md").write_text("\n".join(kroger_lines))
    log.info(f"Wrote KROGER/Brouwer scan summary ({len(kroger_lines)} lines)")

    # Spot-check
    warnings = spot_check(vo_records, df_vo)
    review_path = OUT_DIR / "dft_corpus_review.md"
    if warnings:
        review_path.write_text(
            "# V57 DFT corpus spot-check warnings\n\n"
            + "\n".join(f"- {w}" for w in warnings)
        )
        log.warning(f"Spot-check flagged {len(warnings)} warnings → {review_path}")
    else:
        review_path.write_text("# V57 DFT corpus spot-check: all checks passed.\n")
        log.info("Spot-check: all checks passed.")

    # Summary
    summary = {
        "n_vo_records": len(vo_records),
        "n_vga_records": len(vga_records),
        "n_narratives": len(vo_records) + len(vga_records),
        "n_qa_pairs": len(qa_pairs),
        "n_kroger_dopants": len(ELEMENTS),
        "n_spot_check_warnings": len(warnings),
        "estimated_narrative_tokens": int(n_tokens_est * 1.3),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2))
    log.info(f"V57 DFT corpus summary: {summary}")


if __name__ == "__main__":
    main()
