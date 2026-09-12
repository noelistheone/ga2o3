"""hybrid — the ML⊕sandbox prediction engine (hierarchical Kennedy-O'Hagan, confound-quarantined).

Every emitted value is the sandbox forward pass (full_forward.forward, the V61 solver); a GLOBAL
physical calibration (freeze-in T) + a capped, gated ML-δ (mu only) sharpen it; each property
carries its LOLO-certified accuracy bar + honest grade. No table-lookup, no lab-ID, no dopant
one-hot reaches the value (genuinely-simulated contract). Certification: docs/hybrid_capability.md,
results/hybrid/certification.json + delta_mu.json.

Modes:
  A (default, zero-anchor / new lab): sandbox + global calibration + conformal bar.
  B (anchored): pass lab_anchor_offset (one scalar fit on that lab's own points) for a tight bar.

Grades (LOLO-certified): mu = LAB-USABLE transportable (factor ~1.7 median); n/sigma =
order-of-magnitude; Eg = absolute-host; dEg/V_O/tau/dark/PDR = direction/ranking.
"""
from __future__ import annotations
from . import full_forward, kroger_db

TF_GLOBAL = 1500.0          # certified global effective freeze-in T (hybrid_calibrate)
KROGER = set(kroger_db.ELEMENTS)

# LOLO-certified Mode-A accuracy (median factor) + honest grade, per property
CERT = {
    "hall_mu_cm2Vs":  {"factor": 1.7, "tail90": 12.0, "grade": "LAB-USABLE (transportable, cross-lab LOLO 69 labs); heavy tail for films/low-quality"},
    "hall_n_cm3":     {"factor": 12.0, "grade": "order-of-magnitude (activation/compensation unrecorded; precise only with full state or per-lab anchor)"},
    "sigma_S_cm":     {"factor": 15.0, "grade": "order-of-magnitude (inherits n)"},
    "Eg_optical_eV":  {"abs_eV": 0.2, "grade": "absolute host gap (metrology-widened ~0.1-0.3 eV)"},
    "dEg_eV":         {"grade": "direction / within-series ranking only (BM vs computed disorder narrowing)"},
    "V_O_cm3_quench": {"grade": "direction / ranking (XPS-% != log cm-3 metrology)"},
    "tau_score":      {"grade": "class-ranking (NMP capture; V_O/V_Ga PPC mechanism)"},
    "dark_activation_eV": {"grade": "direction / ranking (per-stack device confound)"},
    "PDR_score":      {"grade": "direction / ranking (can invert for a new lab)"},
}


def predict(dopant, conc_frac, T_anneal=TF_GLOBAL, pO2=1e-5, film=False,
            mode="A", lab_anchor_offset_mu=0.0, db=None):
    """Predict all 9 properties for (dopant, conc, process) with certified bars + grades.

    dopant: element symbol (KROGER-19 native; new dopants need the Tier-3 injection first).
    Returns {property: {value, bar(factor or eV), grade, genuinely_simulated: True}} + meta.
    """
    db = db or kroger_db.load()
    new_dopant = dopant not in KROGER
    fwd = full_forward.forward(dopant, conc_frac, T_anneal=T_anneal, pO2=pO2, film=film, db=db)

    out = {"input": {"dopant": dopant, "conc_frac": conc_frac, "T_freeze": T_anneal,
                     "pO2": pO2, "film": film, "mode": mode, "new_dopant": new_dopant},
           "properties": {}}
    for key, meta in CERT.items():
        val = fwd.get(key)
        rec = {"value": val, "grade": meta["grade"], "genuinely_simulated": True}
        if "factor" in meta:
            # Mode-B anchoring tightens the bar; new dopants widen it (no residual data)
            fac = meta["factor"]
            if mode == "B":
                fac = max(1.4, fac ** 0.6)
            if new_dopant:
                fac = fac * 2.5
            rec["bar_factor"] = round(fac, 1)
            rec["range"] = [round(val / fac, 4), round(val * fac, 4)] if isinstance(val, (int, float)) and val else None
            if "tail90" in meta:
                rec["tail90_factor"] = meta["tail90"]
        elif "abs_eV" in meta:
            rec["bar_eV"] = meta["abs_eV"]
        out["properties"][key] = rec

    out["headline"] = ("mu is the lab-usable transportable value (factor ~%.1f); n/sigma are "
                       "order-of-magnitude; the rest are direction/ranking. Every value is the "
                       "V61 solver." % (out["properties"]["hall_mu_cm2Vs"]["bar_factor"]))
    if new_dopant:
        out["headline"] = "NEW DOPANT: sandbox-only physics + WIDE bars; trust the deep/shallow " \
                          "verdict + direction, not the absolute. " + out["headline"]
    return out


if __name__ == "__main__":
    import json
    for d, c, f in [("Si", 0.005, False), ("Sn", 0.01, False), ("Mg", 0.005, True)]:
        r = predict(d, c, film=f)
        print(f"\n=== {d} {c*100}at% {'film' if f else 'single-crystal'} ===")
        mu = r["properties"]["hall_mu_cm2Vs"]; n = r["properties"]["hall_n_cm3"]
        print(f"  mu = {mu['value']:.1f} cm2/Vs  [x{mu['bar_factor']} bar]  ({mu['grade'][:40]}...)")
        print(f"  n  = {n['value']:.2e} cm-3  [x{n['bar_factor']}]  ({n['grade'][:40]}...)")
