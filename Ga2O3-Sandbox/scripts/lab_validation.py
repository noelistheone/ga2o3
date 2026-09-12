"""Pre-registered lab retrodiction — the decisive same-lab test.

The lab's Phase-105 same-lab anchor measures DEVICE FOMs only (PDR, dark current), which are
exactly the direction/ranking-only rung of our chain. Question: does the physics forward chain
predict the DIRECTION of PDR (doped vs its paired intrinsic) correctly for the annealed lab
dopants Si/Sn/Mg — where literature-ML scored PDR direction 1/4 (BACKWARDS) on this same bench?
Zn is abstained (unannealed → outside the equilibrium model's domain).

PRE-REGISTRATION (committed here BEFORE reading the solver output):
  H1: the sign of (PDR_score_doped − PDR_score_intrinsic) for each of Si, Sn, Mg.
  Lab ground truth (from the corrected table): ALL of Si/Sn/Mg RAISE PDR vs their paired
  intrinsic (Si 2.00→4.28, Sn 2.35→3.30, Mg 3.04→4.10).
  Pass bar (relative): beat the literature-ML baseline that got PDR direction 1/4 backwards —
  i.e. the physics chain must get ≥3/4-equivalent (here ≥2/3 of the annealed dopants) correct.
  Honest caveat: n=3 → 3/3 by chance is p=1/8; the corpus trend battery (separate) supplies the
  statistics. This is the case-study leg.
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db  # noqa: E402

db = kroger_db.load()

# lab dopants at their actual deposition concentrations (cation at%)
LAB = {"Si": 0.0234, "Sn": 0.0265, "Mg": 0.0101, "Zn": 0.0263}
LAB_PDR = {  # corrected Phase-105 table: doped PDR, paired-intrinsic PDR
    "Si": (4.2788, 2.00), "Sn": (3.2964, 2.3483),
    "Mg": (4.0969, 3.0414), "Zn": (4.2299, 3.3838)}
ANNEALED = ["Si", "Sn", "Mg"]   # Zn unannealed → abstain

# process params for the lab's RF-sputter + 800C Ar anneal (Ar ambient = low effective pO2)
PROC = dict(T_anneal=1073.0, pO2=1e-5, EcT_fraction=0.40, film=True, E_B_eV=0.03, Nd_bg=1e17)

# PRE-REGISTERED lab directions (ground truth from the table): all RAISE PDR
prereg = {X: "up" for X in LAB}   # lab: every dopant raises PDR vs its intrinsic

und = ff.forward("undoped", 0.0, db=db, **PROC)
cache = {"n_und": und["hall_n_cm3"]}

results = {}
for X, c in LAB.items():
    r = ff.forward(X, c, db=db, undoped_cache=cache, **PROC)
    dpdr = r["PDR_score"] - und["PDR_score"]
    ddark = r["dark_activation_eV"] - und["dark_activation_eV"]
    pred_dir = "up" if dpdr > 0 else "down"
    lab_dir = "up" if LAB_PDR[X][0] > LAB_PDR[X][1] else "down"
    results[X] = {
        "conc_at%": c * 100, "annealed": X in ANNEALED,
        "PDR_score_doped": round(r["PDR_score"], 2),
        "PDR_score_undoped": round(und["PDR_score"], 2),
        "delta_PDR_score": round(dpdr, 2),
        "predicted_PDR_direction": pred_dir,
        "lab_PDR_direction": lab_dir,
        "PDR_direction_correct": pred_dir == lab_dir,
        "delta_dark_activation_eV": round(ddark, 3),
        "n_cm3": f"{r['hall_n_cm3']:.2e}",
        "sigma_S_cm": f"{r['sigma_S_cm']:.2e}",
    }

annealed_correct = sum(results[X]["PDR_direction_correct"] for X in ANNEALED)
all_correct = sum(results[X]["PDR_direction_correct"] for X in LAB)

# DISCRIMINATION CHECK: does the chain ever predict PDR DOWN? (If it predicts "up" for every
# dopant, the lab 4/4 is a non-discriminating proxy artifact, not a validated prediction.)
all_dopants = [e for e in db.elements if e not in ("Ga", "O")]
up = down = 0
disc = {}
for X in all_dopants:
    try:
        r = ff.forward(X, 0.02, db=db, undoped_cache=cache, **PROC)
        d = r["PDR_score"] - und["PDR_score"]
        disc[X] = round(d, 2)
        up += d > 0
        down += d <= 0
    except Exception:
        pass
discrimination = {"n_up": up, "n_down": down,
                  "discriminates": down > 0,
                  "per_dopant_delta_PDR_score": disc,
                  "note": ("chain predicts BOTH up and down across dopants → discriminating"
                           if down > 0 else "chain predicts UP for ~all dopants → PDR proxy is "
                           "upward-biased; lab 4/4 is a weak (non-discriminating) test — the "
                           "discriminating statistics must come from the corpus trend battery")}

report = {
    "preregistered_lab_directions": prereg,
    "process_params": PROC,
    "results": results,
    "annealed_PDR_direction_score": f"{annealed_correct}/{len(ANNEALED)}",
    "all4_PDR_direction_score": f"{all_correct}/4",
    "discrimination_check": discrimination,
    "literature_ML_baseline": "PDR direction 1/4 (backwards) on this same bench (Phase-105)",
    "verdict": ("PHYSICS BEATS literature-ML on PDR direction"
                if annealed_correct >= 2 else "does NOT beat baseline"),
    "honest_caveat": "n=3 case study; 3/3 by chance p=1/8. Corpus trend battery supplies "
                     "statistics. Zn abstained (unannealed). PDR is direction-only fidelity.",
}
(PROJ / "results/tier2/lab_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
