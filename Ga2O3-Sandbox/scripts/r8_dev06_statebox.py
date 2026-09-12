"""R8 DEV-06 (blocking): convert the four-device point prediction into a regime prediction.
(1) Sweep the a-priori Ar-process state box (pO2 x T_anneal) and record, at EVERY state, the
    doped-vs-paired-intrinsic PDR direction the solver emits for each device dopant -- if the
    direction is invariant across the box, no post-outcome state choice can matter.
(2) Report the solver's dark-current directions for all four devices (the previously
    unreported half), scored against the corrected lab table
    (dark pA: Si 2.3 vs 7.9 DOWN, Sn 11.1 vs 11.0 flat/up, Mg 11.0 vs 13.0 DOWN,
     Zn 17.0 vs 7.9 UP; source Ga2O3-Net data/raw/experimental/ga2o3_exp.csv rows 480-487).
Writes results/tier2/r8_dev06_statebox.json.
"""
import sys, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db

db = kroger_db.load()
LAB = {"Si": 0.0234, "Sn": 0.0265, "Mg": 0.0101, "Zn": 0.0263}
LAB_DARK_DIR = {"Si": "down", "Sn": "flat", "Mg": "down", "Zn": "up"}   # pA, doped vs paired intrinsic
LAB_PDR_DIR = {X: "up" for X in LAB}

PO2 = [1e-16, 1e-13, 1e-10, 1e-7, 1e-4]
TF = [873.0, 1073.0, 1273.0]

out = {"state_box": {"pO2_atm": PO2, "T_anneal_K": TF},
       "per_dopant": {}, "production_state": {"T_anneal": 1073.0, "pO2": 1e-5}}
for X, c in LAB.items():
    grid = []
    for T in TF:
        for p in PO2:
            und = ff.forward("undoped", 0.0, T_anneal=T, pO2=p, film=True,
                             E_B_eV=0.03, Nd_bg=1e17, db=db, undoped_cache={})
            r = ff.forward(X, c, T_anneal=T, pO2=p, film=True, E_B_eV=0.03,
                           Nd_bg=1e17, db=db, undoped_cache={"n_und": und["hall_n_cm3"]})
            dpdr = r["PDR_score"] - und["PDR_score"]
            ddark_act = r["dark_activation_eV"] - und["dark_activation_eV"]
            grid.append({"T": T, "pO2": p,
                         "pdr_dir": "up" if dpdr > 0 else "down",
                         "dark_current_dir": "down" if ddark_act > 0 else "up",
                         "dPDR": round(float(dpdr), 3),
                         "ddark_act_eV": round(float(ddark_act), 4)})
    pdr_dirs = {g["pdr_dir"] for g in grid}
    dark_dirs = {g["dark_current_dir"] for g in grid}
    out["per_dopant"][X] = {
        "grid": grid,
        "pdr_direction_invariant": len(pdr_dirs) == 1,
        "pdr_direction": sorted(pdr_dirs),
        "pdr_matches_lab_everywhere": pdr_dirs == {LAB_PDR_DIR[X]},
        "dark_direction_set": sorted(dark_dirs),
        "lab_dark_dir": LAB_DARK_DIR[X],
    }
    print(X, "PDR dirs:", sorted(pdr_dirs), "invariant:", len(pdr_dirs) == 1,
          "| dark dirs:", sorted(dark_dirs), "lab:", LAB_DARK_DIR[X], flush=True)

# production-state dark directions (the previously unreported half)
und = ff.forward("undoped", 0.0, T_anneal=1073.0, pO2=1e-5, film=True, E_B_eV=0.03,
                 Nd_bg=1e17, db=db, undoped_cache={})
prod = {}
for X, c in LAB.items():
    r = ff.forward(X, c, T_anneal=1073.0, pO2=1e-5, film=True, E_B_eV=0.03,
                   Nd_bg=1e17, db=db, undoped_cache={"n_und": und["hall_n_cm3"]})
    d = r["dark_activation_eV"] - und["dark_activation_eV"]
    pred = "down" if d > 0 else "up"
    prod[X] = {"ddark_act_eV": round(float(d), 4), "pred_dark_current_dir": pred,
               "lab": LAB_DARK_DIR[X],
               "match": pred == LAB_DARK_DIR[X] or LAB_DARK_DIR[X] == "flat"}
    print("prod", X, prod[X], flush=True)
out["production_dark_directions"] = prod

(PROJ / "results/tier2/r8_dev06_statebox.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_dev06_statebox.json")
