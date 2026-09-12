"""R6: confront the Sb deep-level prediction with the OFZ transport experiment
(Li B. et al., Chin. Opt. Lett. 21, 041605 (2023): Hall n = 9.55e16 -> 8.10e18 cm^-3 rising
with Sb content, mu 153 -> 109 cm2/Vs, octahedral-site assignment).

Three solver runs over the experimental loading range at OFZ-like states:
  (a) our HSE alpha=0.25 Sb level (2+/0)=VBM+3.106 (production);
  (b) best-estimate level: alpha=0.33 + exact-anisotropic point charge = VBM+3.65;
  (c) the HYPOTHETICAL shallow donor the measured trend would require (level scan to find the
      shallowest (2+/0) depth below CBM at which predicted n(Sb) still reproduces the measured
      rise to ~8e18 at the top loading).
Writes results/tier3/sb_ofz_confrontation.json."""
import sys, json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db, new_dopant, full_forward as ff
from sandbox.kroger_db import N_SITE

db0 = kroger_db.load()
N_cat = N_SITE * 2.0 / 3.0
EG = 4.9

# loading range spanning the experiment (n up to 8.1e18 implies [Sb] >= ~1e19 if donor-active)
FRACS = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
STATES = [("OFZ_effective", 1500.0, 1e-4, 1e16), ("OFZ_air", 1500.0, 0.2, 1e16)]


def db_with_sb(eps_2p0_above_VBM):
    """extend the DB with an Sb_Ga defect whose (2+/0) sits at the given level (dHo offset-free)."""
    cs = {0: 3.0, 2: 3.0 - 2.0 * eps_2p0_above_VBM}   # A(2)-A(0) = -2*eps
    return new_dopant.extend_with_dopant(db0, "Sb", cs)


def n_curve(dbx, Tf, pO2, Nd_bg):
    out = []
    for f in FRACS:
        r = ff.forward("Sb", f, T_anneal=Tf, pO2=pO2, film=False, Nd_bg=Nd_bg,
                       db=dbx, undoped_cache={})
        out.append({"frac": f, "Sb_cm3": float(f"{f*N_cat:.3g}"),
                    "n_cm3": float(f"{r['hall_n_cm3']:.3g}"),
                    "mu": round(r["hall_mu_cm2Vs"], 1)})
    return out

res = {"experiment": {"source": "Li et al., Chin. Opt. Lett. 21, 041605 (2023), OFZ crystals",
                      "n_range_cm3": [9.55e16, 8.10e18], "mu_range": [153.1, 108.7],
                      "site": "octahedral Ga(II) (agrees with our computed site preference)"}}

for tag, eps in [("our_HSE_a025_level_3.106", 3.106), ("best_estimate_a033_aniso_3.65", 3.65)]:
    dbx = db_with_sb(eps)
    res[tag] = {st[0]: n_curve(dbx, st[1], st[2], st[3]) for st in STATES}
    top = res[tag]["OFZ_effective"][-1]["n_cm3"]
    print(tag, "n at top loading (effective state):", top, flush=True)

# (c) how shallow must the level be to reproduce the measured rise?
scan = []
for depth in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    dbx = db_with_sb(EG - depth)
    c = n_curve(dbx, 1500.0, 1e-4, 1e16)
    scan.append({"level_below_CBM_eV": depth, "n_at_1e-3": c[-1]["n_cm3"],
                 "n_at_3e-5": c[2]["n_cm3"]})
    print("depth", depth, "n_top", c[-1]["n_cm3"], flush=True)
res["required_level_scan"] = scan
ok = [s["level_below_CBM_eV"] for s in scan if s["n_at_1e-3"] >= 5e18]
res["read"] = {
    "measured_top_n": 8.1e18,
    "depths_reproducing_measured_n": ok,
    "conclusion": ("the measured n(Sb) rise requires an effective donor level within "
                   f"~{max(ok) if ok else '<0.1'} eV of the CBM; our computed Sb_Ga (2+/0) sits "
                   "1.2-1.3 eV below the gap-matched CBM with an error chain independently "
                   "validated at 0.04-0.13 eV on V_O -- the transport donor in those crystals "
                   "is inconsistent with isolated substitutional Sb_Ga as computed"),
}
(PROJ / "results/tier3/sb_ofz_confrontation.json").write_text(json.dumps(res, indent=1))
print("wrote results/tier3/sb_ofz_confrontation.json")
