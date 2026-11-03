"""Replay the 98 saved room-temperature outputs, without fitting or label access."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "research"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/predictive_redesign_20261019"))
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.relative_inventory_model import RelativeInventoryModel
from sandbox.coherent_fermi import published_coherent_profile
from si_multidonor_overlay_v1 import make_overlay


def main():
    results = ROOT / "results/predictive_redesign_20261019/proposal_followup_20261021_v1"
    data = pd.read_csv(ROOT / "results/trainable_sandbox_20261017/phase29_joint_scattering/data/scenarios49.csv")
    truth = pd.read_csv(BASE / "source_data/room_temperature_predictions.csv")
    db = make_overlay()[0]
    shared = ChargeFeasibleModel(basis="process", profile=published_coherent_profile(), db=db)
    references = {}
    fits = {}
    for p in (results / "overlay_nH_controls_v1/fits").glob("*.json"):
        item = json.loads(p.read_text())
        if item.get("held") is not None:
            fits[("FixedNHReference", item["held"])] = item["parameters"]
    for p in (results / "constrained_block_v1/primary_fits").glob("*.json"):
        item = json.loads(p.read_text())
        fits[("LearnedNHInventory", item["held"])] = item["parameters"]
    rows = {r.case_id: r for r in data.itertuples(index=False)}
    checks = []
    for record in truth.itertuples(index=False):
        row = rows[record.case_id]
        # Remove measured response fields before passing inputs to the solver.
        row = row._replace(target=np.nan, mu_cm2Vs=np.nan, carrier_cm3=np.nan, observed_log_nH=np.nan)
        w = np.asarray(fits[(record.variant, row.related_cluster)])
        if record.variant == "LearnedNHInventory":
            p = shared.point(row, w, require_observation=False)
        else:
            oxygen = float(row.scenario_pO2_atm)
            if oxygen not in references:
                references[oxygen] = RelativeInventoryModel(
                    dict(field="inventory"), profile=published_coherent_profile(), pO2_atm=oxygen, db=db)
            p = references[oxygen].point(row, np.r_[w, 0.], require_observation=False)
        error = abs(float(p.values[0]) - record.predicted)
        checks.append(dict(case_id=record.case_id, variant=record.variant,
                           absolute_replay_error_dex=error, passed=bool(error < 1e-9)))
    out = BASE / "verification"
    out.mkdir(exist_ok=True)
    report = dict(count=len(checks), passed=sum(r["passed"] for r in checks),
                  maximum_error_dex=max(r["absolute_replay_error_dex"] for r in checks), records=checks)
    (out / "inventory_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    if not all(r["passed"] for r in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
