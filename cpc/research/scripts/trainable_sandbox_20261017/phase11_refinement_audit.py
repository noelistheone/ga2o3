"""Audit the full solver-cap sensitivity and its actual selection consequences."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase11_prediction_audit as audit
import phase11_external_source as ext
import phase2_energy as energy

PARENT=energy.BASE/"phase11_parent_allocation"
OUT=PARENT/"solver_budget_sensitivity"


def main():
    assert (OUT/"execution.json").exists(),"Finish common-cap training first"
    p=json.loads((OUT/"sensitivity_protocol.json").read_text())
    assert energy.sha(ROOT/"scripts/trainable_sandbox_20261017/phase11_solver_budget.py")==p["code_sha256"]
    for name,sha in p["parent_fit_sha256"].items():assert energy.sha(PARENT/"fits"/name)==sha
    audit.OUT=OUT;audit.main()
    losses=[]
    for name,sha in p["parent_fit_sha256"].items():
        v1=json.loads((PARENT/"fits"/name).read_text());v2=json.loads((OUT/"fits"/name).read_text())
        if name[:-5] not in p["affected_keys"]:
            assert energy.sha(OUT/"fits"/name)==sha;continue
        a=min(t["loss"] for t in v1["trials"] if t.get("loss") is not None)
        b=min(t["loss"] for t in v2["trials"] if t.get("loss") is not None)
        losses.append(dict(fit_key=v1["key"],method=v1["method"],original_loss=a,extended_loss=b,
            loss_improvement=a-b,not_worse=b<=a+1e-8))
    energy.csvwrite(OUT/"objective_comparison.csv",losses)
    selections=[]
    for method in json.loads((PARENT/"prediction_protocol.json").read_text())["methods"]:
        first=json.loads((PARENT/"methods"/method/"selections.json").read_text())
        second=json.loads((OUT/"methods"/method/"selections.json").read_text())
        for v1,v2 in zip(first,second):
            assert (v1["held"],v1["budget"])==(v2["held"],v2["budget"])
            selections.append(dict(method=method,held=v1["held"],budget=v1["budget"],
                original_config=v1["selected"]["config"],extended_config=v2["selected"]["config"],
                original_lambda=v1["selected"]["lam"],extended_lambda=v2["selected"]["lam"],
                config_changed=v1["selected"]["config"]!=v2["selected"]["config"],
                lambda_changed=v1["selected"]["lam"]!=v2["selected"]["lam"],
                original_outer_MAE=v1["score"],extended_outer_MAE=v2["score"]))
    energy.write(OUT/"selection_comparison.json",selections)
    # This is a deployment selection check, using no new-source responses.
    ext.PARENT=OUT;ext.OUT=OUT/"external_selection_check"
    new=ext.original_cv_selections()
    original=json.loads((PARENT/"external_PRB2024/training_only_selections.json").read_text())
    changed=[]
    for v1,v2 in zip(original,new):
        assert (v1["method"],v1["budget"])==(v2["method"],v2["budget"])
        b=energy.clean(v2["selected"])
        if v1["selected"]["config"]!=b["config"] or v1["selected"]["lam"]!=b["lam"]:
            changed.append(dict(method=v1["method"],budget=v1["budget"],original=v1["selected"],extended=b))
    payload=dict(passed=all(r["not_worse"] for r in losses),created_utc=energy.now(),
        affected_records=len(losses),original_artifacts_unchanged=True,
        objective_worse_records=sum(not r["not_worse"] for r in losses),
        changed_outer_config_choices=sum(r["config_changed"] for r in selections),
        changed_outer_anchor_lambda_choices=sum(r["lambda_changed"] for r in selections),
        changed_external_deployment_choices=changed,
        external_action="V1 deployed choices unchanged; converged full-source fits and predictions remain valid" if not changed else "deploy refined training-only choices in separate V3 results; retain V1/V2",
        scope="common optimizer-budget sensitivity,not additional physical measurements or a different learned relation")
    energy.write(OUT/"sensitivity_audit.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
