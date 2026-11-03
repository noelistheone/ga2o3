"""Check source-recomputed charge budgets in the independent runtime outputs."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase14_source_adaptation as bench
import phase2_energy as energy
from csv_io_v2 import read_result_csv


def main():
    out=bench.BASE/"research_runtime";index={r["task_id"]:r for r in json.loads((out/"index.json").read_text())}
    bank=bench.bank();source=read_result_csv(bench.OUT/"validation/predictions.csv").set_index(["variant","family","budget","sample_id"])
    errors=[];checked=0
    for row in json.loads((out/"independent_outputs.json").read_text())["rows"]:
        if row["values"] is None:continue
        meta=index[row["task_id"]];sample=row["obs_id"].removeprefix("RT::")
        bundle=json.loads((ROOT/meta["bundle"]).read_text())
        if meta["source_kind"]=="offset":
            f=bank[bundle["fit_key"]];wanted=next(r["charge_outputs"] for r in f["predictions"] if r["obs_id"]==row["obs_id"])
        else:
            r=source.loc[meta["model"],meta["held"],meta["budget"],sample];wanted=[r.logNabs,r.logNq2]
        errors.extend(abs(np.array(wanted)-np.array(row["values"][1:])));checked+=1
    assert checked==1827 and max(errors)<1e-5
    energy.write(out/"charge_budget_audit.json",dict(passed=True,rows=checked,maximum_charge_budget_error=max(errors),
        code_sha256=energy.sha(Path(__file__)),input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            (out/"index.json",out/"independent_outputs.json",bench.OUT/"validation/predictions.csv")},
        scope="same independent-process output,raw/base orsource-recomputedNabs/Nq2;not materialscatteringvalidation"))
    print("Independent charge budgets passed",checked,"rows;maxerror",max(errors))


if __name__=="__main__":main()
