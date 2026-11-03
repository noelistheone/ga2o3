"""Repair V1 diagnostic-dictionary rendering without changing any model fit."""
from pathlib import Path
import json
import numpy as np
import phase11_external_source as v1

original_raw_predict = v1.raw_predict


def raw_predict(fit, rows):
    field = v1.study.METHODS[fit["method"]]["field"]
    if field in ("ridge", "dose_line", "activation"):
        return original_raw_predict(fit, rows)
    vals, diagnostics = [], []
    for row in rows.itertuples():
        try:
            values, _, dg, extra = v1.study.point(row, fit["parameters"], fit["method"], fit["feature_metadata"])
            vals.append(values[0]); diagnostics.append({**dg, **extra})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            vals.append(np.nan); diagnostics.append(dict(error=repr(exc), gradient_valid=False))
    return np.asarray(vals), diagnostics


if __name__ == "__main__":
    p = v1.OUT / "diagnostic_render_fix_v2_protocol.json"
    payload = dict(version=2, created_utc=v1.energy.now(),
        code_sha256=v1.energy.sha(Path(__file__)), original_code_sha256=v1.energy.sha(Path(v1.__file__)),
        original_protocol_sha256=v1.energy.sha(v1.OUT / "evaluation_protocol.json"),
        fix="duplicate Tf_K in dict(**dg,**extra) caused TypeError after fitting;use dictionary merge;no numerical/model/protocol change",
        failure_log_sha256=v1.energy.sha(v1.energy.BASE / "phase11_external.log"),
        shared_training_reused="unchanged V1 full-five-source fits; no external labels")
    if p.exists():
        previous = json.loads(p.read_text())
        assert previous["code_sha256"] == payload["code_sha256"]
    else:
        v1.energy.write(p, payload)
    v1.raw_predict = raw_predict
    v1.main()
