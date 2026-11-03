"""CLI for the local trained ML+Sandbox research predictor."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from sandbox.kroger_db import load
from sandbox.trainable_predictor import DopedPropertyPredictor

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,default=ROOT/"results/trainable_sandbox_20261017/predictor_bundle_v1")
    ap.add_argument("--dopant",required=True)
    ap.add_argument("--incorporated-cm3",type=float,required=True)
    ap.add_argument("--method",default="nonlinear_kernel",
                    choices=("nonlinear_kernel","ordinary_kernel","ridge","full_library"))
    ap.add_argument("--anchors-json",type=Path,help="JSON list of0/1/2 same-condition chemical-dose/Hall anchors")
    ap.add_argument("--po2-atm",type=float,default=None,help="effective O2 scenario; missing calibration inputs remain unknown")
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    try:
        predictor=DopedPropertyPredictor(load(),a.bundle)
        anchors=[] if a.anchors_json is None else json.loads(a.anchors_json.read_text())
        result=predictor.predict(a.dopant,a.incorporated_cm3,method=a.method,
                                 anchors=anchors,pO2_atm=a.po2_atm)
        result=dict(status="conditional_research_prediction",**result)
    except (ValueError,RuntimeError) as exc:
        result=dict(status="refused",reason=str(exc))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    else:print(text,end="")
    return 2 if result["status"]=="refused" else 0

if __name__=="__main__":raise SystemExit(main())
