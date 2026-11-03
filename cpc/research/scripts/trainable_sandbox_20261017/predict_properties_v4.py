"""Current local ML+Sandbox predictor; energy-information modes are explicit."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from sandbox.kroger_db import load
from sandbox.trainable_predictor_v4 import DopedPropertyPredictor

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,default=ROOT/"results/trainable_sandbox_20261017/predictor_bundle_v4")
    ap.add_argument("--dopant",required=True);ap.add_argument("--incorporated-cm3",type=float,required=True)
    ap.add_argument("--method",default="ridge_energy_tuned",choices=(
        "ridge_energy_tuned","goal_ridge_2","goal_mean_2","nonlinear_kernel","ordinary_kernel","ridge","full_library"))
    ap.add_argument("--anchors-json",type=Path);ap.add_argument("--po2-atm",type=float,default=None)
    ap.add_argument("--output",type=Path);a=ap.parse_args()
    try:
        p=DopedPropertyPredictor(load(),a.bundle)
        anchors=[] if a.anchors_json is None else json.loads(a.anchors_json.read_text())
        result=dict(status="conditional_research_prediction",**p.predict(
            a.dopant,a.incorporated_cm3,method=a.method,anchors=anchors,pO2_atm=a.po2_atm))
    except (ValueError,RuntimeError) as exc:result=dict(status="refused",reason=str(exc))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    else:print(text,end="")
    return 2 if result["status"]=="refused" else 0

if __name__=="__main__":raise SystemExit(main())
