"""Public saved predictor with explicit reported-carbon scope disclosure."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sandbox.feasible_property_predictor_v3 import FeasiblePropertyPredictorV3


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'request', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--zero-anchor-gradient', action='store_true')
    a = p.parse_args(); model = FeasiblePropertyPredictorV3(a.bundle); request = json.loads(a.request.read_text())
    value = model.zero_anchor_gradient(**request) if a.zero_anchor_gradient else model.predict(**request)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('Saved prediction with carbon scope disclosure', a.output)


if __name__ == '__main__': main()
