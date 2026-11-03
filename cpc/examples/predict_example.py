"""Run a label-free inventory prediction at a user-specified Si or Sn dose.

The default parameters demonstrate the interface; they are not a fitted model.
Run prepare_kroger_input.py and phase15_build_fermi.py before this example.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sandbox_workflow import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--element", choices=["Si", "Sn"], default="Si")
    parser.add_argument("--dose", type=float, default=1e18, help="Incorporated concentration in cm^-3")
    args = parser.parse_args()
    inputs = SimpleNamespace(element=args.element, incorporated_cm3=args.dose,
                             Tobs_K=300., growth_T_C=700., H_known=False,
                             H_cm3=0., scenario_pO2_atm=1e-5)
    coordinates = np.array([0., 16., 0., 0., 0., 0.])
    value, gradient, valid = Predictor("inventory").point(inputs, coordinates)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError("The example did not return finite predictions and derivatives")
    print(json.dumps({"example_parameters_not_fitted": True,
                      "inputs": vars(inputs), "log10_nH_cm3": float(value),
                      "parameter_gradient": gradient.tolist(),
                      "observation_valid": bool(valid)}, indent=2))


if __name__ == "__main__":
    main()
