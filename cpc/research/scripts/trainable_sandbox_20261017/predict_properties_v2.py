"""Version2 CLI: energetic checks require only declared supplied entries."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.trainable_predictor_v2 import DopedPropertyPredictor
import predict_properties as cli

if __name__=="__main__":
    cli.DopedPropertyPredictor=DopedPropertyPredictor
    if "--bundle" not in sys.argv:
        sys.argv[1:1]=["--bundle",str(ROOT/"results/trainable_sandbox_20261017/predictor_bundle_v2")]
    raise SystemExit(cli.main())
