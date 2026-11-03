"""Use state-rate saved-model factory in the unchanged isolated runtime."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/trainable_sandbox_20261017'))
import phase29_source_runtime as runtime
from sandbox.state_rate_saved import StateRateSavedModel
runtime.SavedJointModel=StateRateSavedModel
if __name__=='__main__':runtime.main()
