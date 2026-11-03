"""Same isolated source runtime, with only the declared tree factory replaced."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/trainable_sandbox_20261017'))
sys.path.insert(0,str(Path(__file__).parent))
import phase29_source_runtime as runtime
from strong_tree_saved import StrongTreeSavedModel
runtime.SavedJointModel=StrongTreeSavedModel
if __name__=='__main__':runtime.main()
