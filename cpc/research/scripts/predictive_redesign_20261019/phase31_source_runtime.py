"""Explicit factory injection into the unchanged label-isolated source runtime."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/trainable_sandbox_20261017'))
import phase29_source_runtime as runtime
from sandbox.correlated_saved_model import CorrelatedSavedModel

# This changes the factory in this isolated process only, never an old file.
runtime.SavedJointModel=CorrelatedSavedModel

if __name__=='__main__':runtime.main()
