"""Replay the frozen V2 code with explicitly typed Python CSV ingestion.

The fit code/protocol remains byte-identical. This wrapper avoids the observed
local C-reader crash on mixed boolean/numeric physical-validity columns.
"""
from pathlib import Path
import runpy
import pandas as pd
from csv_io import read_result_csv

if __name__=="__main__":
    pd.read_csv=read_result_csv
    runpy.run_path(str(Path(__file__).with_name("run_complex_energy_v2.py")),run_name="__main__")
