"""Small-result CSV ingestion using Python's standard library.

Used for bounded research result files after local pandas reader crashes.
Parsing and primitive numeric conversion happen in Python; it is deliberately
not a general replacement for the pandas read_csv API.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

def read_result_csv(path,**kwargs):
    supported={"engine","index_col","dtype"}
    if set(kwargs)-supported:
        raise TypeError("Unsupported CSV option: "+str(set(kwargs)-supported))
    with Path(path).open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:return pd.DataFrame()
    columns={}
    for name in rows[0]:
        values=[r[name] for r in rows]
        populated=[v for v in values if v!=""]
        if populated and all(v in ("True","False") for v in populated):
            columns[name]=[v=="True" if v else np.nan for v in values]
            continue
        normalized=[{"True":"1","False":"0"}.get(v,v) for v in values]
        try:
            parsed=[float(v) if v else np.nan for v in normalized]
        except (TypeError,ValueError):
            parsed=[v if v else np.nan for v in values]
        columns[name]=parsed
    return pd.DataFrame(columns)
