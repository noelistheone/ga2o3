"""Result reader preserving identifier/hash columns as exact strings.

A hex fit hash consisting only of digits and 'e' can look like a float and even
overflow to infinity. Primitive inference must never reinterpret identifiers.
Old frozen aggregations are preserved; this reader repairs metadata by loading
the original per-fold CSV files, not by guessing lost hash values.
"""
from pathlib import Path
import csv
import pandas as pd
from csv_io import read_result_csv as _read

TEXT_COLUMNS={
    "family","element","method","model","policy","role","scope","stratum",
    "training_elements","train_families","acquired_labels","failure","covariance",
}

def read_result_csv(path,**kwargs):
    out=_read(path,**kwargs)
    with Path(path).open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:return out
    for name in rows[0]:
        if (name in TEXT_COLUMNS or name.endswith(("_id","_key","_sha256"))
                or name in ("sha256","key")):
            out[name]=pd.Series([r[name] for r in rows],dtype=object)
    return out
