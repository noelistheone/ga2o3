import sys
from pathlib import Path
import numpy as np
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db
db = kroger_db.load()

for elem in ["Si", "Sn", "Fe", "Mg"]:
    c_x, c_ga = db.col(elem), db.col("Ga")
    mask = (db.dm[:, c_x] == 1) & (db.dm[:, c_ga] == -1) & (np.abs(db.dm).sum(axis=1) == 2)
    rows = np.where(mask)[0]
    print(f"\n=== {elem}_Ga : {rows.size} charge states across defects ===")
    for did in sorted(set(db.cs_ID[rows].tolist())):
        sub = rows[db.cs_ID[rows] == did]
        order = np.argsort(-db.charge[sub])
        sub = sub[order]
        entries = [(int(db.charge[i]), round(float(db.dHo[i]), 3)) for i in sub]
        print(f"  defect#{did}: {entries}")
        # per-defect transition levels (consecutive charges)
        qs = db.charge[sub]
        for i in range(len(sub) - 1):
            qh, ql = int(qs[i]), int(qs[i + 1])
            if qh != ql:
                eps = (db.dHo[sub[i]] - db.dHo[sub[i + 1]]) / (ql - qh)
                print(f"      eps({qh}/{ql}) = {eps:+.3f} above VBM = {5.0 - eps:.3f} below Ec(0K)")
