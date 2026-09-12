import sys
from pathlib import Path
import numpy as np
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db
db = kroger_db.load()
for elem in ["Ta", "Zr", "Ge"]:
    c_x = db.col(elem)
    rows = np.where(db.dm[:, c_x] != 0)[0]
    print(f"=== all {elem}-containing charge states ===")
    for did in sorted(set(db.cs_ID[rows].tolist())):
        sub = rows[db.cs_ID[rows] == did]
        sig = db.dm[sub[0]]
        nz = {db.elements[j]: int(sig[j]) for j in range(len(sig)) if sig[j] != 0}
        entries = sorted([(int(db.charge[i]), round(float(db.dHo[i]), 2)) for i in sub], reverse=True)
        print(f"  defect#{did} dm={nz}: {entries}")
