"""Fetch the pinned upstream input locally; never redistribute it in this ZIP."""
from pathlib import Path
import hashlib
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "research"
COMMIT = "b0b8ce9d092d7208410d31b556d96ff37efd3b86"
NAME = "Ga2O3_Varley_all_defects_new_092724.mat"
URL = f"https://raw.githubusercontent.com/mikescarpulla/KROGER/{COMMIT}/Ga2O3/{NAME}"
EXPECTED = "0a7508c2682d206954af16b125ca948aaf3bbfc843de252effafd3e9d79d3a43"


def main():
    p = ROOT / "external/KROGER/Ga2O3" / NAME
    if p.is_file():
        blob = p.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "MLST-reproduction/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            blob = response.read()
    if hashlib.sha256(blob).hexdigest() != EXPECTED:
        raise RuntimeError("Upstream input differs from the article's recorded MAT checksum.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    subprocess.run([sys.executable, str(ROOT / "scripts/t1_extract_kroger.py")], cwd=ROOT, check=True)
    q = ROOT / "results/phase0/kroger_ga2o3_092724_numeric.npz"
    print("Prepared locally:", q)
    print("NPZ SHA256:", hashlib.sha256(q.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
