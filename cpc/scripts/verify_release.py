"""Verify the CPC release's versioned file index before regeneration."""
from pathlib import Path
import hashlib
import json


def main():
    base = Path(__file__).resolve().parents[1]
    manifest = json.loads((base / "cpc_distribution_manifest_20261031.json").read_text())
    failures = []
    for relative, expected in manifest["files"].items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            failures.append({"path": relative, "reason": "missing or outside release"})
            continue
        blob = path.read_bytes()
        if len(blob) != expected["bytes"] or hashlib.sha256(blob).hexdigest() != expected["sha256"]:
            failures.append({"path": relative, "reason": "checksum or size differs"})
    report = {
        "article_snapshot": manifest["article_snapshot"],
        "repository_release": manifest["repository_release"],
        "files": len(manifest["files"]),
        "passed": len(manifest["files"]) - len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
