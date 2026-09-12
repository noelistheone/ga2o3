"""R8 GRP-04 (blocking): resolve laboratory identity above the DOI level.
(1) Harvest author lists for every transport-corpus DOI (Crossref REST; DataCite for 10.48550
    arXiv DOIs; unresolved DOIs stay singleton groups).
(2) Cluster DOIs into research-group components: two DOIs link if they share the last author
    or share >= 2 normalized author names (family + first initial).
(3) Re-run at group level: (a) REML ICC for carrier density and mobility (custom profiled REML,
    identical to r8_icc_lrt.py), (b) mobility LOLO certification with leave-one-GROUP-out.
Writes results/tier2/r8_author_groups.json (+ author cache authors_cache.json).
"""
import sys, json, time, re, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
CACHE = PROJ / "results/tier2/authors_cache.json"

raw = pd.read_csv(NET / "results/phase65/transport_llm_extracted_v2.csv")
dois = sorted(set(str(d).strip() for d in raw.doi.dropna() if str(d).startswith(("10.",))))
print(f"{len(dois)} unique transport DOIs")

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def fetch(doi):
    if doi in cache:
        return cache[doi]
    url = None
    if doi.lower().startswith("10.48550"):
        url = f"https://api.datacite.org/dois/{urllib.parse.quote(doi, safe='')}"
    else:
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "ga2o3-sandbox-audit (mailto:haofengli228@gmail.com)"}),
                timeout=30) as r:
            j = json.loads(r.read())
        if "message" in j:  # crossref
            auth = [(a.get("family", ""), a.get("given", "")) for a in j["message"].get("author", [])]
        else:  # datacite
            auth = [(a.get("familyName", "") or a.get("name", "").split(",")[0],
                     a.get("givenName", "")) for a in
                    j.get("data", {}).get("attributes", {}).get("creators", [])]
    except Exception as e:
        auth = []
    cache[doi] = auth
    return auth


def norm(fam, giv):
    s = unicodedata.normalize("NFKD", f"{fam}").encode("ascii", "ignore").decode().lower().strip()
    gi = unicodedata.normalize("NFKD", f"{giv}").encode("ascii", "ignore").decode().lower().strip()
    return f"{s}|{gi[:1]}" if s else None


for i, d in enumerate(dois):
    if d not in cache:
        fetch(d)
        if i % 10 == 0:
            CACHE.write_text(json.dumps(cache))
        time.sleep(0.3)
CACHE.write_text(json.dumps(cache))
n_resolved = sum(1 for d in dois if cache.get(d))
print(f"author lists resolved for {n_resolved}/{len(dois)}")

# normalized author sets + last authors
ASET, LAST = {}, {}
for d in dois:
    al = [norm(f, g) for f, g in cache.get(d, [])]
    al = [a for a in al if a]
    ASET[d] = set(al)
    LAST[d] = al[-1] if al else None

# union-find
parent = {d: d for d in dois}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for i, a in enumerate(dois):
    for b in dois[i + 1:]:
        if not ASET[a] or not ASET[b]:
            continue
        shared = ASET[a] & ASET[b]
        if (LAST[a] and LAST[a] == LAST[b]) or (LAST[a] in ASET[b]) or (LAST[b] in ASET[a]) \
                or len(shared) >= 2:
            union(a, b)

group = {d: find(d) for d in dois}
n_groups = len(set(group.values()))
sizes = pd.Series(list(group.values())).value_counts()
print(f"{len(dois)} DOIs -> {n_groups} author-linked groups; multi-DOI groups: {(sizes>1).sum()} "
      f"(largest {sizes.max()})")

# ---- (a) group-level REML ICC for n and mu ----
sys.path.insert(0, str(NET / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("c72", NET / "scripts/_phase72_common.py")
c72 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c72)
spec2 = importlib.util.spec_from_file_location("r8i", PROJ / "scripts/r8_icc_lrt.py")
# import only helpers: reuse by exec of the helper section
helpers = {"__file__": str(PROJ / "scripts/r8_icc_lrt.py")}
src = (PROJ / "scripts/r8_icc_lrt.py").read_text()
exec(src.split("NSIM = 2000")[0], helpers)
suffstats, reml_fit = helpers["suffstats"], helpers["reml_fit"]

out = {"n_dois": len(dois), "n_resolved": n_resolved, "n_groups": n_groups,
       "n_multi_doi_groups": int((sizes > 1).sum()), "largest_group": int(sizes.max()),
       "linkage_rule": "shared last author OR last-author-in-other's-list OR >=2 shared "
                       "normalized authors (family+first initial)"}

for fom in ("hall_carrier_n", "hall_mobility_mu"):
    d = c72.load_fom(fom)
    g_doi = np.asarray([str(x).strip() for x in d["groups"]])
    y = np.asarray(d["y"], float)
    ok = np.isfinite(y)
    y, g_doi = y[ok], g_doi[ok]
    g_grp = np.array([group.get(x, x) for x in g_doi])
    icc_doi = reml_fit(suffstats(y, g_doi))["icc"]
    icc_grp = reml_fit(suffstats(y, g_grp))["icc"]
    out[fom] = {"icc_doi_level": round(icc_doi, 4), "icc_group_level": round(icc_grp, 4),
                "n_doi": int(len(set(g_doi))), "n_groups": int(len(set(g_grp)))}
    print(fom, out[fom], flush=True)

# ---- (b) mobility LOLO at group level ----
spec3 = importlib.util.spec_from_file_location("hc", PROJ / "scripts/hybrid_certify.py")
hc = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(hc)
# monkey-patch: coarsen the doi column to author groups before certification
orig_load = hc.load_rows if hasattr(hc, "load_rows") else None
df_hook = {}


def certify_group(prop):
    import types
    # reuse hc internals by re-running certify with a wrapped dataframe: patch pandas read_csv
    orig_read = pd.read_csv

    def patched(path, *a, **k):
        df = orig_read(path, *a, **k)
        if "transport_llm_extracted_v2" in str(path) and "doi" in df.columns:
            df = df.copy()
            df["doi"] = [group.get(str(x).strip(), str(x).strip()) for x in df["doi"]]
        return df

    pd.read_csv = patched
    try:
        res = hc.certify(prop)
    finally:
        pd.read_csv = orig_read
    return res


res_mu = certify_group("mu")
out["mu_lolo_group_level"] = {k: res_mu[k] for k in res_mu
                              if isinstance(res_mu[k], (int, float, str))}
print("mu LOLO at author-group level:", json.dumps(out["mu_lolo_group_level"])[:400], flush=True)

(PROJ / "results/tier2/r8_author_groups.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_author_groups.json")
