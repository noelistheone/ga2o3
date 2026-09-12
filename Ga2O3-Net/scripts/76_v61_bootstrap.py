"""V61 verification — DOI-level bootstrap CI for the within-DOI relative R^2 / Pearson.

Reads the OOF predictions (results/phase61_v61/oof_<target><tag>.npz) and resamples DOIs (the
leakage-free unit) with replacement to get a confidence interval on the within-DOI relative R^2 and
Pearson. Answers the adversarial question: is +0.035 within-DOI R^2 distinguishable from 0, or noise?
CPU-only; writes results/phase61_v61/bootstrap_<target><tag>.json.
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]


def demean_by_group(vals, groups):
    out = np.array(vals, float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = out[m] - out[m].mean()
    return out


def within_r2_pearson(y, r, doi, min_n=2):
    keep = np.zeros(len(y), bool)
    for g in np.unique(doi):
        m = doi == g
        if m.sum() >= min_n and np.std(y[m]) > 1e-9 and np.std(r[m]) > 1e-12:
            keep |= m
    if keep.sum() < 3:
        return float("nan"), float("nan"), int(keep.sum())
    yd = demean_by_group(y[keep], doi[keep]); rd = demean_by_group(r[keep], doi[keep])
    beta = np.sum(rd * yd) / (np.sum(rd * rd) + 1e-12)
    r2 = 1.0 - np.sum((yd - beta * rd) ** 2) / (np.sum(yd ** 2) + 1e-12)
    pear = float(np.corrcoef(rd, yd)[0, 1]) if np.std(rd) > 0 else float("nan")
    return float(r2), pear, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="vacancy_concentration")
    ap.add_argument("--tag", default="")          # "" full, or "_no_solver" etc.
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    npz = PROJ / f"results/phase61_v61/oof_{args.target}{args.tag}.npz"
    d = np.load(npz, allow_pickle=True)
    r = np.nanmean(d["r"], axis=0); y = d["y"].astype(float); doi = d["doi"].astype(str)
    pt_r2, pt_pear, n_used = within_r2_pearson(y, r, doi)

    rng = np.random.default_rng(args.seed)
    uniq = np.unique(doi)
    boots_r2, boots_pear = [], []
    for _ in range(args.nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(doi == g)[0] for g in samp])
        # relabel DOIs so duplicated groups stay separate units
        relab = np.concatenate([[f"{g}#{k}"] * (doi == g).sum() for k, g in enumerate(samp)])
        b_r2, b_pear, _ = within_r2_pearson(y[idx], r[idx], relab)
        if np.isfinite(b_r2):
            boots_r2.append(b_r2); boots_pear.append(b_pear)
    boots_r2 = np.array(boots_r2); boots_pear = np.array(boots_pear)
    rep = dict(
        target=args.target, tag=args.tag, n_used=n_used, point_within_R2=pt_r2, point_within_pearson=pt_pear,
        R2_CI95=[float(np.percentile(boots_r2, 2.5)), float(np.percentile(boots_r2, 97.5))],
        R2_median=float(np.median(boots_r2)), R2_frac_gt0=float((boots_r2 > 0).mean()),
        pearson_CI95=[float(np.percentile(boots_pear, 2.5)), float(np.percentile(boots_pear, 97.5))],
        pearson_frac_gt0=float((boots_pear > 0).mean()), nboot=len(boots_r2))
    out = PROJ / f"results/phase61_v61/bootstrap_{args.target}{args.tag}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
