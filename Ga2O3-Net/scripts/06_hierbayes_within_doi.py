"""Phase 54 V54-D1 — Hierarchical Bayesian within-DOI re-evaluation.

Replaces the bootstrap CI on within-DOI concentration sensitivity with a
NumPyro NUTS-fitted random-intercept-and-slope model. Partial pooling shrinks
per-DOI estimates toward the population mean and tightens CIs without
requiring more labels — addresses the V53-ζ regression #3 (within-DOI CI
ambiguity) by statistical efficiency, not new data.

Hierarchical model (non-centered, D=4 funnel-safe):
    μ_α, μ_β  ~ Normal(0, 1)
    σ_α, σ_β  ~ HalfNormal(0.5)          # tighter than HalfCauchy to avoid funnel
    σ          ~ HalfCauchy(1)
    z_α[d], z_β[d] ~ Normal(0, 1)         for d ∈ DOIs
    α_d = μ_α + σ_α · z_α[d]
    β_d = μ_β + σ_β · z_β[d]
    y[i] ~ Normal(α_{doi[i]} + β_{doi[i]} · logc[i], σ)

where y = (pred_platt - true) is the V_O residual and logc = log10(concentration_at%).

Outputs:
    {bundle}_hierbayes_summary.csv     per-DOI (β_mean, hpd_lo, hpd_hi, stat_sig)
    {bundle}_diagnostics.txt           r-hat / ESS / divergent transitions
    {bundle}_traces.png                4-panel chain trace
    {bundle}_comparison.csv            bootstrap vs hier-Bayes CI width-reduction

CLI:
    python scripts/06_hierbayes_within_doi.py \\
        --bundle-dir results/phase53v53z_contrastive_vc_5seed \\
        --target vacancy_concentration \\
        --output-dir results/phase54v54d1_hierbayes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def _apply_dataset_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate eval_phase42_vs_baselines.py:46-49 filter for sample_idx alignment."""
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    return df


def _load_and_merge(bundle_dir: Path, src_csv: Path, target: str) -> pd.DataFrame:
    oof = pd.read_csv(bundle_dir / "oof_predictions.csv")
    src = _apply_dataset_filter(pd.read_csv(src_csv))
    src["sample_idx"] = np.arange(len(src), dtype=int)
    if len(oof) != len(src):
        raise RuntimeError(
            f"OOF rows ({len(oof)}) != post-filter src rows ({len(src)}); "
            "filter recipe drifted vs dataset"
        )
    merged = oof.merge(
        src[["sample_idx", "doi", "concentration_at%", "method", "element"]],
        on="sample_idx", how="left",
    )
    work = merged.dropna(subset=[
        f"{target}_true", f"{target}_pred_platt", "doi", "concentration_at%",
    ]).copy()
    # Sputter-only for the within-DOI metric (matches eval_sputter_physics.py).
    work = work[work["method"].fillna("").str.lower().str.contains("sputter")]
    work["c"] = pd.to_numeric(work["concentration_at%"], errors="coerce")
    work = work[work["c"] > 0].copy()
    work["logc"] = np.log10(work["c"])
    work["resid"] = work[f"{target}_pred_platt"] - work[f"{target}_true"]
    return work.reset_index(drop=True)


def _hier_model_nc(logc, doi_idx, y, D):
    """Non-centered hierarchical regression."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    mu_beta  = numpyro.sample("mu_beta",  dist.Normal(0.0, 1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(0.5))
    sigma_beta  = numpyro.sample("sigma_beta",  dist.HalfNormal(0.5))
    sigma       = numpyro.sample("sigma",       dist.HalfCauchy(1.0))
    with numpyro.plate("doi", D):
        z_alpha = numpyro.sample("z_alpha", dist.Normal(0.0, 1.0))
        z_beta  = numpyro.sample("z_beta",  dist.Normal(0.0, 1.0))
    alpha_d = numpyro.deterministic("alpha_d", mu_alpha + sigma_alpha * z_alpha)
    beta_d  = numpyro.deterministic("beta_d",  mu_beta  + sigma_beta  * z_beta)
    y_pred = alpha_d[doi_idx] + beta_d[doi_idx] * logc
    numpyro.sample("obs", dist.Normal(y_pred, sigma), obs=y)


def _fit(work: pd.DataFrame, n_chains: int, warmup: int, draws: int,
         target_accept: float, seed: int):
    """Fit hierarchical model on the prepared dataframe. Returns mcmc handle + idx map."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    doi_codes, doi_levels = pd.factorize(work["doi"])
    D = len(doi_levels)

    kernel = NUTS(_hier_model_nc, target_accept_prob=target_accept, max_tree_depth=12)
    mcmc = MCMC(kernel,
                 num_warmup=warmup, num_samples=draws,
                 num_chains=n_chains, progress_bar=True,
                 chain_method="parallel" if n_chains > 1 else "sequential")
    mcmc.run(
        jax.random.PRNGKey(seed),
        logc=jnp.asarray(work["logc"].values, dtype=jnp.float32),
        doi_idx=jnp.asarray(doi_codes, dtype=jnp.int32),
        y=jnp.asarray(work["resid"].values, dtype=jnp.float32),
        D=D,
    )
    return mcmc, doi_levels


def _summarize(mcmc, doi_levels: pd.Index, work: pd.DataFrame, out: Path) -> pd.DataFrame:
    import arviz as az
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata, var_names=["beta_d", "alpha_d", "mu_beta", "sigma_beta", "sigma"],
                         hdi_prob=0.95)
    summary.to_csv(out)

    # Build per-DOI table.
    beta_post = idata.posterior["beta_d"].values   # [chain, draw, D]
    beta_flat = beta_post.reshape(-1, beta_post.shape[-1])
    rows = []
    for d, doi in enumerate(doi_levels):
        n_d = int((work["doi"] == doi).sum())
        b = beta_flat[:, d]
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append({
            "doi": doi,
            "n": n_d,
            "beta_mean": float(b.mean()),
            "beta_hpd_lo": float(lo),
            "beta_hpd_hi": float(hi),
            "stat_sig_at_95": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


def _bootstrap_within_doi(work: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap CI on per-DOI Spearman of (predicted residual vs logc).
    Mirrors eval_sputter_physics.bootstrap_within_doi_ci semantics: simple
    percentile CI over n_boot resamples of within-DOI rows."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    rows = []
    for doi, g in work.groupby("doi"):
        if len(g) < 3:
            continue
        if g["c"].nunique() < 2:
            continue
        if (g["logc"].max() - g["logc"].min()) < 0.05:
            continue
        rho_obs, _ = spearmanr(g["logc"].values, g["resid"].values)
        if np.isnan(rho_obs):
            continue
        # Bootstrap
        rhos = []
        for _ in range(n_boot):
            sample = g.sample(n=len(g), replace=True, random_state=rng.integers(0, 1 << 31))
            if sample["c"].nunique() < 2:
                continue
            r, _ = spearmanr(sample["logc"].values, sample["resid"].values)
            if not np.isnan(r):
                rhos.append(r)
        if not rhos:
            continue
        lo, hi = np.percentile(rhos, [2.5, 97.5])
        rows.append({
            "doi": doi,
            "n": int(len(g)),
            "boot_rho": float(rho_obs),
            "boot_lo": float(lo),
            "boot_hi": float(hi),
            "boot_width": float(hi - lo),
            "boot_stat_sig": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


def _comparison(hb: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side bootstrap vs hier-Bayes CI table."""
    merged = bs.merge(hb, on="doi", how="outer", suffixes=("_bs", "_hb"))
    merged["hb_width"] = (merged["beta_hpd_hi"] - merged["beta_hpd_lo"]).astype(float)
    # CI-width units differ (Spearman ρ vs slope β), so report fractional width
    # reduction relative to each method's own scale: % reduction in CI width
    # vs a per-method reference is not directly comparable. Instead report:
    #  - bootstrap ρ stat_sig (rejecting H0: ρ=0)
    #  - hier-Bayes β stat_sig (rejecting H0: β=0)
    # and per-method widths.
    return merged[[
        "doi", "n_bs", "boot_rho", "boot_lo", "boot_hi", "boot_width", "boot_stat_sig",
        "beta_mean", "beta_hpd_lo", "beta_hpd_hi", "hb_width", "stat_sig_at_95",
    ]].rename(columns={"n_bs": "n", "stat_sig_at_95": "hb_stat_sig"})


def _power_analysis(mcmc, work: pd.DataFrame, n_new_list=(2, 4, 8),
                    n_per_doi: int = 4, n_sim: int = 300, seed: int = 0) -> pd.DataFrame:
    """For each posterior sample draw n_new synthetic DOIs at posterior effect
    sizes, simulate y, fit per-DOI fixed-effects regression, and report the
    fraction of simulations where ALL existing+new DOIs are stat-sig at α=0.05.
    """
    import jax
    rng = np.random.default_rng(seed)

    post = mcmc.get_samples()
    mu_beta_p   = np.array(post["mu_beta"])
    sigma_beta_p = np.array(post["sigma_beta"])
    sigma_p     = np.array(post["sigma"])
    K = len(mu_beta_p)

    logc_grid = np.linspace(work["logc"].min(), work["logc"].max(), n_per_doi)
    # Use the actual within-DOI stat-sig count as baseline.
    existing_doi_count = int(work["doi"].nunique())

    out = []
    for n_new in n_new_list:
        pass_count = 0
        for _ in range(n_sim):
            k = rng.integers(0, K)
            mu_beta = float(mu_beta_p[k]); sigma_beta = float(sigma_beta_p[k]); sigma = float(sigma_p[k])
            # Generate n_new synthetic DOIs
            beta_new = rng.normal(mu_beta, sigma_beta, size=n_new)
            sig_per_doi = []
            for b in beta_new:
                y_sim = b * logc_grid + rng.normal(0.0, sigma, size=n_per_doi)
                # OLS slope test (2-sided, t-stat)
                x = logc_grid - logc_grid.mean()
                yc = y_sim - y_sim.mean()
                if (x ** 2).sum() < 1e-9:
                    sig_per_doi.append(False); continue
                slope = (x * yc).sum() / (x ** 2).sum()
                resid = yc - slope * x
                dof = n_per_doi - 2
                if dof <= 0:
                    sig_per_doi.append(False); continue
                s2 = (resid ** 2).sum() / dof
                se = float(np.sqrt(s2 / (x ** 2).sum())) if s2 > 0 else 0.0
                if se <= 0:
                    sig_per_doi.append(False); continue
                t = slope / se
                # 2-sided p<0.05 with small df: |t| > 4.30 (df=2), 3.18 (df=3), 2.78 (df=4)
                crit = {2: 4.30, 3: 3.18, 4: 2.78}.get(dof, 2.0)
                sig_per_doi.append(abs(t) > crit)
            # Existing 4 DOIs are also in the simulation pool (assume current
            # stat-sig pass rate is 1/4 — power = fraction where new+existing
            # all stat-sig: this is a conservative simulation).
            new_pass = sum(sig_per_doi)
            need_all_pass = (new_pass + 1 == n_new + existing_doi_count)
            if need_all_pass:
                pass_count += 1
        out.append({
            "n_new_dois": n_new,
            "n_per_doi": n_per_doi,
            "n_sim": n_sim,
            "pass_rate_all_stat_sig": pass_count / n_sim,
        })
    return pd.DataFrame(out)


def _traces_plot(mcmc, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import arviz as az
    idata = az.from_numpyro(mcmc)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    az.plot_trace(idata, var_names=["mu_beta", "sigma_beta"], axes=axes[:, :])
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


def _diagnostics(mcmc, out_txt: Path) -> str:
    import arviz as az
    import io
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata, hdi_prob=0.95)
    n_eff = summary["ess_bulk"].min()
    rhat = summary["r_hat"].max()
    buf = io.StringIO()
    buf.write(f"Phase 54 V54-D1 — NUTS hierarchical model diagnostics\n")
    buf.write(f"  max r-hat       : {rhat:.4f}  (gate: < 1.05)\n")
    buf.write(f"  min ESS bulk    : {n_eff:.0f}  (gate: > 200)\n")
    buf.write(f"  divergent count : {int(mcmc.get_extra_fields()['diverging'].sum())}\n\n")
    buf.write(summary.to_string())
    out_txt.write_text(buf.getvalue())
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="V54-D1 hierarchical Bayesian within-DOI")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--target", default="vacancy_concentration")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--src-csv", type=Path,
                        default=PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp.csv")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--power-analysis", action="store_true",
                        help="Run posterior-predictive power simulation")
    args = parser.parse_args()

    import numpyro
    numpyro.set_host_device_count(max(args.chains, 1))

    bundle_name = args.bundle_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    work = _load_and_merge(args.bundle_dir, args.src_csv, args.target)
    n_dois = work["doi"].nunique()
    print(f"Bundle              : {bundle_name}")
    print(f"After filter, rows  : {len(work)}  ({n_dois} DOIs)")
    if len(work) < 6 or n_dois < 2:
        raise RuntimeError(
            "Insufficient rows / DOIs for hierarchical model. "
            "Check that the bundle's OOF has VC labels."
        )

    print("Fitting NUTS …")
    mcmc, doi_levels = _fit(
        work, args.chains, args.warmup, args.draws, args.target_accept, args.seed,
    )

    diag_txt = _diagnostics(mcmc, args.output_dir / f"{bundle_name}_diagnostics.txt")
    print(diag_txt.split("\n\n")[0])

    summary_df = _summarize(mcmc, doi_levels, work,
                             args.output_dir / f"{bundle_name}_hierbayes_summary.csv")
    print("\nPer-DOI β posterior:")
    per_doi = pd.DataFrame({
        "doi": doi_levels,
        "n": [int((work["doi"] == d).sum()) for d in doi_levels],
    })
    beta_post = mcmc.get_samples()["beta_d"]   # JAX array [n_samples, D]
    bp = np.array(beta_post)
    per_doi["beta_mean"] = bp.mean(axis=0)
    per_doi["beta_hpd_lo"] = np.percentile(bp, 2.5, axis=0)
    per_doi["beta_hpd_hi"] = np.percentile(bp, 97.5, axis=0)
    per_doi["stat_sig_at_95"] = (per_doi["beta_hpd_lo"] > 0) | (per_doi["beta_hpd_hi"] < 0)
    print(per_doi.to_string(index=False))
    per_doi.to_csv(args.output_dir / f"{bundle_name}_per_doi.csv", index=False)

    # Bootstrap vs hier-Bayes side-by-side
    bs_df = _bootstrap_within_doi(work)
    cmp_df = _comparison(per_doi, bs_df)
    cmp_df.to_csv(args.output_dir / f"{bundle_name}_comparison.csv", index=False)
    print(f"\nBootstrap stat-sig: {int(bs_df['boot_stat_sig'].sum())}/{len(bs_df)}")
    print(f"Hier-Bayes stat-sig: {int(per_doi['stat_sig_at_95'].sum())}/{len(per_doi)}")

    try:
        _traces_plot(mcmc, args.output_dir / f"{bundle_name}_traces.png")
    except Exception as e:
        print(f"Traces plot skipped: {e}")

    if args.power_analysis:
        print("\nRunning power analysis …")
        pwr_df = _power_analysis(mcmc, work)
        pwr_df.to_csv(args.output_dir / f"{bundle_name}_power_analysis.csv", index=False)
        print(pwr_df.to_string(index=False))


if __name__ == "__main__":
    main()
