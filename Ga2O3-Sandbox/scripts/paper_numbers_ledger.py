"""Numbers ledger for the npj paper: re-read EVERY metric to be cited from its file of record.
Output paper/npj/NUMBERS.json with {claim: {value, file, field}} — the writing cites only from here."""
import json
from pathlib import Path
P = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
N = Path("/home/lawrence/Physics/Ga2O3-Net")
L = {}
def add(k, f, path, transform=None):
    d = json.load(open(f))
    v = d
    for p in path.split("."):
        v = v[int(p)] if p.lstrip("-").isdigit() and isinstance(v, list) else v[p]
    L[k] = {"value": transform(v) if transform else v, "file": str(f), "field": path}

# Act 2 — sandbox validation
add("hse_vs_kroger_VO_Ef", P/"results/tier2/hse_formation_energy.json", "E_f_HSE_VO_q0_Orich_eV")
add("pbe_VO_Ef", P/"results/tier2/hse_formation_energy.json", "comparison.PBE_VO_q0_Orich")
add("within_paper_n_sign_frac", P/"results/tier2/corpus_trend_battery.json", "T2_within_paper_n_sign.frac")
add("within_paper_n_sign_n", P/"results/tier2/corpus_trend_battery.json", "T2_within_paper_n_sign.n_contrasts")
add("within_paper_n_sign_correct", P/"results/tier2/corpus_trend_battery.json", "T2_within_paper_n_sign.correct_sign")
add("PDR_sign_frac", P/"results/tier2/corpus_trend_battery.json", "T3_within_paper_PDR_sign.frac")
add("cross_dopant_rho", P/"results/tier2/corpus_trend_battery.json", "T1_cross_dopant_n_ranking.spearman_rho")
add("cross_dopant_p", P/"results/tier2/corpus_trend_battery.json", "T1_cross_dopant_n_ranking.perm_p")
add("si_series_spearman", P/"results/tier2/relative_validation.json", "series.1.spearman")
add("si_series_npoints", P/"results/tier2/relative_validation.json", "series.1.n_points")
lab = json.load(open(P/"results/tier2/lab_validation.json"))
L["lab_PDR_direction_correct"] = {"value": sum(1 for v in lab["results"].values() if v["PDR_direction_correct"]),
                                   "of": len(lab["results"]), "file": str(P/"results/tier2/lab_validation.json")}
# MLIP benchmark
add("mace_geom_rmsd", P/"results/tier3/geom_fidelity.json", "rmsd_MACE_vs_QE_final_A")
add("sb_level_hse_belowCBM", P/"results/tier3/new_dopant_hse.json", "dopants.Sb.e_2plus_0_below_CBM_eV")
add("bi_level_hse_belowCBM", P/"results/tier3/new_dopant_hse.json", "dopants.Bi.e_2plus_0_below_CBM_eV")
add("vo_2plus0_inhouse", P/"results/tier3/hse_vo_2plus0_level.json", "eps_2plus_0_above_VBM_eV")
add("disorder_dEg_ensemble", P/"results/tier3/disorder_deg.json", "dEg_disorder_eV")
add("disorder_gap_crystal", P/"results/tier3/disorder_deg.json", "crystal_gap_PBE_eV")
add("disorder_dEg_mattersim", P/"results/tier3/disorder_deg_mattersim.json", "dEg_disorder_eV")
# Act 3 — hybrid certification
add("mu_LOLO_dex", P/"results/hybrid/certification.json", "properties.mu.modeA_crosslab_LOLO_median_dex")
add("mu_LOLO_factor", P/"results/hybrid/certification.json", "properties.mu.modeA_factor")
add("mu_nlabs", P/"results/hybrid/certification.json", "properties.mu.n_labs")
add("n_LOLO_dex", P/"results/hybrid/certification.json", "properties.n.modeA_crosslab_LOLO_median_dex")
add("n_LOLO_factor", P/"results/hybrid/certification.json", "properties.n.modeA_factor")
add("n_modeB_dex", P/"results/hybrid/certification.json", "properties.n.modeB_anchored_median_dex")
add("n_withinlab_floor", P/"results/hybrid/certification.json", "properties.n.within_lab_scatter_floor_dex")
add("delta_mu_skill_share", P/"results/hybrid/delta_mu.json", "delta_skill_share")
# Act 4 — designer
add("design_cond_number", P/"results/hybrid/design_sensitivity.json", "condition_number")
add("design_global_dex", P/"results/hybrid/design_validation.json", "pooled_median_dex.global")
add("design_floor_dex", P/"results/hybrid/design_validation.json", "pooled_median_dex.floor")
add("design_des2", P/"results/hybrid/design_validation.json", "pooled_median_dex.des2")
add("design_rand2", P/"results/hybrid/design_validation.json", "pooled_median_dex.rand2")
add("design_des3", P/"results/hybrid/design_validation.json", "pooled_median_dex.des3")
add("design_rand3", P/"results/hybrid/design_validation.json", "pooled_median_dex.rand3")
add("design_nlabs", P/"results/hybrid/design_validation.json", "n_labs")
(P/"paper/npj/NUMBERS.json").write_text(json.dumps(L, indent=1))
print(f"ledger: {len(L)} claims"); [print(f"  {k} = {v['value']}") for k,v in list(L.items())[:12]]
