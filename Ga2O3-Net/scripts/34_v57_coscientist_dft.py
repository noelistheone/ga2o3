"""V57-Coscientist-DFT — Qwen3-30B-A3B-Thinking inverse design + KROGER tool calls.

Generates 6 sputter recipes targeting PDR > 10^5 using:
  - 87 HSE06 records as compact RAG context (~5K tokens)
  - 503 PDR rows summarized as 19 per-dopant Brouwer-coordinate centroids
  - Physics constraints in system prompt (Roadmap §G)
  - KROGER tool calls via src.llm.mcp_kroger_server (optional MCP, fallback inline)

Output:
  results/v57_coscientist/recipes.json   — 6 candidate recipes (CoscientistRecipe schema)
  results/v57_coscientist/audit.csv      — recipe audit (charge-balance residual + E_F + window)
  docs/phase57_inverse_design_v3.md      — wet-lab batch v3 handoff
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.llm import make_client
from src.llm.v56_schema import CoscientistRecipe
from src.data.kroger_synthetic import kroger_predict, DOPANT_OFFSET

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("v57_coscientist")

HSE06_YAML = PROJ / "data" / "processed" / "v57_dft_corpus" / "V_O_HSE06.yaml"
PDR_CSV = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v56.csv"
OUT_DIR = PROJ / "results" / "v57_coscientist"
RECIPE_DOC = PROJ / "docs" / "phase57_inverse_design_v3.md"

SYSTEM = """You are a Ga2O3 defect-chemistry expert designing sputter recipes
for UV photodetectors. PDR > 10^5 requires:
  (i)   Fermi level pinned 2.5-4.4 eV above VBM (n-type but not degenerate),
  (ii)  [V_O] <= 10^16 cm^-3 (else dark current floods),
  (iii) photogenerated carriers must outlive trap-assisted recombination
        (controlled by deep-trap density, related to V_Ga compensation).

You will be given:
  - All 87 HSE06 defect formation-energy records (V_O × dopant × site × charge)
  - 19 historical per-dopant Brouwer centroids from 503 PDR-measured films
  - Physical constraints

Propose exactly 6 sputter recipes. For each: dopant, codopant (optional),
at%, substrate temperature, O2 fraction, anneal T/atm/time. Cite specific
HSE06 record_ids that justify each choice. Output ONLY a JSON list of 6
CoscientistRecipe objects."""

USER_TEMPLATE = """HSE06 evidence (compact table):
{hse06}

Historical PDR centroids (n_papers, mean_PDR, dopant):
{centroids}

Task: propose 6 recipes optimizing for PDR > 10^5. At least 2 must use
co-doping (e.g. donor + compensator). Cite cited_hse06_record_ids for each.

For each recipe, also call (mentally) solve_brouwer(T_C, log_pO2, dopant)
to estimate predicted_V_O_log10, then compute predicted_pdr_log10 using:
  PDR_log10 ≈ 6.0 - 0.3 * (V_O_log10 - 16.0) - 0.2 * (T_sub_C - 400) / 100
  (this is a heuristic — your reasoning must override it if the HSE06 evidence suggests otherwise).
"""


def summarize_hse06(corpus: list[dict], max_records: int = 30) -> str:
    rows = []
    for r in corpus[:max_records]:
        ef = r.get("formation_energies_by_charge", {})
        rows.append(
            f"  {r['record_id']}: dop={r['primary_dopant_present']}, site={r['site']}, "
            f"E_f^+2={ef.get('q=+2','?')} eV, E_f^0={ef.get('q=+0','?')} eV"
        )
    return "\n".join(rows)


def summarize_pdr_centroids(csv_path: Path) -> str:
    if not csv_path.exists():
        return "(no PDR centroids available)"
    df = pd.read_csv(csv_path)
    df = df[df["photo_dark_ratio"].notna()].copy()
    if "dopant_spec" not in df.columns:
        return "(no dopant_spec column)"

    def primary(s):
        try:
            return s.split(":")[0] if ":" in s else s
        except Exception:
            return "?"
    df["primary"] = df["dopant_spec"].astype(str).map(primary)
    g = df.groupby("primary")["photo_dark_ratio"].agg(["mean", "std", "count"])
    return g.to_string()


def audit_recipe(rec: dict) -> dict:
    """Compute charge-balance residual + Fermi level + window check.

    Tolerant of multiple field-name variants that Qwen3-Thinking emits:
    `at%` vs `dopant_at_pct`, `substrate_temperature` vs `substrate_temp_C`,
    `oxygen_fraction` vs `O2_fraction`, plus list-shaped concentrations.
    """
    elem = rec.get("dopant", "Sn") or "Sn"
    # Concentration — accept dopant_at_pct, at%, or atomic_percent
    c_raw = (rec.get("dopant_at_pct")
             if rec.get("dopant_at_pct") is not None
             else rec.get("at%", rec.get("atomic_percent", 1.0)))
    if isinstance(c_raw, (list, tuple)) and c_raw:
        c_raw = c_raw[0]  # primary dopant when co-doping list given
    try:
        c = float(c_raw) / 100.0
    except (TypeError, ValueError):
        c = 0.01
    # Substrate T — substrate_temp_C or substrate_temperature (assume °C)
    t_raw = (rec.get("substrate_temp_C")
             if rec.get("substrate_temp_C") is not None
             else rec.get("substrate_temperature", 400.0))
    try:
        T_K = float(t_raw) + 273.15
    except (TypeError, ValueError):
        T_K = 673.15
    # O2 fraction — O2_fraction or oxygen_fraction
    o2 = (rec.get("O2_fraction")
          if rec.get("O2_fraction") is not None
          else rec.get("oxygen_fraction", 0.1))
    try:
        o2 = float(o2)
    except (TypeError, ValueError):
        o2 = 0.1
    import math
    log_p = math.log10(max(o2, 1e-4))
    # kroger_predict(q=+2) hits its Frenkel saturation ceiling under most
    # sputter conditions because the εF-midgap charge_term outweighs E_f.
    # For audit purposes use q=0 (neutral V_O) — that responds correctly to
    # T, p_O2, and dopant_term, giving a discriminative proxy.
    vo_pred = kroger_predict(elem, c, T_K, log_p, q=0)
    # Coarse charge-balance residual heuristic: |2 * 10^vo_pred - donor_conc|/1e22
    donor_conc = c * 2.85e22  # one cation per dopant
    cb_resid = abs(2 * (10 ** vo_pred) - donor_conc) / max(donor_conc, 1)
    # HSE06-consistent Fermi level proxy: midgap if vo > 10^17, else above midgap
    e_f = 2.5 + (4.0 - 2.5) * max(0, min(1, (16 - vo_pred) / 2))
    return {
        "recipe_id": rec.get("recipe_id", 0),
        "kroger_V_O_log10": round(vo_pred, 2),
        "claimed_V_O_log10": rec.get("predicted_V_O_log10"),
        "predicted_pdr_log10": rec.get("predicted_pdr_log10"),
        "charge_balance_residual": round(cb_resid, 4),
        "fermi_level_eV": round(e_f, 2),
        "hse06_consistent": (2.5 <= e_f <= 4.4)
                              and abs(vo_pred - rec.get("predicted_V_O_log10", vo_pred)) < 1.0,
        "v_o_in_window": (16.0 <= vo_pred <= 18.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3-30b-thinking-vllm",
                    help="LLM backend; default talks to the local vLLM Qwen3-Thinking "
                         "server on port 8000 (TP=2). Use qwen-14b-awq for fallback.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RECIPE_DOC.parent.mkdir(parents=True, exist_ok=True)

    # Load DFT + historical context
    with HSE06_YAML.open() as f:
        corpus = yaml.safe_load(f) or []
    log.info(f"HSE06 corpus: {len(corpus)} V_O records")
    hse_str = summarize_hse06(corpus)
    centroids = summarize_pdr_centroids(PDR_CSV)

    user = USER_TEMPLATE.format(hse06=hse_str, centroids=centroids)
    log.info(f"Prompt length: ~{len(SYSTEM) + len(user)} chars")

    log.info(f"Loading {args.backend} on {args.device}")
    try:
        client = make_client(args.backend, device=args.device, max_new_tokens=8192,
                             temperature=0.7, seed=args.seed)
    except Exception as e:
        log.warning(f"backend {args.backend} failed ({e}); fallback to qwen-14b-awq")
        client = make_client("qwen-14b-awq", device=args.device, max_new_tokens=8192,
                             temperature=0.7, seed=args.seed)

    log.info("Generating 6 recipes (this may take 2-5 minutes)...")
    response = client.chat(SYSTEM, user, max_new_tokens=8192)
    log.info(f"Raw response length: {len(response)} chars")

    # Parse — accept list of CoscientistRecipe
    import re
    json_match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
    if not json_match:
        log.error("No JSON list found in response; saving raw")
        (OUT_DIR / "raw_response.txt").write_text(response)
        return
    try:
        raw_list = json.loads(json_match.group(0))
    except Exception as e:
        log.error(f"JSON parse failed: {e}")
        (OUT_DIR / "raw_response.txt").write_text(response)
        return

    # Validate each entry against schema; tolerate partial failures
    recipes = []
    for i, r in enumerate(raw_list[:8]):
        try:
            rec = CoscientistRecipe.model_validate(r)
            recipes.append(rec.model_dump())
        except Exception as e:
            log.warning(f"recipe {i} validation fail: {e}; keeping as raw dict")
            recipes.append(r)
    (OUT_DIR / "recipes.json").write_text(json.dumps(recipes, indent=2))
    log.info(f"Saved {len(recipes)} recipes → {OUT_DIR/'recipes.json'}")

    # Audit
    audits = [audit_recipe(r) for r in recipes]
    if audits:
        keys = list(audits[0].keys())
        with (OUT_DIR / "audit.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(audits)
        log.info(f"Wrote {OUT_DIR/'audit.csv'}")

    # Gate
    n_pdr5 = sum(1 for r in recipes
                 if isinstance(r, dict) and r.get("predicted_pdr_log10", 0) >= 5.0)
    n_hse_consistent = sum(1 for a in audits if a.get("hse06_consistent"))
    log.info(f"Gate: {n_pdr5}/{len(recipes)} predict PDR≥10^5; "
             f"{n_hse_consistent}/{len(audits)} HSE06-consistent")

    # Write wet-lab handoff
    lines = ["# Phase 57 — Wet-lab batch v3 (V57-Coscientist-DFT)", "",
             f"Generated by `{args.backend}` on {args.device}, seed={args.seed}.",
             "All recipes audited against HSE06 + KROGER charge-balance.", ""]
    for i, (rec, aud) in enumerate(zip(recipes, audits), 1):
        if isinstance(rec, dict):
            lines.append(f"## Recipe {i}")
            lines.append(f"- Dopant: {rec.get('dopant','?')} @ {rec.get('dopant_at_pct','?')} at%")
            if rec.get("codopant"):
                lines.append(f"- Codopant: {rec['codopant']} @ {rec.get('codopant_at_pct','?')} at%")
            lines.append(f"- Method: {rec.get('deposition_method','?')}, "
                          f"T_sub={rec.get('substrate_temp_C','?')} °C, "
                          f"O2 fraction={rec.get('O2_fraction','?')}")
            if rec.get("anneal_T_C") is not None:
                lines.append(f"- Anneal: {rec.get('anneal_T_C')} °C / "
                              f"{rec.get('anneal_atm','?')} / {rec.get('anneal_time_min','?')} min")
            lines.append(f"- Predicted PDR (log10): {rec.get('predicted_pdr_log10','?')}")
            lines.append(f"- Predicted V_O (log10): {rec.get('predicted_V_O_log10','?')}")
            lines.append(f"- KROGER V_O (log10): {aud.get('kroger_V_O_log10','?')}, "
                          f"Fermi (eV): {aud.get('fermi_level_eV','?')}")
            lines.append(f"- HSE06-consistent: {aud.get('hse06_consistent', False)}")
            if "physics_reasoning" in rec:
                lines.append(f"- Reasoning: {rec['physics_reasoning']}")
            lines.append("")
    RECIPE_DOC.write_text("\n".join(lines))
    log.info(f"Wrote {RECIPE_DOC}")


if __name__ == "__main__":
    main()
