"""V57-Coscientist-DFT — MCP KROGER tool server (stub).

Exposes three KROGER/Brouwer tools to Qwen3-30B-A3B-Thinking via the
ModelContextProtocol (MCP) stdio transport:

  solve_brouwer(T_C, log_pO2, dopant_spec) → {V_O_log10, E_F_eV, charge_balance_residual}
  hse06_window_check(dopant, V_O_log10)   → {in_window, lower, upper}
  predict_pdr(recipe)                      → {pdr_log10, pdr_std}

V57-Coscientist-DFT calls these via Qwen3's native --enable-auto-tool-choice
when running under vLLM. When vLLM TP=2 is unavailable, the scripts/34_v57_coscientist_dft.py
falls back to in-prompt heuristics + this module's pure-Python helpers
(no MCP server actually running).

Launch (when vLLM stack works):
  python -m src.llm.mcp_kroger_server
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ))

import math
from src.data.kroger_synthetic import kroger_predict, DOPANT_OFFSET


# Pure-Python tool implementations (callable without MCP) ---------------------


def solve_brouwer(T_C: float, log_pO2: float, dopant_spec: str = "Sn:0.01") -> dict:
    """Closed-form Brouwer solution for log10[V_O^+2] given (T, log_pO2, dopant).

    Returns {V_O_log10, E_F_eV (assumed mid-gap), charge_balance_residual}.
    """
    elem = dopant_spec.split(":")[0].split(",")[0]
    try:
        c = float(dopant_spec.split(":")[1].split(",")[0])
    except Exception:
        c = 1e-2
    T_K = T_C + 273.15
    v_o = kroger_predict(elem, c, T_K, log_pO2, q=2)
    donor_conc = c * 2.85e22
    cb_resid = abs(2 * 10 ** v_o - donor_conc) / max(donor_conc, 1)
    e_f = 2.5 + 1.5 * max(0, min(1, (16 - v_o) / 2))
    return {
        "V_O_log10": round(v_o, 2),
        "E_F_eV":    round(e_f, 2),
        "charge_balance_residual": round(cb_resid, 4),
    }


def hse06_window_check(dopant: str, V_O_log10: float) -> dict:
    """Per-dopant HSE06 plausibility window — fixed [16, 18] for now, can be
    tightened once full HSE06 corpus is loaded."""
    lower, upper = 16.0, 18.0
    return {
        "in_window": (lower <= V_O_log10 <= upper),
        "lower": lower, "upper": upper,
        "dopant": dopant,
    }


def predict_pdr(recipe: dict) -> dict:
    """Coarse PDR predictor from V54-A1 + V55-Ext ensemble — stub.

    Real implementation should load the locked frontier model bundles. For
    Phase 57 inverse-design ranking it's enough to score recipes by the
    Brouwer-derived heuristic in scripts/34_v57_coscientist_dft.py:audit_recipe.
    """
    elem = recipe.get("dopant", "Sn")
    c = recipe.get("dopant_at_pct", 1.0) / 100.0
    T_C = recipe.get("substrate_temp_C", 400.0)
    o2 = recipe.get("O2_fraction", 0.1)
    sb = solve_brouwer(T_C, math.log10(max(o2, 1e-4)), f"{elem}:{c}")
    pdr = 6.0 - 0.3 * (sb["V_O_log10"] - 16.0) - 0.2 * (T_C - 400) / 100
    return {"pdr_log10": round(pdr, 2), "pdr_std": 0.5}


# MCP server (only used when V57-Coscientist runs under vLLM auto-tool-choice)


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as e:
        print(f"mcp package missing or broken: {e}", file=sys.stderr)
        sys.exit(1)
    mcp = FastMCP("v57-kroger")

    @mcp.tool()
    def solve_brouwer_tool(T_C: float, log_pO2: float, dopant_spec: str) -> dict:
        return solve_brouwer(T_C, log_pO2, dopant_spec)

    @mcp.tool()
    def hse06_window_check_tool(dopant: str, V_O_log10: float) -> dict:
        return hse06_window_check(dopant, V_O_log10)

    @mcp.tool()
    def predict_pdr_tool(recipe: dict) -> dict:
        return predict_pdr(recipe)

    mcp.run()


if __name__ == "__main__":
    main()
