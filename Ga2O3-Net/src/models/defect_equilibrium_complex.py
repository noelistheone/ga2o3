"""Phase 63 Opt-A (V63) — SELF-CONSISTENT defect-complex (DAP) defect-equilibrium solver for beta-Ga2O3.

Upgrade over the Phase-62 multi-dopant solver: donor-acceptor PAIRING and the Fermi level are solved
JOINTLY instead of in two decoupled steps. Physics (docs/phase63_design.md sec Opt-A / sec 2):

  * The Coulomb attraction that drives pairing acts between the IONIZED species A^+...B^-, so the pairing
    strength depends on E_F through the joint-ionization factor g_ion = f_donor(E_F)*f_acceptor(E_F):
        E_bind_eff(E_F) = E_bind_neutral + g_ion(E_F) * E_Coulomb        (E_Coulomb < 0 for donor-acceptor)
        K(E_F)          = Z * exp( -E_bind_eff(E_F) / kT )
  * Bound NEUTRAL pairs are removed from the charge-setting free pool:
        c_i^free = c_i - Sum_pairs x_P^(i)
        F(E_F) = p - n + Sum_q q[V_O^q] + Sum_i z_i c_i^free f_i(E_F) = 0

WELL-POSEDNESS. At FIXED pairing {x_P}, F(E_F) is the V62 residual -> strictly decreasing -> unique inner
root (Phase-62 theorem). The E_F-dependence of {x_P} is handled by an OUTER damped fixed point that
converges in <=~6 iterations across the physical sweep (verified numerically; joint residual reported).
The inner monotone bisection is reused verbatim from defect_equilibrium_codoping.solve_equilibrium_codoping.

REDUCTION. No donor-acceptor pairs (all same sign, or D=1, or E_bind>=0) => x_P=0 => reduces EXACTLY to
the Phase-62 solver, which reduces to V61 at D=1.

RULE 1/2. The binding energies are MACE/DFT-derived and are treated as a DETACHED frozen physics signal:
the pairing partition is computed under no_grad and enters as a (detached) modification of the effective
free concentrations. Gradients to the trainable bounded dH_f head flow ONLY through the final inner E_F
solve, exactly as V61/V62. float64 internally, runs on GPU.
"""
from __future__ import annotations

import torch

from src.models.defect_equilibrium import KB_EV, EG_GA2O3, N_GA_SITES  # noqa: F401
from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping, _residual_codoping
from src.models.dap_coulomb import COULOMB_EV_ANGSTROM, EPS_R_GA2O3_STATIC, D_CATION_NN_ANGSTROM


def _single_pair_xP(a_i, a_j, K):
    """Physical root of the single-pair mass action  x = K (a_i - x)(a_j - x), in [0, min(a_i,a_j)].

    a_i, a_j, K are [B] tensors (site fractions; K dimensionless). Returns x_P [B]. Stable: uses the
    minus-branch quadratic root (<= min(a_i,a_j)); when K->0, x_P->0; when K->inf, x_P->min(a_i,a_j).
    """
    s = a_i + a_j
    b = 1.0 + K * s
    disc = (b * b - 4.0 * K * K * a_i * a_j).clamp_min(0.0)
    x = (b - torch.sqrt(disc)) / (2.0 * K + 1e-300)
    return x.clamp(min=torch.zeros_like(x), max=torch.minimum(a_i, a_j).clamp_min(0.0))


def _solve_pairing(K, c, dop_mask, n_sweep: int = 30):
    """Gauss-Seidel pairing partition for a multiset of dopants.

    Args:
        K:        [B,D,D] symmetric association constants (0 where no pairing / unbound).
        c:        [B,D] total cation fractions.
        dop_mask: [B,D] 1 real / 0 pad.
        n_sweep:  Gauss-Seidel sweeps over the (i<j) pair list.
    Returns:
        x_P:    [B,D,D] symmetric bound-pair fractions.
        c_free: [B,D] free (unpaired) cation fractions.
    The coupled system x_P_ij = K_ij (c_i - Sum_{j'} x_P_ij')(c_j - Sum_{i'} x_P_i'j) is solved by sweeping
    the single-pair quadratic with availabilities that exclude the current pair's own amount (so each update
    is the exact local root). Converges monotonically for dilute fractions; D is small (<=~5) so we loop pairs.
    """
    B, D = c.shape
    x_P = torch.zeros(B, D, D, dtype=c.dtype, device=c.device)
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    if not pairs:
        return x_P, c.clone()
    for _ in range(n_sweep):
        for (i, j) in pairs:
            Kij = K[:, i, j]
            if torch.all(Kij <= 0):
                continue
            paired_i = x_P[:, i, :].sum(dim=1) - x_P[:, i, j]      # i's pairs excluding (i,j)
            paired_j = x_P[:, j, :].sum(dim=1) - x_P[:, i, j]
            a_i = (c[:, i] - paired_i).clamp_min(0.0) * dop_mask[:, i]
            a_j = (c[:, j] - paired_j).clamp_min(0.0) * dop_mask[:, j]
            xij = _single_pair_xP(a_i, a_j, Kij.clamp_min(0.0))
            xij = torch.where(Kij > 0, xij, torch.zeros_like(xij))
            x_P[:, i, j] = xij
            x_P[:, j, i] = xij
    c_free = (c - x_P.sum(dim=2)).clamp_min(0.0)
    return x_P, c_free


def solve_equilibrium_complex(dH_f, T_K, log_pO2, z_dop, c_dop_frac, e_level,
                              E_bind_neutral=None, dop_mask=None, q_vals=None,
                              E_g: float = EG_GA2O3, n_bisect: int = 60,
                              n_outer: int = 40, outer_damp: float = 0.6,
                              use_coulomb: bool = True, Z_coord: float = 8.0,
                              eps_r: float = EPS_R_GA2O3_STATIC, d_cation: float = D_CATION_NN_ANGSTROM,
                              pair_tol: float = 1e-9) -> dict:
    """Differentiable self-consistent multi-dopant defect equilibrium WITH donor-acceptor pairing.

    Args (extends solve_equilibrium_codoping):
        E_bind_neutral: [B,D,D] symmetric MACE neutral binding [eV] (negative=bound) between species; None
                        => no pairing (reduces to the Phase-62 solver). Only donor-acceptor entries should
                        be negative; like-charge / isovalent entries 0 (or positive => no pair).
        use_coulomb:    add the point-charge Coulomb correction (ionization-weighted) to the binding.
        Z_coord:        configurational coordination factor in K = Z exp(-Ebind/kT).
        n_outer:        max outer fixed-point iterations; outer_damp damps the c_free update.
    Returns solve_equilibrium_codoping's dict PLUS:
        x_pair [B,D,D], c_free [B,D], bound_frac [B], outer_iters (int), outer_resid (float, last max dc).
    """
    dev = dH_f.device
    z_dop = torch.as_tensor(z_dop, dtype=torch.float64, device=dev)
    c_dop_frac = torch.as_tensor(c_dop_frac, dtype=torch.float64, device=dev)
    e_level = torch.as_tensor(e_level, dtype=torch.float64, device=dev)
    if z_dop.dim() == 1:
        z_dop = z_dop.view(-1, 1); c_dop_frac = c_dop_frac.view(-1, 1); e_level = e_level.view(-1, 1)
    B, D = z_dop.shape
    if dop_mask is None:
        dop_mask = torch.ones_like(z_dop)
    dop_mask = torch.as_tensor(dop_mask, dtype=torch.float64, device=dev)
    T_K = torch.as_tensor(T_K, dtype=torch.float64, device=dev)
    kT = (KB_EV * T_K).clamp_min(1e-4)                                  # [B]

    # --- pairing pre-conditioner (DETACHED frozen physics signal, Rule 1) -------------------------
    if E_bind_neutral is None or D < 2:
        x_pair = torch.zeros(B, D, D, dtype=torch.float64, device=dev)
        c_free = c_dop_frac
        outer_iters, outer_resid = 0, 0.0
    else:
        E_bind_neutral = torch.as_tensor(E_bind_neutral, dtype=torch.float64, device=dev)
        # per-pair Coulomb energy from the (fixed) ionized charges: donor(+1)-acceptor(-1) -> attractive.
        zz = z_dop.view(B, D, 1) * z_dop.view(B, 1, D)                  # [B,D,D] sign products
        E_coul = zz * (COULOMB_EV_ANGSTROM / (eps_r * d_cation)) if use_coulomb \
            else torch.zeros(B, D, D, dtype=torch.float64, device=dev)
        # pairs only between opposite-sign (donor-acceptor) species with a negative neutral binding.
        da_mask = ((z_dop.view(B, D, 1) * z_dop.view(B, 1, D)) < 0).double()
        da_mask = da_mask * dop_mask.view(B, D, 1) * dop_mask.view(B, 1, D)
        kTm = kT.view(B, 1, 1)
        with torch.no_grad():
            c_free = c_dop_frac.clone()
            last = c_free.clone()
            outer_iters = 0
            for it in range(n_outer):
                outer_iters = it + 1
                # ionized fractions at the current E_F (use the codoping solver's dop_ion magnitude)
                sol = solve_equilibrium_codoping(dH_f.detach(), T_K, log_pO2, z_dop, c_free, e_level,
                                                 dop_mask=dop_mask, q_vals=q_vals, E_g=E_g, n_bisect=n_bisect)
                f_ion = sol["dop_ion"].abs().clamp(0.0, 1.0)            # [B,D] ionized fraction magnitude
                g_ion = f_ion.view(B, D, 1) * f_ion.view(B, 1, D)      # [B,D,D] joint-ionization factor
                E_eff = E_bind_neutral + g_ion * E_coul                # [B,D,D] (Coulomb active when ionized)
                K = Z_coord * torch.exp(-E_eff / kTm) * da_mask        # [B,D,D] (0 where not a bound DA pair)
                K = torch.where(E_eff < 0, K, torch.zeros_like(K))     # only bound channels pair
                _, c_free_new = _solve_pairing(K, c_dop_frac, dop_mask)
                c_free = (1.0 - outer_damp) * c_free + outer_damp * c_free_new
                dc = (c_free - last).abs().max().item()
                last = c_free.clone()
                if dc < pair_tol:
                    break
            outer_resid = float(dc)
            # recompute final pairing matrix at the converged free pool (for diagnostics)
            sol = solve_equilibrium_codoping(dH_f.detach(), T_K, log_pO2, z_dop, c_free, e_level,
                                             dop_mask=dop_mask, q_vals=q_vals, E_g=E_g, n_bisect=n_bisect)
            f_ion = sol["dop_ion"].abs().clamp(0.0, 1.0)
            g_ion = f_ion.view(B, D, 1) * f_ion.view(B, 1, D)
            E_eff = E_bind_neutral + g_ion * E_coul
            K = torch.where(E_eff < 0, Z_coord * torch.exp(-E_eff / kTm) * da_mask, torch.zeros_like(E_coul))
            x_pair, c_free = _solve_pairing(K, c_dop_frac, dop_mask)
        # keep gradient to c_dop_frac (identity minus a detached constant); pairing itself is frozen (Rule 1)
        c_free = (c_dop_frac - x_pair.detach().sum(dim=2)).clamp_min(0.0)

    # --- final DIFFERENTIABLE inner solve at the (frozen) free concentrations ----------------------
    out = solve_equilibrium_codoping(dH_f, T_K, log_pO2, z_dop, c_free, e_level,
                                     dop_mask=dop_mask, q_vals=q_vals, E_g=E_g, n_bisect=n_bisect)
    minc = torch.minimum(c_dop_frac, c_dop_frac).clamp_min(1e-30)  # placeholder; real bound_frac below
    paired_total = x_pair.sum(dim=(1, 2)) * 0.5                    # each pair counted once
    total_dop = (c_dop_frac * dop_mask).sum(dim=1).clamp_min(1e-30)
    bound_frac = (2.0 * paired_total / total_dop).clamp(0.0, 1.0)  # fraction of dopant atoms in bound pairs
    out.update(x_pair=x_pair, c_free=c_free, bound_frac=bound_frac,
               outer_iters=outer_iters, outer_resid=outer_resid)
    return out
