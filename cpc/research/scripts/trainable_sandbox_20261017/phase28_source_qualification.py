"""Versioned source qualification; raw observations never repair each other.

Rounding intervals are a display-precision diagnostic, not experimental CIs.
The charge-budget check is a deliberately restricted uniform one-band model.
Neither check establishes a unique cause for a source inconsistency.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e

ROOT = e.ROOT
OUT = e.BASE / 'phase28_paired_transport'
DATA = OUT / 'source_observations_v1.json'
AUDIT = OUT / 'source_qualification_v1'
ELEMENTARY_CHARGE_C = Decimal('1.602176634e-19')


def declare():
    data = json.loads(DATA.read_text())
    assert len(data['conditions']) == 10
    inputs = [DATA, OUT / 'input_audit_protocol_before_reading.json',
        OUT / 'literature/Je2023_Sn_ribbons.pdf',
        OUT / 'literature/RSC_C5CE01106J_crossref.json',
        e.BASE / 'phase25_shared_process/data/conditions33.csv',
        e.BASE / 'artifact_manifest_phase27.json']
    value = dict(created_utc=e.now(), stage='reported same-condition transport qualification',
        primary_goal='ML+Sandbox dopant-property prediction;nH primary',
        retrospective=True, source_leads_chosen_without_predictions=True,
        source_conditions=10, no_model_predictions_or_refits=True,
        data_policy='preserve reported values and missing/not measurable entries;no nearest joins or repairs',
        rounding_assumption='each printed positive decimal rounded to nearest last displayed unit;not measured uncertainty or a claim about author rounding practice',
        rounding_identity='q=e*nH*muH*rho;interval product over all three printed rounding bins;1 must be inside for display rounding alone to suffice',
        identity_assumptions='same reported Hall convention,conducting geometry and measurement condition;no rH=1 or one-carrier microscopic assumption required for the definition identity',
        charge_budget_assumptions='diagnostic only:uniform material,singly positive Si/Sn,all other defects nonpositive,p negligible,extra background fixed1e14cm^-3,and the current scalar model rH>=1',
        charge_budget_conclusion='if nH exceeds the generous upper donor sum,that restricted model cannot represent the observation;not a proof the measurement is wrong or Fe alone explains it',
        candidate_policy='RSC A/B may enter a separately declared conditional frozen transfer;unknown per-sample growthT,pO2,H and quantitative carbon explicit;Je raw paired conditions retained outside current single-dopant input domain',
        old33_roles_and_results_unchanged=True, derived_sigma_is_not_an_independent_label=True,
        source_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))},
        no_new_DFT_MD=True, no_new_material_measurements=True, goal_complete=False)
    target = AUDIT / 'protocol_before_calculation.json'
    assert not target.exists(), target
    e.write(target, value)
    return value


def printed_interval(value):
    """Half a last displayed decimal unit, retaining scientific notation."""
    v = Decimal(value)
    half = Decimal(10) ** v.as_tuple().exponent / Decimal(2)
    assert v > half > 0
    return v - half, v + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print('Declared10source conditions;no model predictions or fits', flush=True)
        return
    protocol = json.loads((AUDIT / 'protocol_before_calculation.json').read_text())
    for kind in ('source_hashes', 'code_hashes'):
        for path, digest in protocol[kind].items():
            assert e.sha(ROOT / path) == digest, path
    data = json.loads(DATA.read_text())
    rows = []
    with localcontext() as ctx:
        ctx.prec = 55
        for item in data['conditions']:
            values = item['reported']
            pair = all(values.get(k) is not None for k in ('Sn_cm3', 'nH_cm3', 'muH_cm2Vs'))
            row = dict(source_id=item['source_id'], condition_id=item['condition_id'],
                printed_sample=item['sample'], provenance_pair_level=item['pair_level'],
                reported_condition_triple=pair, intentionally_Sn_doped=item['intentionally_Sn_doped'],
                Sn_cm3=float(values['Sn_cm3']) if values.get('Sn_cm3') else None,
                Si_cm3=float(values['Si_cm3']) if values.get('Si_cm3') else None,
                Fe_cm3=float(values['Fe_cm3']) if values.get('Fe_cm3') else None,
                nH_cm3=float(values['nH_cm3']) if values.get('nH_cm3') else None,
                muH_cm2Vs=float(values['muH_cm2Vs']) if values.get('muH_cm2Vs') else None,
                rho_Ohm_cm=float(values['rho_Ohm_cm']) if values.get('rho_Ohm_cm') else None,
                Hall_T_K_reported=item['Hall_T_K_reported'],
                RT_explicit=item['RT_explicit'],
                identity_applicable=False, identity_q=None, identity_q_lo=None, identity_q_hi=None,
                rounding_only_identity_possible=None, min_abs_log10_identity_after_rounding=None,
                generous_SiSn_background_bound_cm3=None,
                restricted_uniform_charge_budget_exceeded=None,
                minimum_extra_positive_charge_cm3=None,
                conditional_frozen_transfer_candidate=pair and item['source_id']=='Gogova2015',
                main_cohort_admitted=False, actual_new_prediction=False,
                missing_or_scope=item['scope_notes'])
            if all(values.get(k) is not None for k in ('nH_cm3', 'muH_cm2Vs', 'rho_Ohm_cm')):
                ns, mus, rhos = (Decimal(values[k]) for k in ('nH_cm3', 'muH_cm2Vs', 'rho_Ohm_cm'))
                bounds = [printed_interval(values[k]) for k in ('nH_cm3', 'muH_cm2Vs', 'rho_Ohm_cm')]
                lo = ELEMENTARY_CHARGE_C
                hi = ELEMENTARY_CHARGE_C
                for a, b in bounds:
                    lo *= a
                    hi *= b
                possible = lo <= Decimal(1) <= hi
                distance = Decimal(0) if possible else (lo.log10() if lo > 1 else -hi.log10())
                row.update(identity_applicable=True,
                    identity_q=float(ELEMENTARY_CHARGE_C*ns*mus*rhos),
                    identity_q_lo=float(lo), identity_q_hi=float(hi),
                    rounding_only_identity_possible=possible,
                    min_abs_log10_identity_after_rounding=float(distance))
            if values.get('Si_cm3') is not None and values.get('Sn_cm3') is not None and values.get('nH_cm3') is not None:
                nlo = printed_interval(values['nH_cm3'])[0]
                upper = printed_interval(values['Sn_cm3'])[1]+printed_interval(values['Si_cm3'])[1]+Decimal('1e14')
                row.update(generous_SiSn_background_bound_cm3=float(upper),
                    restricted_uniform_charge_budget_exceeded=nlo > upper,
                    minimum_extra_positive_charge_cm3=float(max(nlo-upper,Decimal(0))))
            rows.append(row)
    e.csvwrite(AUDIT / 'condition_audit.csv', rows)
    summary = dict(created_utc=e.now(), source_conditions=len(rows),
        reported_condition_triples=sum(x['reported_condition_triple'] for x in rows),
        intentionally_Sn_doped_triples=sum(x['reported_condition_triple'] and x['intentionally_Sn_doped'] for x in rows),
        identity_checked=sum(x['identity_applicable'] for x in rows),
        identity_not_explained_by_display_rounding=sum(x['rounding_only_identity_possible'] is False for x in rows),
        restricted_uniform_charge_budget_exceeded=sum(x['restricted_uniform_charge_budget_exceeded'] is True for x in rows),
        conditional_frozen_transfer_candidates=[x['condition_id'] for x in rows if x['conditional_frozen_transfer_candidate']],
        raw_observational_pairing_is_separate_from_model_domain=True,
        main_cohort_new_admissions=0, actual_new_training=0, actual_new_predictions=0,
        actual_new_sigma_independent_labels=0, original_Sn_fullchain_validation_still_not_established=True,
        all_missing_and_unfavorable_rows_retained=True,
        identity_conflict_is_not_unique_error_attribution=True, goal_complete=False)
    assert len(rows) == 10
    assert summary['reported_condition_triples'] == 7
    assert summary['intentionally_Sn_doped_triples'] == 6
    assert summary['identity_checked'] == 5
    assert len(summary['conditional_frozen_transfer_candidates']) == 2
    e.write(AUDIT / 'summary.json', summary)
    checks = dict(passed=True, all10rows_retained=True,
        declared_hashes_verified=True, all7reported_pairs_preserved=True,
        original_main33_not_modified=True, no_corrected_labels_invented=True,
        summary_is_not_a_claim_all_source_values_are_consistent=True,
        condition_audit_sha256=e.sha(AUDIT / 'condition_audit.csv'),
        summary_sha256=e.sha(AUDIT / 'summary.json'))
    e.write(AUDIT / 'audit.json', checks)
    print(json.dumps(summary,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
