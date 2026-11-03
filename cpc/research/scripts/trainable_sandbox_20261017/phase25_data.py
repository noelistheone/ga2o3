"""Qualify a separate, retrospective 33-condition chemical/Hall task.

The original 29-row task and every existing role remain immutable. Published
condition pairing does not invent raw wafer IDs or experimental covariance.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
import phase16_prediction as parent

BASE = e.BASE
OUT = BASE / 'phase25_shared_process'
LEEDY = '10.1063/1.5047214'
MCCANDLESS = '10.1063/5.0101132'
OLD = parent.PRED[parent.PRED.kind.eq('RT')].copy().reset_index(drop=True)
OLD_ROLES = read(BASE / 'episodes_v1.csv')
SOURCE = BASE / 'phase19_delivery/leedy2018_v2'


def cluster(family):
    return 'related_AFRL_Cornell_Wright' if family in (LEEDY, MCCANDLESS) else family


def table():
    return read(OUT / 'data/conditions33.csv')


def roles():
    return read(OUT / 'data/episodes33.csv')


def main():
    target = OUT / 'data/qualification.json'
    if target.exists():
        d = json.loads(target.read_text())
        for name, h in d['input_hashes'].items():
            assert e.sha(ROOT / name) == h, name
        assert len(table()) == 33
        print('Verified existing phase25 data contract', flush=True)
        return
    assert len(OLD) == 29 and OLD.sample_id.is_unique and OLD.family.nunique() == 5
    new = read(SOURCE / 'samples.csv')
    # Use the stable string suffix, never the numeric-looking sample_label.
    new['target_label'] = new.sample_id.str.split('::target-', regex=False).str[-1]
    assert set(new.target_label) == {'0.01', '0.025', '0.05', '0.1', '1'}
    assert len(new) == 5 and new.uniform_scalar_input.eq(False).sum() == 1
    assert new.loc[~new.uniform_scalar_input, 'target_label'].item() == '0.05'
    rows = []
    for r in new.itertuples():
        x = dict(sample_id=r.sample_id, family=LEEDY, sample_label='target-' + r.target_label,
            element='Si', incorporated_cm3=float(r.incorporated_cm3), growth_T_C=550.,
            Tobs_K=300., kind='RT', endpoint='nH', obs_id='RT::' + r.sample_id,
            target=float(np.log10(r.observed_nH_cm3)), observed_log_nH=float(np.log10(r.observed_nH_cm3)),
            carrier_cm3=float(r.observed_nH_cm3), mu_cm2Vs=float(r.observed_muH_cm2Vs),
            measurement_T_K=300., H_known=False, H_cm3=0., carbon_known=False, carbon_cm3=0.,
            mobility_eligible=True, method='PLD', substrate='(010) Fe-compensated beta-Ga2O3',
            pO2_atm=float(r.measured_pO2_atm), scenario_pO2_atm=float(r.measured_pO2_atm),
            pressure_status='reported molecular O2/Ar mixture:0.05*1.3Pa',
            source_input_tier='published SIMS/Hall condition pairing; no raw wafer identifiers',
            thermal_cap_K=823.15, anneal_note='470C/60s N2 contact RTA; full cooling history unavailable',
            source_file=str((BASE / 'phase19_delivery/literature/Leedy2018_APL_Materials_101102.pdf').relative_to(ROOT)),
            source_locator='article p7 Fig7 and p8 Fig8;1wt Hall pair from abstract/p3 prose',
            nominal_weight_pct_SiO2=float(r.target_label), concentration_kind='printed representative SIMS film concentration',
            incorporated_reading_status='reported representative chemical value; experimental covariance unknown',
            observation_T_status='300K comparison convention; exact Hall instrument T not reported in inspected methods',
            independent_physical_ID_verified=False, condition_pairing_verified=True,
            included_primary=bool(r.uniform_scalar_input),
            not_blind=True, postfilm_anneal_C=470., postfilm_anneal_seconds=60.)
        rows.append(x)
    new_all = pd.DataFrame(rows)
    primary = new_all[new_all.included_primary].copy()
    assert len(primary) == 4
    data = pd.concat([OLD, primary], ignore_index=True)
    data['related_cluster'] = data.family.map(cluster)
    data['new_phase25_condition'] = data.family.eq(LEEDY)
    data['oxygen_input_kind'] = np.where(data.family.eq(LEEDY),
        'reported molecular gas; used conditionally in effective endpoint closure',
        'unchanged old1e-5atm effective scenario; not asserted measured equilibriumO2')
    assert len(data) == 33 and data.sample_id.is_unique and data.family.nunique() == 6 and data.related_cluster.nunique() == 5
    # Preserve the old data values exactly in their existing columns.
    pd.testing.assert_frame_equal(data.loc[:28, OLD.columns].reset_index(drop=True), OLD, check_dtype=False)
    ep = OLD_ROLES.to_dict('records')
    anchor_labels = {0:[], 1:['0.025'], 2:['0.025', '0.1']}
    for b, anchors in anchor_labels.items():
        for r in primary.itertuples():
            label = r.sample_id.split('::target-')[-1]
            ep.append(dict(sample_id=r.sample_id, family=LEEDY, budget=b,
                role='anchor' if label in anchors else 'query'))
    ep = pd.DataFrame(ep)
    assert len(ep) == len(OLD_ROLES) + 12
    for b, expected in ((0,33),(1,26),(2,20)):
        q = ep[ep.budget.eq(b) & ep.role.eq('query')]
        assert len(q) == expected, (b, len(q))
    anchor = primary[primary.sample_id.str.endswith('::target-0.025')].iloc[0]
    second = primary[primary.sample_id.str.endswith('::target-0.1')].iloc[0]
    separation = float(np.log10(second.incorporated_cm3 / anchor.incorporated_cm3))
    assert separation > .5
    e.csvwrite(OUT / 'data/conditions33.csv', data.to_dict('records'))
    e.csvwrite(OUT / 'data/episodes33.csv', ep.to_dict('records'))
    e.csvwrite(OUT / 'data/new_all5_with_nonuniform.csv', new_all.to_dict('records'))
    inputs = [BASE / 'dataset_v1.csv', BASE / 'episodes_v1.csv', SOURCE / 'samples.csv', SOURCE / 'protocol.json',
        SOURCE / 'source_metadata_amendment.json', BASE / 'phase19_delivery/literature/Leedy2018_APL_Materials_101102.pdf',
        BASE / 'phase19_delivery/literature/Leedy2018_Fig8_original.png',
        BASE / 'phase24_chemical_process/inventory_scope/audit.json']
    e.write(target, dict(passed=True, created_utc=e.now(), conditions=33, old_conditions_unchanged=29, added_conditions=4,
        publication_families=6, conservative_related_clusters=5, independent_labs_verified=False,
        source_DOI=LEEDY, nominal_target_is_not_chemical_input=True, nonuniform_condition_retained_separately='target-0.05',
        scalar_input_qualification='Fig7 representative SIMS plus Fig8/printed Hall matched by target/growth condition; raw wafer IDs unavailable',
        pairing_grade='source-condition correspondence; not raw instrument linkage or proof of one-to-one physical wafer identity',
        repeated_or_missing_observations_not_invented=True,
        label_conflict=dict(condition='target-0.025', primary='Fig8 reading', retained_alternative_nH=3.25e19,
            retained_alternative_mu=20., effect='predeclared prose sensitivity, not best-error label selection'),
        pressure_atm=float(.05 * 1.3 / 101325.), observation_T_status='300K comparison convention, not exact reported instrument T',
        postfilm_contact_RTA='470C60sN2', full_thermal_history_known=False,
        new_anchor_labels=anchor_labels, new_anchor_log10_dose_separation=separation, common_queries=20,
        newsource_was_previously_evaluated=True, retrospective=True, no_blind_validation_claim=True,
        source_cluster_reason='Leedy AFRL/WrightState and McCandless Cornell/AFRL/WrightState share broad institutions; conservative association, not identical growth lab assertion',
        no_new_experimental_measurements=True, no_new_DFT_MD=True,
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs}, code_sha256=e.sha(Path(__file__))))
    e.write(OUT / 'data/literature_read_depth.json', dict(created_utc=e.now(),
        Leedy2018=dict(DOI=LEEDY, primary='https://corescholar.libraries.wright.edu/physics/1482/',
            cached_original_read='methods, articlepp3-9, Fig7/Fig8 viewed; printed SIMS and current-work symbols checked; no raw data',
            repository_frontmatter_verified=True),
        McCandless2022=dict(DOI=MCCANDLESS, primary='https://djena.engineering.cornell.edu/papers-new/2022/mccandless2022controlled.pdf',
            read_this_stage='author/affiliation/frontmatter; growth/SIMS/Hall pp2-4; not complete author-code replication'),
        other_four_sources='cached author/affiliation text checked for broad clustering; no claim of exhaustive laboratory network independence'))
    print('Qualified phase25:33 conditions/6 DOI/5 related clusters;old29 and roles unchanged', flush=True)


if __name__ == '__main__':
    main()
