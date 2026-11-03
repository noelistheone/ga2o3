"""Make actual endpoint/species support and inherited guard roles explicit."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE = e.BASE / 'phase27_charge_feasible'


def main():
    delivery = BASE / 'delivery'; audit = delivery / 'documentation_audit.json'
    old = json.loads(audit.read_text()); assert old['passed'] and old['presets'] == 624
    archive = delivery / 'documentation_v1_archive'; assert not archive.exists(); archive.mkdir()
    for path in (delivery / 'README.md', audit, ROOT / 'scripts/trainable_sandbox_20261017/phase27_delivery_docs.py'):
        shutil.copyfile(path, archive / path.name)
    data = e.BASE / 'phase25_shared_process/data'; table = read(data / 'conditions33.csv'); roles = read(data / 'episodes33.csv')
    common = set(roles[roles.budget.eq(2) & roles.role.eq('query')].sample_id)
    rows = []
    for element in ('Si', 'Sn'):
        group = table[table.element.eq(element)]; mu = group[group.mobility_eligible.eq(True)]
        rows.append(dict(element=element, carrier_conditions=len(group), carrier_DOIs=group.family.nunique(),
            fullchain_mu_pairs=len(mu), fullchain_mu_DOIs=mu.family.nunique(),
            common20_mu_pairs=int(mu.sample_id.isin(common).sum()), independent_sigma_labels=0))
    assert sum(r['carrier_conditions'] for r in rows) == 33
    assert rows[0]['fullchain_mu_pairs'] == 9 and rows[1]['fullchain_mu_pairs'] == 0
    role_rows = []
    for budget in (0, 1, 2):
        a = roles[roles.budget.eq(budget)]
        row = dict(budget=budget, anchors=int(a.role.eq('anchor').sum()), guards=int(a.role.eq('guard').sum()),
            queries=int(a.role.eq('query').sum()))
        assert sum(row[k] for k in ('anchors', 'guards', 'queries')) == 33
        assert row['anchors'] == 6 * budget
        role_rows.append(row)
    e.csvwrite(delivery / 'validation_scope.csv', rows); e.csvwrite(delivery / 'role_counts.csv', role_rows)
    guide = (delivery / 'README.md').read_text()
    paragraph = '''## 性质验证覆盖和角色计数

主队列有14个Si条件和19个Sn条件；载流子任务同时包含两种元素。当前9组实际化学输入—Hall迁移率配对全部来自Si，共3 DOI；共同查询只有4组Si配对。Sn完整迁移率链目前没有配对真值，不能把接口能够输出Sn迁移率称作已验证的Sn材料精度。电导仍为派生量，没有新增独立标签。数据计数见validation_scope.csv。

每个DOI仍使用0/1/2个允许Hall锚点，总计0/6/12个。继承的Sn数据源在正预算下另有1个guard点，因此全查询为33/26/20；guard不是额外锚点，其标签不用于该留出来源适应。固定common20比较继续使用完全相同的查询点。角色计数见role_counts.csv，原有角色未改变。

'''
    assert '## 输出与范围\n' in guide
    guide = guide.replace('## 输出与范围\n', paragraph + '## 输出与范围\n', 1)
    (delivery / 'README.md').write_text(guide)
    value = dict(old, version=2, created_utc=e.now(),
        scope_addendum='all9 fullchain mu pairs areSi;Snmu unvalidated;positive budgets retain oneguard',
        parent_documentation_audit_sha256=e.sha(archive / 'documentation_audit.json'),
        guide_sha256=e.sha(delivery / 'README.md'), original_V1_guide_and_audit_preserved=True,
        code_sha256=e.sha(Path(__file__)), code_hashes={str(Path(__file__).relative_to(ROOT)): e.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in (data / 'conditions33.csv', data / 'episodes33.csv',
            archive / 'documentation_audit.json', archive / 'README.md')},
        scope_csv_sha256=e.sha(delivery / 'validation_scope.csv'), role_csv_sha256=e.sha(delivery / 'role_counts.csv'),
        no_predictions_models_scores_or_roles_changed=True)
    e.write(audit, value)
    print('Actual property scope', rows, 'roles', role_rows, 'V1preserved;no numerical change', flush=True)


if __name__ == '__main__': main()
