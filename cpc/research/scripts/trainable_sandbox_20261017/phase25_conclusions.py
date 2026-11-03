"""Evidence-specific prediction conclusions after all transfer audits complete."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
OUT=e.BASE/'phase25_shared_process';FACTS=[]


def v(path,sel,column):
    p=OUT/path;d=read(p)
    for k,x in sel.items():d=d[d[k].eq(x)]
    assert len(d)==1,(p,sel,len(d));x=float(d[column].iloc[0])
    FACTS.append(dict(path=str(p.relative_to(ROOT)),sha256=e.sha(p),selector=sel,column=column,value=x))
    return f'{x:.6f}'


def n(method,b):return v('prediction/summary.csv',dict(scheme='cluster',variant=method,budget=b,scope='common20'),'nH_MAE')


def main():
    for p in ('delivery/cohort_replay_v2/audit.json','external_Azizie_v2/audit.json','Leedy_label_sensitivity_v2/audit.json','delivery/final_checks/audit.json'):
        assert json.loads((OUT/p).read_text())['passed'],p
    s=[]
    s.append('## 12. 实际结果支持的判断与下一项预测改进')
    s.append('本轮没有建立新的共享库存学习增量。关联组留出、同common20的固定物理零锚点为'+n('FixedPhysical',0)+'dex，小于所测直接Ridge '+n('DirectRidge',0)+'、剂量直线 '+n('DoseLine',0)+'及中位数激活比例 '+n('MedianActivation',0)+'。这支持当前物理先验在零锚点任务中有预测价值，不证明新增共享函数有效或物理普遍优于ML。工艺库存模型在两套方案的全部11个外层折选择精确零修正参考；这些折不是11项独立研究。')
    s.append('少锚点结论不同。内部工艺模型1/2锚点为'+n('InventoryProcess_internal',1)+'/'+n('InventoryProcess_internal',2)+'dex，弱于中位数激活比例 '+n('MedianActivation',1)+'/'+n('MedianActivation',2)+'，以及UnitDose加允许来源偏移的 '+n('UnitDose',1)+'/'+n('UnitDose',2)+'。跨预算和来源的稳定增量没有出现，不能将源内拟合或某个来源的解除惩罚结果包装为全队列成功。')
    sel=dict(scheme='cluster',variant='InventoryProcess_internal',scope='common20',budget=2)
    mu=v('prediction/paired_chain_summary.csv',sel,'physical_or_native_mu_MAE')
    ct=v('prediction/paired_chain_summary.csv',sel,'common_CT_mu_MAE')
    s.append('固定共同查询仅4组mu配对/3 DOI。内部两锚点原生mu误差 '+mu+'dex，统一CT读出为 '+ct+'。本队列下原生输运读出有条件相对信号，但绝对误差仍大，也未建立新共享关系在完整mu链上的稳定收益。全部9配对与共同4配对不混淆；sigma仍是派生量。')
    ext=dict(context='cluster::without:related_AFRL_Cornell_Wright',input_tier='learned_process_proxy_12',scope='common_query',budget=2)
    a=v('external_Azizie_v2/summary.csv',dict(ext,variant='InventoryProcess_internal'),'nH_MAE')
    b=v('external_Azizie_v2/summary.csv',dict(ext,variant='MedianActivation'),'nH_MAE')
    s.append('Azizie的过程化学代理、删除整个关联组、同common10条件下，内部模型两锚点为'+a+'dex，中位数激活比例为'+b+'；冻结转移未修复差距。约数化学输入的高剂量k也保留明显失配。不能只展示full上下文或把来源关系、O3/未知历史及输入误差唯一归因为一种机制。')
    s.append('训练内选族策略同样不稳定，不能以BestOverall名字承诺外层最优。代码部署、全部预测回放、守恒/IFT核对和新的信息都已实际完成，但这些不使系统goal自动完成，也不证明本方法的普遍创新优势。当前支持的最强结论是：固定物理在零锚点有用，新增内部共享和来源坐标仍需重设计；一个单一方法尚未同时解决各来源和各锚点预算。')
    s.append('下一步首先检验一个具体假设：直接学习池内孤立原子份额，能否消除相对热平衡参考赔率随剂量变化而产生的锚点弱响应/查询强响应，并在同信息预算下提高真实nH预测。与旧alpha单向转移、有限相对赔率场的数学关系和非等价边界已明确。先完成状态/梯度/可达性门槛，再另立协议训练；不同时加入神经网络、空间PDE或新微观速率。该方案尚未实现或验证，不保证增益。具体执行见docs/ML_SANDBOX_POOL_FRACTION_NEXT_20261018.md。')
    p=OUT/'analysis/predictive_conclusions.md';p.write_text('\n\n'.join(s)+'\n')
    choices=json.loads((OUT/'prediction/shared_choices.json').read_text())
    process=[r for r in choices if r['held'] is not None and r['method']=='InventoryProcess']
    assert len(process)==11 and all(r['reg'] is None for r in process)
    e.write(OUT/'analysis/conclusion_receipt.json',dict(notes_sha256=e.sha(p),facts=FACTS,
        process_outer_choices=11,all_process_reference_fallback=True,
        new_shared_predictive_increment_established=False,stable_internal_source_increment_established=False,
        independent_blind_or_lab_validation=False,goal_complete=False,
        next_predictive_work='first gate absolute isolated-atom share learning versus bounded relative-reference coordinates,then matched nH prediction training if gates pass',
        next_theory='docs/ML_SANDBOX_POOL_FRACTION_NEXT_20261018.md',code_sha256=e.sha(Path(__file__))))
    print('Actual evidence conclusions saved',len(FACTS),'traceable claims;goal active',flush=True)


if __name__=='__main__':main()
