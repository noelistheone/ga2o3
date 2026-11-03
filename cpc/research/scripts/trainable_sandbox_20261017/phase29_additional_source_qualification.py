"""Record source-reading decisions; none changes the current frozen experiment."""
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e
from phase29_additional_source_acquisition import OUT


def main():
    target=OUT/'qualification.json'
    assert not target.exists(),'preserve prior qualification decision'
    evidence=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.pyc')
    e.write(OUT/'qualification_protocol_before.json',dict(created_utc=e.now(),
        code_sha256=e.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in evidence},
        source_reading_not_material_validation=True,no_predictions_or_model_selection=True))
    old=OUT/'Liu2025_SI_media/image2.tiff'
    converted=old.with_suffix('.png')
    with Image.open(old) as a,Image.open(converted) as b:
        assert a.size==b.size==(1902,579)
        assert np.array_equal(np.asarray(a),np.asarray(b))
    sources=[dict(id='Wang2025',doi='10.1063/5.0257824',
        url='https://doi.org/10.1063/5.0257824',
        reading='publisher abstract and first-page preview;Fig3 caption;full numerical graph not acquired',
        status='pending_full_graph_and_methods',
        reason='SIMS/Hall pairing is a lead only;95percent activation is not an independent chemical label',
        chemical_incorporation_cm3=None,admitted_conditions=0,
        figure_download_status=403,initial_crossref_status=429,
        known_related_source='same Xidian pulsed-growth author group as2024S&I/CGD leads;sample reuse not audited'),
        dict(id='Liu2025',doi='10.1063/5.0271487',
        url='https://doi.org/10.1063/5.0271487',
        SI_url='https://doi.org/10.60893/figshare.apm.c.7958438',
        reading='publisher Methods/results andoriginalSI;FigS2 actualembeddedimage visually read',
        status='not_eligible_as_absolute_chemical_input',
        reason='FigS2 usesNormalizedintensity versusSputterTime;no calibratedSncm-3 fromthisgraph',
        chemical_incorporation_cm3=None,admitted_conditions=0,
        additional_scope='S2Hallunmeasurable isnotzero;S3reportedCV isnotHall;relatedannealedconditions andsurfacechannel needexplicitgeometry'),
        dict(id='Khan2026v1',doi='10.48550/arXiv.2608.07823',
        url='https://arxiv.org/abs/2608.07823v1',
        reading='28pagePDF acquired;methods pp4-5,XPS pp7-8,Tables1-3 pp19-21;Tables1/3 images actuallyviewed',
        status='not_eligible_as_absolute_chemical_input',
        reason='Table1Sn/(Sn+Ga)wtpercent issourceloading;Table3ND/NA aretransport-fitparameters,notindependentSIMS',
        chemical_incorporation_cm3=None,reported_Hall_conditions=5,admitted_conditions=0,
        additional_scope='preprint;XPSgivesGa/Oratio,notcalibratedvolumetricSn;fitteddonordensities cannotbeusedasindependentinputs')]
    acquisition=json.loads((OUT/'acquisition.json').read_text())
    followup=json.loads((OUT/'acquisition_followup.json').read_text())
    for row in [*acquisition['records'],*followup['records']]:
        if 'path' in row:assert e.sha(ROOT/row['path'])==row['sha256']
    protocol=json.loads((OUT/'qualification_protocol_before.json').read_text())
    for name,digest in protocol['input_hashes'].items():assert e.sha(ROOT/name)==digest,name
    result=dict(passed=True,created_utc=e.now(),source_records=sources,
        meaning_of_pass='source identity,reading limits andfile integrity;notdataeligibility orpredictionaccuracy',
        admitted_conditions=0,primary35_conditions_and_common20_unchanged=True,
        no_new_fits_or_predictions=True,unavailable_sources_andHTTPfailures_retained=True,
        lossless_TIFF_PNG_pixel_identity=True,
        actually_viewed=[str(p.relative_to(ROOT)) for p in
            (converted,OUT/'Khan2026v1_page19.png',OUT/'Khan2026v1_page21.png')],
        evidence_sha256=protocol['input_hashes'],goal_complete=False)
    e.write(target,result)
    lines=['# 新Sn来源资格核查：不改变本轮训练', '',
        '按输入与配对完整性选文献，没有运行预测来挑来源。通过表示本次阅读与文件记录一致，不表示已取得合格新标签。', '',
        '| 来源 | 实际阅读范围 | 资格判断 |', '| --- | --- | --- |',
        '| [Wang2025](https://doi.org/10.1063/5.0257824) | 摘要、首页预览及Fig3图注；原始图获取403 | 暂待完整方法与数值图，不能用激活比例倒推独立化学输入 |',
        '| [Liu2025](https://doi.org/10.1063/5.0271487) | 正文方法/结果、原SI及实际查看FigS2 | SIMS纵轴为归一化强度；不能转为绝对Sn浓度。S2不可测不记零，S3的C-V值不作Hall标签 |',
        '| [Khan2026v1](https://arxiv.org/abs/2608.07823v1) | 方法、XPS、表1–3；表1/3图像已查看 | 源料质量比与输运拟合施主浓度都不能替代独立化学测量 |', '',
        '本次新增入训练条件为0；当前35条件、49输入情景、common20及固定主方法完全不改。新文献与既有曲线已经查看，后续使用须声明回顾性研究。', '',
        '原始下载、HTTP失败、DOI/SI指针和文件SHA均保留；TIFF只作无损格式转换供查看，逐像素一致。未重新拟合、未预测、未改npj稿件。', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('Additionalsource audit:3leads,0admissions;boundexperimentunchanged',flush=True)


if __name__=='__main__':main()
