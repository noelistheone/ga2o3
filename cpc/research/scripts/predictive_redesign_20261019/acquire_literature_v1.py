"""Acquire primary sources, preserving failures and exact reading inputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,subprocess,requests

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
OUT=c.BASE/'literature_v1'
SOURCES=[
 ('cagrad2021','https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf',
  'https://arxiv.org/pdf/2110.14048'),
 ('kendall2018','https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf',
  'https://arxiv.org/pdf/1705.07115'),
 ('ma2016','https://djena.engineering.cornell.edu/papers/2016/apl16_nan_GaOx_transport.pdf',None),
 ('erginsoy1950','https://open.metu.edu.tr/bitstream/handle/11511/119134/PhysRev.79.1013.pdf',None),
 ('look_leedy2019','https://par.nsf.gov/servlets/purl/10104796',None),
]


def acquire(record):
    name,primary,fallback=record;attempts=[];pdf=None
    for i,url in enumerate([primary]+([fallback] if fallback else [])):
        try:
            r=requests.get(url,timeout=35)
            path=OUT/f'{name}_attempt{i}.response';path.write_bytes(r.content)
            ok=r.status_code==200 and r.content.startswith(b'%PDF')
            attempts.append(dict(url=url,final_url=r.url,status=r.status_code,
                bytes=len(r.content),sha256=c.e.sha(path),is_pdf=ok))
            if ok:
                pdf=OUT/(name+'.pdf');pdf.write_bytes(r.content)
                subprocess.run(['pdftotext','-layout',str(pdf),str(pdf.with_suffix('.txt'))],check=True)
                break
        except (requests.RequestException,subprocess.CalledProcessError) as exc:
            attempts.append(dict(url=url,error=repr(exc)))
    return dict(id=name,attempts=attempts,pdf=str(pdf.relative_to(c.ROOT)) if pdf else None,
        acquired=bool(pdf),read_depth='acquired_not_yet_reviewed',primary_source=True)


def main():
    assert not (OUT/'acquisition.json').exists(),'Preserve previous acquisition'
    OUT.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(acquire,SOURCES))
    local=c.ROOT.parent/'Ga2O3-Net/extra-paper/doping_extracted/doping/si/AdvElectMaterials-2024-Tarntair-MaterialPropertiesofnTypeGa2O3EpilayerswithInSituDopingGrownon3.pdf'
    assert local.is_file()
    values.append(dict(id='tarntair2024',local_path=str(local),sha256=c.e.sha(local),
        doi='10.1002/aelm.202300679',acquired=True,read_depth='text Table1/p4 andmicrostructure discussion;PDF visualcheck pending'))
    c.write(OUT/'acquisition.json',dict(created_utc=c.e.now(),sources=values,
        acquired_pdf_count=sum(v['acquired'] for v in values),
        failures_preserved=True,acquisition_does_not_imply_reading=True,
        code_sha256=c.e.sha(Path(__file__)),goal_complete=False))
    print([(v['id'],v['acquired']) for v in values],flush=True)


if __name__=='__main__':main()
