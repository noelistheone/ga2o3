"""Follow actual public figure/SI links, preserving initial retrieval failures."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from phase29_additional_source_acquisition import OUT, acquire
import phase2_energy as e


def main():
    urls={
        'Wang2025_Fig3.jpg':'https://www.researchgate.net/publication/390769541/figure/download/fig3/AS%3A11431281376741923%401744673713996/a-Resistivity-carrier-concentration-and-Hall-mobility-of-the-b-Ga2O3-thin-films-b.jpg',
        'Liu2025_SI_article.json':'https://api.figshare.com/v2/articles/29715608',
    }
    protocol=OUT/'protocol_before_followup.json'
    assert not protocol.exists(),'preserve previous follow-up'
    e.write(protocol,dict(created_utc=e.now(),candidate_urls=urls,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in
            (Path(__file__),Path(__file__).with_name('phase29_additional_source_acquisition.py'))},
        no_model_training_or_cohort_change=True,no_predictions_used=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(lambda x:acquire(*x),urls.items()))
    metadata=json.loads((OUT/'acquired/Liu2025_SI_article.json').read_text())
    files=metadata.get('files',[])
    print('Liu SI files',[(x['name'],x['download_url']) for x in files],flush=True)
    for i,item in enumerate(files):
        name='Liu2025_SI_'+str(i)+Path(item['name']).suffix
        records.append(acquire(name,item['download_url']))
    e.write(OUT/'acquisition_followup.json',dict(created_utc=e.now(),records=records,
        SI_article_id=29715608,source_qualification_pending=True,admitted_conditions=0,
        fits_or_predictions=0,goal_complete=False))


if __name__=='__main__':main()
