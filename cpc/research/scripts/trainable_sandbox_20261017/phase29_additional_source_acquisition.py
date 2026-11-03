"""Acquire independent Sn leads without changing the bound training cohort."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e

OUT = e.BASE / 'phase29_joint_scattering/additional_source_audit_v1'
URLS = {
    'Wang2025_crossref.json': 'https://api.crossref.org/works/10.1063/5.0257824',
    'Liu2025_crossref.json': 'https://api.crossref.org/works/10.1063/5.0271487',
    'Wang2025_publisher.html': 'https://aipp.silverchair-cdn.com/article-minimal/3344061',
    'Wang2025_figure_download.html': 'https://www.researchgate.net/figure/figure/a-Resistivity-carrier-concentration-and-Hall-mobility-of-the-b-Ga2O3-thin-films-b_fig3_390769541/download',
    'Liu2025_publisher.html': 'https://aipp.silverchair-cdn.com/article-minimal/3358416',
    'Liu2025_SI_collection.json': 'https://api.figshare.com/v2/collections/7958438/articles',
    'Khan2026v1.pdf': 'https://arxiv.org/pdf/2608.07823v1',
}


def acquire(name, url):
    row = dict(name=name, url=url, created_utc=e.now())
    try:
        response = requests.get(url, timeout=(10, 40),
            headers={'User-Agent': 'PhysicsResearchSourceAudit/1.0'})
        row.update(status_code=response.status_code, final_url=response.url,
            content_type=response.headers.get('content-type'), bytes=len(response.content))
        # Keep failed responses as well. A file ending in .pdf is certified as
        # a PDF only by its actual bytes, not by the request URL or filename.
        destination = OUT / 'acquired' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists(), str(destination)
        destination.write_bytes(response.content)
        row.update(path=str(destination.relative_to(ROOT)), sha256=e.sha(destination),
            is_pdf=response.content.startswith(b'%PDF-'),
            HTTP_success=bool(response.ok))
        if name.endswith('.json'):
            try:
                parsed = response.json()
                row['JSON_parsed'] = True
                if isinstance(parsed, list): row['JSON_records'] = len(parsed)
            except requests.exceptions.JSONDecodeError:
                row['JSON_parsed'] = False
    except requests.RequestException as exc:
        row.update(HTTP_success=False, error=repr(exc))
    print(name, row.get('status_code', 'request_error'), row.get('bytes', 0), flush=True)
    return row


def main():
    protocol = OUT / 'protocol_before_acquisition.json'
    assert not protocol.exists(), 'Preserve previous acquisition before a versioned follow-up'
    e.write(protocol, dict(created_utc=e.now(), code_sha256=e.sha(Path(__file__)),
        candidate_urls=URLS, selection='search-visible Sn chemistry/Hall pairing or independent temperature transport;no prediction-score selection',
        inclusion_requires=['beta phase and explicit dopant/sample identity',
            'independent chemical incorporation measurement, not inferred from Hall activation',
            'same-condition Hall density and mobility with geometry and process scope',
            'related samples/sources grouped and unavailable/censored values retained'],
        no_cohort_or_protocol_change=True, no_new_fits_or_predictions=True,
        primary35_conditions_and_common20_unchanged=True,
        all_sources_already_exposed_retrospective=True, goal_complete=False))
    results = []
    with ThreadPoolExecutor(max_workers=len(URLS)) as pool:
        pending = [pool.submit(acquire, name, url) for name, url in URLS.items()]
        for future in as_completed(pending): results.append(future.result())
    e.write(OUT / 'acquisition.json', dict(created_utc=e.now(), records=results,
        full_source_qualification_pending=True, admitted_conditions=0,
        fits_or_predictions=0, goal_complete=False))


if __name__ == '__main__':
    main()
