"""Reviewer point 7: SEMANTIC extraction audit (not just 'the number appears in the quote').
Checks each stored verbatim quote against the row's recorded semantics:
  peak-vs-typical, measurement temperature, Hall-vs-C-V, bulk-vs-film."""
from pathlib import Path as _P
_SB = _P(__file__).resolve().parents[1]          # <root>/Ga2O3-Sandbox
_NET = _SB.parent / "Ga2O3-Net"                  # sibling checkout / same release
import json, re
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path(str(_NET) + '/results/phase65/transport_llm_extracted_v2.csv')
OUT = Path(str(_SB) + '/results/revision_r9/semantic_extraction_audit.json')
df = pd.read_csv(SRC)
df['_has_val'] = df['mu_cm2Vs'].notna() | df['carrier_cm3'].notna()
pool = df[df['_has_val'] & df['quote'].notna() & (df['quote'].astype(str).str.len() > 30)].copy()

PEAK  = re.compile(r'\b(peak|maximum|max\.?|highest|best|optimum|optimal)\b', re.I)
LOWT  = re.compile(r'\bat\s*(?:1?\d{1,2})\s*K\b|\b(77|100|150|200|4\.2)\s*K\b|low[- ]temperature', re.I)
RT    = re.compile(r'room[- ]temperature|\bRT\b|\b(29[0-9]|30[0-9])\s*K\b', re.I)
CV    = re.compile(r'C\s*[-–—]\s*V|capacitance[- ]voltage|Mott[- ]Schottky|CV\s+profil', re.I)
HALL  = re.compile(r'\bHall\b|van\s*der\s*Pauw', re.I)
BULK  = re.compile(r'\bbulk\b|single[- ]crystal|substrate|EFG|Czochralski|float[- ]zone|\bOFZ\b|melt[- ]grown', re.I)
FILM  = re.compile(r'\bfilm\b|epilayer|epitaxial|thin[- ]film|\blayer\b|as[- ]deposited|sputter', re.I)
FILM_METHOD = re.compile(r'MOCVD|MOVPE|HVPE|MBE|LPCVD|PLD|sputter|ALD|CVD|spray|sol[- ]gel', re.I)
BULK_METHOD = re.compile(r'melt|EFG|Czochralski|float|OFZ|bulk|single', re.I)

def audit(r):
    q = str(r['quote']); flags = []
    if PEAK.search(q):
        flags.append('qualifier: quote says peak/maximum but the row carries no peak marker')
    if LOWT.search(q) and not RT.search(q):
        flags.append('temperature: quote names a non-room temperature')
    mk = str(r.get('measurement_kind') or '')
    if CV.search(q) and not HALL.search(q) and re.search(r'hall', mk, re.I):
        flags.append('method: quote is C-V but measurement_kind says Hall')
    gm = str(r.get('growth_method') or '')
    if BULK.search(q) and not FILM.search(q) and FILM_METHOD.search(gm):
        flags.append('sample form: quote reads bulk/single-crystal but growth_method is a film method')
    if FILM.search(q) and not BULK.search(q) and BULK_METHOD.search(gm) and not FILM_METHOD.search(gm):
        flags.append('sample form: quote reads film but growth_method is bulk')
    return flags

pool['_flags'] = pool.apply(audit, axis=1)
pool['_clean'] = pool['_flags'].apply(len) == 0

rng = np.random.default_rng(42)
idx = rng.choice(len(pool), size=min(50, len(pool)), replace=False)
samp = pool.iloc[idx]

from collections import Counter
cnt_all  = Counter(f.split(':')[0] for fl in pool['_flags'] for f in fl)
cnt_samp = Counter(f.split(':')[0] for fl in samp['_flags'] for f in fl)

res = {
  '_source': str(SRC), '_seed': 42,
  'rows_in_table': int(len(df)),
  'rows_with_value_and_quote': int(len(pool)),
  'full_pool': {'n': int(len(pool)), 'semantically_clean': int(pool['_clean'].sum()),
                'rate': round(float(pool['_clean'].mean()), 3), 'flag_counts': dict(cnt_all)},
  'random_50': {'n': int(len(samp)), 'semantically_clean': int(samp['_clean'].sum()),
                'rate': round(float(samp['_clean'].mean()), 3), 'flag_counts': dict(cnt_samp)},
  'examples': [{'doi': str(r['doi']), 'flags': r['_flags'], 'quote': str(r['quote'])[:220]}
               for _, r in pool[~pool['_clean']].head(8).iterrows()],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != 'examples'}, indent=1))
print('\nfirst flagged examples:')
for e in res['examples'][:4]:
    print(' -', e['flags'], '|', e['quote'][:130].replace('\n',' '))
