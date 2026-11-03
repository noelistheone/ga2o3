"""Bind displayed result values to immutable inputs and exact report spans."""
from pathlib import Path
import json,re
import numpy as np
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

ROOT=e.ROOT


def sanitize_units(text):
    replacements={'cm⁻³':r'\(\mathrm{cm}^{-3}\)','s⁻¹':r'\(\mathrm{s}^{-1}\)',
        'cm⁻¹':r'\(\mathrm{cm}^{-1}\)'}
    counts={}
    for source,target in replacements.items():
        counts[source]=text.count(source);text=text.replace(source,target)
    return text,counts


class Evidence:
    def __init__(self):
        self.claims=[];self.frames={};self.documents={}

    def frame(self,path):
        path=Path(path)
        if path not in self.frames:self.frames[path]=read(path)
        return self.frames[path]

    def document(self,path):
        path=Path(path)
        if path not in self.documents:self.documents[path]=json.loads(path.read_text())
        return self.documents[path]

    def csv_value(self,path,selector,column):
        part=self.frame(path)
        for key,value in selector.items():part=part[part[key].eq(value)]
        assert len(part)==1,(str(path),selector,column)
        v=part[column].iloc[0]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):v=None
        source=dict(kind='csv',path=str(Path(path).relative_to(ROOT)),sha256=e.sha(path),
            selector=e.clean(selector),column=column)
        return e.clean(v),source

    def json_value(self,path,keys):
        value=self.document(path)
        for key in keys:value=value[key]
        return e.clean(value),dict(kind='json',path=str(Path(path).relative_to(ROOT)),sha256=e.sha(path),keys=keys)

    def add(self,value,source,fmt=None):
        if value is None:shown='未覆盖'
        elif isinstance(value,bool):shown='是' if value else '否'
        elif fmt is not None:shown=format(value,fmt)
        else:shown=str(value)
        index=len(self.claims);token='@@EV'+str(index).zfill(5)+'@@'
        self.claims.append(dict(id=index,value=value,rendered=shown,source=source))
        return token

    def csv(self,path,selector,column,fmt='.6f'):
        value,source=self.csv_value(path,selector,column)
        return self.add(value,source,fmt)

    def fact(self,path,keys,fmt=None):
        value,source=self.json_value(path,keys)
        return self.add(value,source,fmt)

    def difference(self,path,a,b,column):
        va,sa=self.csv_value(path,a,column);vb,sb=self.csv_value(path,b,column)
        value=None if va is None or vb is None else float(va-vb)
        return self.add(value,dict(kind='difference',a=sa,b=sb),'+.6f')

    def render(self,text):
        spans=[];parts=[];end=0;length=0
        for match in re.finditer(r'@@EV(\d{5})@@',text):
            index=int(match[1]);claim=self.claims[index]
            chunk=text[end:match.start()];parts.append(chunk);length+=len(chunk)
            start=length;parts.append(claim['rendered']);length+=len(claim['rendered'])
            spans.append(dict(claim_id=index,start=start,end=length))
            end=match.end()
        parts.append(text[end:]);rendered=''.join(parts)
        assert len(spans)==len(self.claims) and {s['claim_id'] for s in spans}==set(range(len(self.claims)))
        assert '@@EV' not in rendered
        return rendered,spans


def source_value(source):
    kind=source['kind']
    if kind=='difference':
        a=source_value(source['a']);b=source_value(source['b'])
        return None if a is None or b is None else float(a-b)
    path=ROOT/source['path'];assert e.sha(path)==source['sha256'],str(path)
    if kind=='json':
        value=json.loads(path.read_text())
        for key in source['keys']:value=value[key]
        return e.clean(value)
    assert kind=='csv'
    table=read(path)
    for key,value in source['selector'].items():table=table[table[key].eq(value)]
    assert len(table)==1
    value=table[source['column']].iloc[0]
    return None if isinstance(value,(float,np.floating)) and not np.isfinite(value) else e.clean(value)


def audit(report,claims):
    text=Path(report).read_text();rows=[]
    for span in claims['spans']:
        c=claims['claims'][span['claim_id']];actual=source_value(c['source'])
        value_ok=actual==c['value']
        span_ok=text[span['start']:span['end']]==c['rendered']
        rows.append(dict(id=c['id'],source_value_matches=bool(value_ok),
            displayed_span_matches=bool(span_ok),passed=bool(value_ok and span_ok)))
    controls=not any(ord(c)<32 and c!='\n' for c in text)
    inline=text.count(r'\(')==text.count(r'\)')
    display=text.count(r'\[')==text.count(r'\]')
    return dict(passed=all(r['passed'] for r in rows) and controls and inline and display,claims_checked=len(rows),rows=rows,
        report_sha256=e.sha(report),math_control_characters_absent=not any(ord(c)<32 and c!='\n' for c in text),
        inline_math_balanced=inline,display_math_balanced=display,
        no_new_training_or_selection=True,goal_complete=False)
