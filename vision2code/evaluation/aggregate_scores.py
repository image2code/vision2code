from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any,Iterable,Mapping
from vision2code.utils.io import read_csv,write_csv
def parse_float(value:Any)->float|None:
    try:
        t=str(value).strip(); return float(t) if t else None
    except Exception: return None
def mean(values:Iterable[float|None])->float|None:
    clean=[v for v in values if v is not None]; return sum(clean)/len(clean) if clean else None
def macro_mean_by_dataset(rows:Iterable[Mapping[str,Any]],score_field:str='rating_final_0_to_5')->float|None:
    g=defaultdict(list)
    for r in rows:
        s=parse_float(r.get(score_field))
        if s is not None: g[str(r.get('dataset') or '')].append(s)
    ms=[sum(v)/len(v) for v in g.values() if v]; return sum(ms)/len(ms) if ms else None
def summarize_rating_csv(path:str|Path,out_path:str|Path|None=None):
    rows=read_csv(path); g=defaultdict(list)
    for r in rows: g[(str(r.get('model_slug') or ''),str(r.get('benchmark_split') or r.get('split') or ''))].append(r)
    out=[]
    for (m,sp),grp in sorted(g.items()): out.append({'model_slug':m,'split':sp,'rows':len(grp),'mean_rating_0_to_5':mean(parse_float(r.get('rating_final_0_to_5')) for r in grp),'macro_mean_by_dataset_0_to_5':macro_mean_by_dataset(grp)})
    if out_path: write_csv(out_path,out)
    return out
