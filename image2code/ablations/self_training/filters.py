from __future__ import annotations
from typing import Any,Callable,Mapping
def _score(row:Mapping[str,Any],key:str)->float|None:
    try:
        v=row.get(key); return float(v) if v is not None and str(v).strip() else None
    except Exception: return None
def is_valid_pair(row:Mapping[str,Any])->bool: return _score(row,'r1_score') is not None and _score(row,'r2_score') is not None
def filter_predicates(alpha:float=4.0)->dict[str,Callable[[Mapping[str,Any]],bool]]:
    return {'all_valid':lambda r:is_valid_pair(r),'r1_ge_alpha_r2_ge_r1':lambda r:is_valid_pair(r) and _score(r,'r1_score')>=alpha and _score(r,'r2_score')>=_score(r,'r1_score'),'r1_ge_alpha_r2_lt_r1':lambda r:is_valid_pair(r) and _score(r,'r1_score')>=alpha and _score(r,'r2_score')<_score(r,'r1_score'),'r1_lt_alpha':lambda r:is_valid_pair(r) and _score(r,'r1_score')<alpha,'r1_ge_alpha':lambda r:is_valid_pair(r) and _score(r,'r1_score')>=alpha}
