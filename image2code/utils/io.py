from __future__ import annotations
import csv,json
from pathlib import Path
from typing import Any,Iterable,Mapping,Sequence

def read_json(path:str|Path)->Any: return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path:str|Path,payload:Any)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def read_jsonl(path:str|Path)->list[dict[str,Any]]:
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        obj=json.loads(line)
        if not isinstance(obj,dict): raise ValueError(f"JSONL row {i} is not an object: {path}")
        rows.append(obj)
    return rows
def write_jsonl(path:str|Path,rows:Iterable[Mapping[str,Any]])->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as h:
        for row in rows: h.write(json.dumps(dict(row),ensure_ascii=False)+"\n")
def read_csv(path:str|Path)->list[dict[str,str]]:
    with Path(path).open("r",encoding="utf-8-sig",newline="",errors="replace") as h: return [dict(r) for r in csv.DictReader(h)]
def write_csv(path:str|Path,rows:Sequence[Mapping[str,Any]],fieldnames:Sequence[str]|None=None)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if fieldnames is None:
        keys=[]
        for row in rows:
            for k in row:
                if k not in keys: keys.append(str(k))
        fieldnames=keys
    with path.open("w",encoding="utf-8",newline="") as h:
        wr=csv.DictWriter(h,fieldnames=list(fieldnames),extrasaction="ignore"); wr.writeheader()
        for row in rows: wr.writerow({f:row.get(f,"") for f in fieldnames})
