from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from image2code.utils.io import read_csv,read_jsonl
from image2code.utils.paths import resolve_data_dir
REQUIRED_KAGGLE_FILES=['manifest.csv','manifest.jsonl','source_licenses_provenance.csv','croissant.json']; REQUIRED_KAGGLE_DIRS=['images']
def locate_data_dir(data_dir:str|Path|None=None)->Path:
    p=resolve_data_dir(data_dir,required=True); assert p is not None; return p
def check_layout(data_dir:str|Path|None=None)->list[str]:
    root=locate_data_dir(data_dir); missing=[]
    for n in REQUIRED_KAGGLE_FILES:
        if not (root/n).is_file(): missing.append(n)
    for n in REQUIRED_KAGGLE_DIRS:
        if not (root/n).is_dir(): missing.append(n+'/')
    return missing
def load_manifest_csv(data_dir:str|Path|None=None)->list[dict[str,str]]: return read_csv(locate_data_dir(data_dir)/'manifest.csv')
def load_manifest_jsonl(data_dir:str|Path|None=None)->list[dict[str,Any]]: return read_jsonl(locate_data_dir(data_dir)/'manifest.jsonl')
def split_rows(rows:list[dict[str,Any]],split:str)->list[dict[str,Any]]:
    aliases={split,split.replace('-','_')}; return [r for r in rows if str(r.get('split') or r.get('benchmark_split') or '') in aliases]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',type=Path); args=ap.parse_args(); root=locate_data_dir(args.data_dir); missing=check_layout(root)
    if missing: raise SystemExit('Missing Kaggle files/directories: '+', '.join(missing))
    rows=load_manifest_csv(root); counts={}
    for r in rows: counts[str(r.get('split') or '')]=counts.get(str(r.get('split') or ''),0)+1
    print({'data_dir':str(root),'manifest_rows':len(rows),'split_counts':counts})
if __name__=='__main__': main()
