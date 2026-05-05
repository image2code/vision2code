from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from typing import Any
from vision2code.data.load_kaggle_dataset import check_layout,load_manifest_csv,load_manifest_jsonl
from vision2code.data.source_metadata import EXPECTED_SOURCE_COUNTS_TEST,EXPECTED_SOURCE_COUNTS_TEST_MINI,EXPECTED_SPLIT_COUNTS
from vision2code.utils.hashing import sha256_file
PRIVATE_MARKERS=["/usr/"+"xtmp/","/usr/"+"project/","/home/"+"users/"]
def _split_name(raw:str)->str: return raw.replace('_','-')
def validate_rows(rows:list[dict[str,Any]],data_dir:str|Path|None=None,*,allow_small:bool=False)->list[str]:
    errors=[]; root=Path(data_dir).resolve() if data_dir else None
    if not rows: return ['manifest has no rows']
    for i,row in enumerate(rows,1):
        txt=json.dumps(row,ensure_ascii=False); image_path=str(row.get('image_path') or '')
        for m in PRIVATE_MARKERS:
            if m in txt: errors.append(f'row {i}: private/local path marker found')
        if image_path:
            p=Path(image_path)
            if p.is_absolute(): errors.append(f'row {i}: image_path is absolute')
            if '..' in p.parts: errors.append(f'row {i}: image_path escapes package')
            if root is not None:
                full=root/image_path
                if not full.exists(): errors.append(f'row {i}: missing image {image_path}')
                h=str(row.get('image_sha256') or '').strip()
                if h and full.exists() and sha256_file(full)!=h: errors.append(f'row {i}: sha256 mismatch for {image_path}')
    counts=Counter(_split_name(str(r.get('split') or r.get('benchmark_split') or '')) for r in rows)
    if not allow_small:
        if counts.get('test')!=EXPECTED_SPLIT_COUNTS['test']: errors.append(f"expected {EXPECTED_SPLIT_COUNTS['test']} test rows, found {counts.get('test',0)}")
        if counts.get('test-mini')!=EXPECTED_SPLIT_COUNTS['test-mini']: errors.append(f"expected {EXPECTED_SPLIT_COUNTS['test-mini']} test-mini rows, found {counts.get('test-mini',0)}")
        if len(rows)!=EXPECTED_SPLIT_COUNTS['kaggle_manifest_rows']: errors.append(f"expected {EXPECTED_SPLIT_COUNTS['kaggle_manifest_rows']} manifest rows, found {len(rows)}")
    return errors
def validate_source_counts(rows:list[dict[str,Any]],*,allow_small:bool=False)->list[str]:
    if allow_small: return []
    errors=[]; by={'test':Counter(),'test-mini':Counter()}
    for r in rows:
        s=_split_name(str(r.get('split') or r.get('benchmark_split') or ''))
        if s in by: by[s][str(r.get('source_dataset') or r.get('dataset') or '')]+=1
    for split,exp in {'test':EXPECTED_SOURCE_COUNTS_TEST,'test-mini':EXPECTED_SOURCE_COUNTS_TEST_MINI}.items():
        got=dict(sorted(by[split].items()))
        if got!=dict(sorted(exp.items())): errors.append(f'source counts changed for {split}: {got}')
    return errors
def validate_kaggle_dir(data_dir:str|Path,*,allow_small:bool=False)->list[str]:
    errors=[f'missing layout item: {m}' for m in check_layout(data_dir)]
    if errors: return errors
    csv_rows=load_manifest_csv(data_dir); jsonl_rows=load_manifest_jsonl(data_dir)
    if len(csv_rows)!=len(jsonl_rows): errors.append(f'manifest.csv rows ({len(csv_rows)}) != manifest.jsonl rows ({len(jsonl_rows)})')
    errors.extend(validate_rows(csv_rows,data_dir,allow_small=allow_small)); errors.extend(validate_source_counts(csv_rows,allow_small=allow_small)); return errors
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',type=Path,required=True); ap.add_argument('--allow-small',action='store_true'); args=ap.parse_args(); errors=validate_kaggle_dir(args.data_dir,allow_small=args.allow_small)
    if errors:
        print('Manifest validation failed:'); [print('- '+e) for e in errors]; raise SystemExit(1)
    print('Manifest validation passed.')
if __name__=='__main__': main()
