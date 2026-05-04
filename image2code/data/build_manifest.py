from __future__ import annotations
import argparse,json
from collections import Counter
from typing import Any
from image2code.data.source_metadata import EXPECTED_SOURCE_COUNTS_TEST,EXPECTED_SOURCE_COUNTS_TEST_MINI
from image2code.utils.paths import repo_root
def load_filtered_manifest(split:str)->list[dict[str,Any]]:
    name={'test-mini':'test-mini-filtered.json','test':'test-filtered.json'}[split]; return json.loads((repo_root()/'configs'/'data'/name).read_text(encoding='utf-8'))
def summarize_manifest(split:str)->dict[str,Any]:
    rows=load_filtered_manifest(split); return {'split':split,'rows':len(rows),'source_counts':dict(sorted(Counter(str(r.get('dataset') or r.get('source_dataset') or '') for r in rows).items()))}
def validate_filtered_manifests()->list[str]:
    errors=[]
    for split,counts in {'test-mini':EXPECTED_SOURCE_COUNTS_TEST_MINI,'test':EXPECTED_SOURCE_COUNTS_TEST}.items():
        if summarize_manifest(split)['source_counts']!=dict(sorted(counts.items())): errors.append(f'{split} source counts changed')
    mini={(r.get('dataset'),r.get('question_folder'),r.get('sample_id')) for r in load_filtered_manifest('test-mini')}; test={(r.get('dataset'),r.get('question_folder'),r.get('sample_id')) for r in load_filtered_manifest('test')}
    if not mini.issubset(test): errors.append('test-mini is not a subset of test')
    return errors
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['test-mini','test'],default='test-mini'); ap.add_argument('--validate',action='store_true'); args=ap.parse_args()
    if args.validate:
        errors=validate_filtered_manifests()
        if errors: raise SystemExit('\n'.join(errors))
    print(json.dumps(summarize_manifest(args.split),indent=2))
if __name__=='__main__': main()
