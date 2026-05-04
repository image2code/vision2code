from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from image2code.data.source_metadata import CATEGORY_BY_DATASET
from image2code.utils.io import write_csv
from image2code.utils.paths import repo_root
def reproduce(output_dir:Path|None=None)->list[Path]:
    root=repo_root(); out=output_dir or root/'paper_assets'/'tables'; out.mkdir(parents=True,exist_ok=True); outputs=[]
    for split,fn in [('test-mini','test-mini-filtered.json'),('test','test-filtered.json')]:
        rows=json.loads((root/'configs'/'data'/fn).read_text(encoding='utf-8')); by=Counter(str(r.get('dataset') or '') for r in rows); dom=Counter()
        for ds,c in by.items(): dom[CATEGORY_BY_DATASET.get(ds,'Other')]+=c
        t1=out/f'benchmark_stats_by_dataset_{split}.csv'; t2=out/f'benchmark_stats_by_domain_{split}.csv'; write_csv(t1,[{'split':split,'dataset':ds,'count':c,'domain':CATEGORY_BY_DATASET.get(ds,'Other')} for ds,c in sorted(by.items())]); write_csv(t2,[{'split':split,'domain':d,'count':c} for d,c in sorted(dom.items())]); outputs += [t1,t2]
    return outputs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path); args=ap.parse_args(); [print(p) for p in reproduce(args.output_dir)]
if __name__=='__main__': main()
