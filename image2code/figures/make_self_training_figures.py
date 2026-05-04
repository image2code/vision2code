from __future__ import annotations
import argparse
from pathlib import Path
from image2code.utils.io import read_csv,write_csv
from image2code.utils.paths import repo_root
def reproduce(output_dir:Path|None=None)->list[Path]:
    root=repo_root(); src=root/'results'/'paper_outputs'/'ablations'; out=output_dir or root/'paper_assets'/'tables'; out.mkdir(parents=True,exist_ok=True); outputs=[]
    for n in ['ssl_scores.csv','test_time_scaling_scores.csv','self_training_summary.csv','tool_use_ablation_summary.csv']:
        p=src/n
        if p.exists(): target=out/n; write_csv(target,read_csv(p)); outputs.append(target)
    return outputs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path); args=ap.parse_args(); [print(p) for p in reproduce(args.output_dir)]
if __name__=='__main__': main()
