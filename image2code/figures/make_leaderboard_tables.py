from __future__ import annotations
import argparse
from pathlib import Path
from image2code.utils.io import read_csv,write_csv
from image2code.utils.paths import repo_root
def reproduce(output_dir:Path|None=None)->list[Path]:
    root=repo_root(); src=root/'results'/'paper_outputs'/'main_leaderboard'; out=output_dir or root/'paper_assets'/'tables'; out.mkdir(parents=True,exist_ok=True); outputs=[]
    mapping={'paper_table_summary.csv':'paper_table_summary.csv','main_benchmark_ratings.csv':'main_leaderboard.csv','main_benchmark_ratings_by_dataset.csv':'main_leaderboard_by_dataset.csv','generic_rubric_ratings.csv':'generic_rubric_table.csv','cosine_similarity_scores.csv':'cosine_similarity_table.csv','render_success_rates.csv':'render_success_table.csv'}
    for s,t in mapping.items(): target=out/t; write_csv(target,read_csv(src/s)); outputs.append(target)
    return outputs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path); args=ap.parse_args(); [print(p) for p in reproduce(args.output_dir)]
if __name__=='__main__': main()
