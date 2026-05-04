#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from image2code.data.validate_manifest import validate_kaggle_dir
from image2code.utils.anonymization import scan_repo
REQ_FILES=['README.md','LICENSE','CITATION.cff','.gitignore','.env.example','pyproject.toml','environment.yml','configs/data/test-mini-filtered.json','configs/data/test-filtered.json','data/example_manifest_small.csv','results/paper_outputs/main_leaderboard/paper_table_summary.csv','docs/REPRODUCIBILITY.md','docs/DATASET.md','docs/HUMAN_VALIDATION.md','docs/COMPUTE.md','docs/LICENSES_AND_PROVENANCE.md']
REQ_MODS=['image2code.data.load_kaggle_dataset','image2code.data.validate_manifest','image2code.rendering.render_python','image2code.rendering.failure_taxonomy','image2code.evaluation.dataset_rubrics','image2code.evaluation.generic_rubric','image2code.metrics.embedding_similarity','image2code.generation.prompts','image2code.ablations.self_training.filters','image2code.figures.make_leaderboard_tables']
BAD_NAMES={'.env','.DS_Store'}; SKIP_PARTS={'.git','__pycache__','.pytest_cache'}; BAD_PARTS={'node_modules','wandb','logs','checkpoints'}; BAD_SUFFIX={'.safetensors','.ckpt','.pth','.pt','.arrow','.pyc'}
def validate_tree(root):
    e=[]
    for rel in REQ_FILES:
        if not (root/rel).exists(): e.append(f'missing required file: {rel}')
    for p in root.rglob('*'):
        if any(part in SKIP_PARTS for part in p.relative_to(root).parts):
            continue
        rel=p.relative_to(root).as_posix()
        if p.name in BAD_NAMES: e.append(f'forbidden file: {rel}')
        if any(part in BAD_PARTS for part in p.relative_to(root).parts): e.append(f'forbidden path component: {rel}')
        if p.suffix in BAD_SUFFIX: e.append(f'forbidden large/checkpoint artifact: {rel}')
    return e
def validate_imports():
    e=[]
    for m in REQ_MODS:
        try: importlib.import_module(m)
        except Exception as ex: e.append(f'import failed: {m}: {ex}')
    return e
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',type=Path); args=ap.parse_args(); root=Path(__file__).resolve().parents[1]; e=[]
    e+=validate_tree(root); e+=validate_imports()
    for rel,findings in scan_repo(root).items(): e.append(f'anonymization finding in {rel}: {findings}')
    e+=validate_kaggle_dir(args.data_dir or root/'data'/'fixture_kaggle',allow_small=args.data_dir is None)
    if e:
        print('Repository validation failed:'); [print('- '+x) for x in e]; raise SystemExit(1)
    print('Repository validation passed.')
if __name__=='__main__': main()
