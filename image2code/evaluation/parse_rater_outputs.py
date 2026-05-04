from __future__ import annotations
import argparse
from pathlib import Path
from image2code.evaluation.dataset_rubrics import parse_json_object
def parse_rater_text(text:str)->dict: return parse_json_object(text)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path',type=Path); args=ap.parse_args(); print(parse_rater_text(args.path.read_text(encoding='utf-8')))
if __name__=='__main__': main()
