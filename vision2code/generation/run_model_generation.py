from __future__ import annotations
import argparse,os
from pathlib import Path
from vision2code.data.load_kaggle_dataset import load_manifest_csv
from vision2code.generation.prompts import SYSTEM_PROMPT,USER_PROMPT
PROVIDER_ENV={'openai':'OPENAI_API_KEY','gemini':'GOOGLE_API_KEY','together':'TOGETHER_API_KEY','local':'HF_TOKEN'}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--provider',choices=sorted(PROVIDER_ENV),required=True); ap.add_argument('--model',required=True); ap.add_argument('--data_dir',type=Path,required=True); ap.add_argument('--output_dir',type=Path,required=True); ap.add_argument('--split',default='test_mini'); ap.add_argument('--num_samples',type=int,default=0); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args(); rows=[r for r in load_manifest_csv(args.data_dir) if str(r.get('split'))==args.split]
    if args.num_samples: rows=rows[:args.num_samples]
    if args.dry_run: print({'provider':args.provider,'model':args.model,'rows':len(rows),'system_prompt':SYSTEM_PROMPT,'user_prompt':USER_PROMPT}); return
    env=PROVIDER_ENV[args.provider]
    if args.provider!='local' and not os.getenv(env): raise RuntimeError(f'Set {env} before running generation.')
    raise SystemExit('Generation is key-gated; provide API/checkpoint credentials for a full rerun.')
if __name__=='__main__': main()
