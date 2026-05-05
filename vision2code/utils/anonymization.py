from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
PRIVATE_PATHS=["/usr/"+"xtmp/ap843","/usr/"+"project/xtmp/ap843","/home/"+"users/ap843"]
PRIVATE_WORDS=["WANDB_"+"ENTITY=","wandb"+"."+"ai/","chrome-for-testing"+"/chrome"]
SECRET_PATTERNS=[re.compile(r"sk-[A-Za-z0-9_-]{20,}"),re.compile(r"AIza[0-9A-Za-z_-]{20,}"),re.compile(r"hf_[A-Za-z0-9]{20,}")]
EMAIL_RE=re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TEXT_SUFFIXES={'.py','.md','.txt','.json','.jsonl','.csv','.tsv','.toml','.yml','.yaml','.sh','.cff','.example',''}
def scan_text(text:str)->list[str]:
    f=[]
    for v in PRIVATE_PATHS:
        if v in text: f.append(f'private path: {v}')
    for v in PRIVATE_WORDS:
        if v in text: f.append(f'private token/identifier: {v}')
    for p in SECRET_PATTERNS:
        if p.search(text): f.append(f'secret-like pattern: {p.pattern}')
    for m in EMAIL_RE.findall(text):
        if 'example.com' not in m and 'apache.org' not in m: f.append(f'email: {m}')
    return f
def iter_text_files(root:str|Path)->Iterable[Path]:
    root=Path(root)
    for p in root.rglob('*'):
        rel=p.relative_to(root).as_posix()
        if rel=="vision2code/utils/anonymization.py":
            continue
        if p.is_file() and not any(part in {'.git','__pycache__','.pytest_cache'} for part in p.parts) and (p.suffix.lower() in TEXT_SUFFIXES or p.name=='.env.example'): yield p
def scan_repo(root:str|Path)->dict[str,list[str]]:
    root=Path(root); out={}
    for p in iter_text_files(root):
        findings=scan_text(p.read_text(encoding='utf-8',errors='replace'))
        if findings: out[str(p.relative_to(root))]=findings
    return out
