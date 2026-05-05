from __future__ import annotations
import os,tempfile
from pathlib import Path

def repo_root()->Path: return Path(__file__).resolve().parents[2]
def resolve_data_dir(data_dir:str|Path|None=None,*,required:bool=False)->Path|None:
    raw=str(data_dir or os.getenv('VISION2CODE_DATA_DIR','')).strip()
    if not raw:
        if required: raise RuntimeError('Set VISION2CODE_DATA_DIR or pass --data_dir.')
        return None
    p=Path(raw).expanduser().resolve()
    if required and not p.exists(): raise FileNotFoundError(f'Data directory does not exist: {p}')
    return p
def default_results_dir()->Path: return repo_root()/'results'/'paper_outputs'
def default_paper_assets_dir()->Path: return repo_root()/'paper_assets'
def temp_render_dir(prefix:str='vision2code_render_'): return tempfile.TemporaryDirectory(prefix=prefix)
def repo_relative(path:str|Path)->str:
    p=Path(path).resolve()
    try: return p.relative_to(repo_root()).as_posix()
    except ValueError: return p.as_posix()
