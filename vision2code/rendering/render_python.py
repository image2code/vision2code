from __future__ import annotations
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Tuple

def render_matplotlib_code(code:str, output_path:str|Path, *, timeout_sec:int=30, target_size:Tuple[int,int]=(512,512), target_dpi:int=100)->dict[str,object]:
    requested_output_path=Path(output_path).expanduser()
    output_path=requested_output_path.resolve(); output_path.parent.mkdir(parents=True,exist_ok=True)
    if output_path.exists(): output_path.unlink()
    width,height=target_size
    wrapper = f"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

_FORCED_OUTPUT_PATH = {str(output_path)!r}
OUTPUT_PATH = _FORCED_OUTPUT_PATH
_TARGET_WIDTH = {int(width)}
_TARGET_HEIGHT = {int(height)}
_TARGET_DPI = {int(target_dpi)}
_ORIG_PYPLOT_SAVEFIG = plt.savefig
_ORIG_FIGURE_SAVEFIG = Figure.savefig

def _prepare_figure(fig):
    fig.set_size_inches(_TARGET_WIDTH / _TARGET_DPI, _TARGET_HEIGHT / _TARGET_DPI, forward=True)

def _forced_savefig_kwargs(kwargs):
    safe_kwargs = dict(kwargs)
    safe_kwargs.pop('fname', None)
    safe_kwargs.pop('filename', None)
    safe_kwargs.pop('filename_or_obj', None)
    safe_kwargs['dpi'] = _TARGET_DPI
    safe_kwargs['bbox_inches'] = None
    safe_kwargs['pad_inches'] = 0
    return safe_kwargs

def _forced_pyplot_savefig(*args, **kwargs):
    _prepare_figure(plt.gcf())
    tail_args = tuple(args[1:]) if args else ()
    return _ORIG_PYPLOT_SAVEFIG(_FORCED_OUTPUT_PATH, *tail_args, **_forced_savefig_kwargs(kwargs))

def _forced_figure_savefig(self, *args, **kwargs):
    _prepare_figure(self)
    tail_args = tuple(args[1:]) if args else ()
    return _ORIG_FIGURE_SAVEFIG(self, _FORCED_OUTPUT_PATH, *tail_args, **_forced_savefig_kwargs(kwargs))

plt.savefig = _forced_pyplot_savefig
Figure.savefig = _forced_figure_savefig

{code}
"""
    env=dict(os.environ)
    env['MPLBACKEND']='Agg'
    mpl_config_dir=env.get('MPLCONFIGDIR') or os.path.join(env.get('XDG_CACHE_HOME', tempfile.gettempdir()), 'matplotlib')
    Path(mpl_config_dir).mkdir(parents=True, exist_ok=True)
    env['MPLCONFIGDIR']=mpl_config_dir
    env.setdefault('PYTHONNOUSERSITE','1')
    with tempfile.TemporaryDirectory(prefix='vision2code_exec_') as td:
        sp=Path(td)/'run.py'; sp.write_text(wrapper,encoding='utf-8')
        try: proc=subprocess.run([sys.executable,str(sp)],cwd=str(output_path.parent),env=env,text=True,capture_output=True,timeout=timeout_sec,check=False)
        except subprocess.TimeoutExpired: return {'render_success':False,'status':'timeout','output_path':''}
    if proc.returncode!=0: return {'render_success':False,'status':(proc.stderr or proc.stdout or 'execution_failed')[:4000],'output_path':''}
    if not output_path.exists() or output_path.stat().st_size==0: return {'render_success':False,'status':'no_output_image','output_path':''}
    return {'render_success':True,'status':'ok','output_path':str(requested_output_path)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('code_path',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--timeout-sec',type=int,default=30); args=ap.parse_args(); print(render_matplotlib_code(args.code_path.read_text(encoding='utf-8'),args.output,timeout_sec=args.timeout_sec))
if __name__=='__main__': main()
