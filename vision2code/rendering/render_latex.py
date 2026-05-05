from __future__ import annotations
from pathlib import Path
def render_latex(tex_source:str,output_png:str|Path,*,timeout_sec:int=60)->dict[str,object]: return {'render_success':False,'status':'latex renderer requires pdflatex in full rerun','output_path':''}
