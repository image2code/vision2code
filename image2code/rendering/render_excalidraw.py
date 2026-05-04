from __future__ import annotations
from pathlib import Path
def render_excalidraw(scene_json:str|Path,output_png:str|Path,*,renderer_dir:str|Path|None=None,timeout_sec:int=60)->dict[str,object]: return {'render_success':False,'status':'excalidraw renderer requires node/vendor install in full rerun','output_path':''}
