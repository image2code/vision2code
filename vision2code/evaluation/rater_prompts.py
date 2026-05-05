from __future__ import annotations
from pathlib import Path
from typing import Any,Mapping
from vision2code.evaluation.dataset_rubrics import build_rater_repair_prompt,build_rater_user_content
from vision2code.evaluation.generic_rubric import build_generic_rater_repair_prompt,build_generic_rater_user_content
def dataset_rater_prompt(source_image_path:str|Path,rendered_image_path:str|Path,*,dataset_name:str,metadata:Mapping[str,Any]|None=None): return build_rater_user_content(source_image_path=Path(source_image_path),rendered_image_path=Path(rendered_image_path),metadata=metadata,dataset_name=dataset_name)
def generic_rater_prompt(source_image_path:str|Path,rendered_image_path:str|Path,*,dataset_name:str='',metadata:Mapping[str,Any]|None=None): return build_generic_rater_user_content(source_image_path=Path(source_image_path),rendered_image_path=Path(rendered_image_path),metadata=metadata,dataset_name=dataset_name)
