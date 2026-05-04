from __future__ import annotations
from typing import Any,Mapping
from image2code.evaluation.dataset_rubrics import apply_rating_caps
def apply_caps(provisional_rating_0_to_5:float,rubric:Mapping[str,Any],category_results:Mapping[str,Any],*,execution_status:str='ok',candidate_inspection:Mapping[str,Any]|None=None)->dict[str,Any]:
    return apply_rating_caps(provisional_rating_0_to_5=provisional_rating_0_to_5,rubric=rubric,category_results=category_results,execution_status=execution_status,candidate_inspection=candidate_inspection)
