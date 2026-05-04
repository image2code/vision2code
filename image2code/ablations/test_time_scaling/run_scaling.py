from __future__ import annotations
from image2code.generation.prompts import REFINEMENT_PROMPT_TEMPLATE
def build_refinement_prompt(previous_code:str)->str: return REFINEMENT_PROMPT_TEMPLATE.format(previous_code=previous_code)
