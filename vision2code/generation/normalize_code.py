from __future__ import annotations
import re
def normalize_generated_code(code:str)->str:
    code=code.strip(); code=re.sub(r"<think>.*?</think>\s*","",code,flags=re.DOTALL)
    if '</think>' in code: code=code.rsplit('</think>',1)[-1].lstrip()
    code=code.replace('<think>','').replace('</think>','')
    m=re.findall(r"```(?:python)?\s*(.*?)```",code,flags=re.DOTALL|re.IGNORECASE)
    if m: return m[-1].strip()
    return code
