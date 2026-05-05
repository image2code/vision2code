from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_file(path:str|Path)->str:
    d=hashlib.sha256()
    with Path(path).open('rb') as h:
        for chunk in iter(lambda:h.read(1024*1024),b''): d.update(chunk)
    return d.hexdigest()
def sha256_text(text:str)->str: return hashlib.sha256(text.encode('utf-8')).hexdigest()
