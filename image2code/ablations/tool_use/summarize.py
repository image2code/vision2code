from __future__ import annotations
from pathlib import Path
from image2code.utils.io import read_csv,write_csv
def summarize_tool_use(eval_csv:Path,output_csv:Path):
    rows=read_csv(eval_csv); out=[{'input_csv':eval_csv.as_posix(),'rows':len(rows)}]; write_csv(output_csv,out); return out
