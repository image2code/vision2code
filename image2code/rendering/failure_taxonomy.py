from __future__ import annotations
import re
PLOT_LABELS={'hallucinated_api_argument':'Hallucinated API/argument','shape_3d_geometry':'Shape/3D/geometry','other_runtime':'Other runtime','syntax_truncation':'Syntax/truncation','missing_dependency_file':'Missing dependency/file','timeout':'Timeout'}
def classify_failure(status:str)->str:
    text=(status or '').strip(); lower=text.lower()
    if 'timeout' in lower or 'timed out' in lower: return 'timeout'
    if lower=='no_output_image' or 'no_output_image' in lower or 'no output' in lower: return 'other_runtime'
    ex=re.findall(r'([A-Za-z_]*Error):',text); exception=ex[-1] if ex else ''
    if exception=='SyntaxError': return 'syntax_truncation'
    if exception in {'ModuleNotFoundError','ImportError','FileNotFoundError'}: return 'missing_dependency_file'
    if exception in {'ValueError','IndexError','KeyError','RuntimeError','AxisError','_UFuncNoLoopError'}: return 'shape_3d_geometry'
    if exception in {'AttributeError','NameError'}: return 'hallucinated_api_argument'
    if exception=='TypeError' and ('got multiple values' in lower or 'required positional argument' in lower or 'positional arguments but' in lower or 'got both' in lower or 'col must' in lower or 'must be an instance' in lower or 'add_line()' in lower or 'pathpatch_2d_to_3d()' in lower): return 'hallucinated_api_argument'
    return 'other_runtime'
