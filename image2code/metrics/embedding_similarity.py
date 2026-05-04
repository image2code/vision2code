from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from image2code.metrics.focus_texts import DATASET_EMBEDDING_FOCUS_TEXTS,focus_text_for_dataset
def scale_cosine_to_0_to_5(value:float|None)->float|None:
    if value is None: return None
    return max(0.0,min(5.0,((float(value)+1.0)/2.0)*5.0))
def l2_normalize(vector:Sequence[float])->np.ndarray:
    arr=np.asarray(vector,dtype=np.float64); norm=float(np.linalg.norm(arr))
    if norm<=0.0 or not math.isfinite(norm): raise ValueError('Cannot normalize a zero or non-finite vector')
    return arr/norm
def cosine_similarity(left:Sequence[float],right:Sequence[float])->float: return float(np.dot(l2_normalize(left),l2_normalize(right)))
