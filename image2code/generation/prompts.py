SYSTEM_PROMPT=("You are an expert Python plotting assistant. Given an input image, write Python code to recreate it using matplotlib/numpy as needed. Save the image only to the Python variable OUTPUT_PATH. Do not quote OUTPUT_PATH and do not append a file extension to it. Return code only.")
USER_PROMPT="Generate Python code to recreate this image. Return code only."
REFINEMENT_PROMPT_TEMPLATE="""You are improving Python code that recreates a reference image.

Previous Python code:
```python
{previous_code}
```

Rewrite the Python code so the rendered output better matches the reference image. Save the image only to OUTPUT_PATH. Return code only."""
