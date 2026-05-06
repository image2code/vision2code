# Tool-Use Ablations

`run_tool_ablation.py` runs OpenAI generation for executable targets other than Python:

- `latex_docvqa`: standalone LaTeX document source
- `excalidraw_json`: Excalidraw scene JSON

The prompts match the ablation setup and keep outputs self-contained: no external images, URLs, shell escape, HTML, SVG, Mermaid, or network resources.

Example:

```bash
python3 -m vision2code.ablations.tool_use.run_tool_ablation \
  --task latex_docvqa \
  --manifest path/to/docvqa_latex_manifest.jsonl \
  --output-dir results/ablations/tool_use/latex_docvqa \
  --model gpt-5.4-mini \
  --num-samples 1 \
  --env-file .env
```

Render generated artifacts:

```bash
python3 -m vision2code.ablations.tool_use.render_outputs \
  --task latex_docvqa \
  --inference-csv results/ablations/tool_use/latex_docvqa/inference_latex_docvqa.csv \
  --output-dir results/ablations/tool_use/latex_docvqa
```

Evaluate the renders with the default local rater:

```bash
python3 -m vision2code.ablations.tool_use.evaluate \
  --task latex_docvqa \
  --render-csv results/ablations/tool_use/latex_docvqa/render_latex.csv \
  --provider local_vllm \
  --rater-model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --rater-base-url http://127.0.0.1:8000/v1 \
  --rater-api-key EMPTY \
  --env-file .env
```

LaTeX rendering needs `pdflatex` and `pdftoppm`; `latexmk` is used when available. Excalidraw rendering uses the checked-in browser bundle under `vendor/excalidraw_renderer/` and requires Chrome or Chromium.

Saved paper summaries are under `results/paper_outputs/ablations/tool_use/` and `results/paper_outputs/ablations/tool_use_ablation_summary.csv`.
