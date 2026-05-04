# Compute

## CPU-Only

Saved-output reproduction is CPU-only:

```bash
scripts/reproduce_main_tables.sh
scripts/reproduce_error_analysis.sh
scripts/reproduce_human_correlation.sh
scripts/reproduce_ablations.sh
scripts/reproduce_figures.sh
```

The smoke test needs Python, Pillow, NumPy, pandas, and Matplotlib. Rendering runs with `MPLBACKEND=Agg` in a temporary directory.

## API Reruns

Full proprietary-model reruns require user-provided keys in `.env`:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `TOGETHER_API_KEY`

The repository includes wrappers and prompt templates, not keys.

## Local Model And Training Reruns

Local Hugging Face/Qwen reruns require `HF_TOKEN` if the selected model requires access, plus explicit model/checkpoint paths supplied by the user. Training and large ablations require GPUs sized for the selected checkpoint. No private checkpoint, cache, or cluster launch path is included.

## Tool-Use Ablations

LaTeX rendering requires a local TeX installation. Excalidraw rendering requires Node/npm dependencies installed by the reviewer; `node_modules/` is intentionally excluded.
