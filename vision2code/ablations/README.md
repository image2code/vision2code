# Ablations

This directory contains runnable ablation entrypoints plus the saved CSV summaries used by the paper-table reproduction scripts.

The scripts are intentionally path-parameterized. They do not include private checkpoint paths, scheduler launchers, API keys, W&B settings, or raw training outputs. Expensive reruns require the user to provide models, API keys, GPU resources, and input directories.

## Entry Points

```bash
# Self-training dataset construction from paired R1/R2 ratings.
bash scripts/prepare_self_training_data.sh --help

# Optional Qwen3-VL full fine-tuning on those datasets.
bash scripts/train_self_training_model.sh --help

# Test-time scaling generation and rendering.
bash scripts/run_test_time_scaling.sh --help

# Cosine/image-embedding baselines.
bash scripts/run_cosine_baselines.sh --help

# LaTeX and Excalidraw tool-use generation.
bash scripts/run_tool_use_ablation.sh --help

# Render and rate generated LaTeX/Excalidraw artifacts.
bash scripts/render_tool_use_ablation.sh --help
bash scripts/evaluate_tool_use_ablation.sh --help
```

Saved paper summaries are under `results/paper_outputs/ablations/` and are copied to `paper_assets/tables/` by:

```bash
bash scripts/reproduce_ablations.sh
```
