# Image2Code Reproducibility Repository

This is an anonymized, reviewer-facing code repository for reproducing the Image2Code benchmark tables, figures, validation analyses, and ablations. Raw benchmark images are distributed separately through Kaggle; this repository contains code, configs, filtered manifest metadata, compact saved outputs, and small fixtures.

## Quickstart

```bash
cd image2code
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Use the Kaggle package either through `IMAGE2CODE_DATA_DIR` or `--data_dir`:

```bash
scripts/download_kaggle_data.sh --data_dir /path/to/kaggle/input/image2code-neurips-2026
export IMAGE2CODE_DATA_DIR=/path/to/kaggle/input/image2code-neurips-2026
python3 scripts/validate_release_data.py --data_dir "$IMAGE2CODE_DATA_DIR"
scripts/run_smoke_test.sh --data_dir "$IMAGE2CODE_DATA_DIR" --num_samples 3
```

Without Kaggle data, `python3 scripts/validate_repo.py` uses `data/fixture_kaggle` for fixture-only validation.

## Reproduce Saved-Output Tables And Figures

```bash
scripts/reproduce_main_tables.sh
scripts/reproduce_error_analysis.sh
scripts/reproduce_human_correlation.sh
scripts/reproduce_ablations.sh
scripts/reproduce_figures.sh
```

Outputs are written to `paper_assets/tables/` and `paper_assets/figures/`.

## Repo Map

- `configs/`: release counts, model placeholders, rubric and ablation configs.
- `image2code/data/`: Kaggle loading, manifest validation, split/source metadata.
- `image2code/rendering/`: Python/Matplotlib, LaTeX, Excalidraw, sandbox, failure taxonomy.
- `image2code/evaluation/`: dataset rubrics, generic rubric, rater prompts, parsing, aggregation, guardrails.
- `image2code/generation/`: prompts, code normalization, API/local model generation wrappers.
- `image2code/metrics/`: embedding similarity baselines and focus texts.
- `image2code/ablations/`: self-training filters, test-time scaling, tool-use summaries.
- `image2code/figures/`: paper table and figure builders.
- `results/paper_outputs/`: compact saved summaries used to regenerate paper tables.
- `data/fixture_kaggle/`: three-row synthetic Kaggle-layout fixture for CI and smoke checks.

Full model reruns require provider API keys or local checkpoints supplied by the user in `.env`; no keys or private checkpoint paths are included.
