# Vision2Code

Code, configs, saved outputs, and scripts for the Vision2Code benchmark. Raw benchmark images are distributed separately through Kaggle.

## Install

```bash
cd vision2code
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Data

```bash
scripts/download_kaggle_data.sh --data_dir /path/to/kaggle/input/vision2code
export VISION2CODE_DATA_DIR=/path/to/kaggle/input/vision2code
python3 scripts/validate_release_data.py --data_dir "$VISION2CODE_DATA_DIR"
scripts/run_smoke_test.sh --data_dir "$VISION2CODE_DATA_DIR" --num_samples 3
```

For a fixture-only check without Kaggle data:

```bash
scripts/run_smoke_test.sh --data_dir data/fixture_kaggle --num_samples 3
```

## Tables And Figures

```bash
scripts/reproduce_main_tables.sh
scripts/reproduce_error_analysis.sh
scripts/reproduce_human_correlation.sh
scripts/reproduce_ablations.sh
scripts/reproduce_figures.sh
```

Outputs go to `paper_assets/tables/` and `paper_assets/figures/`.

## Layout

- `configs/`: release counts, model placeholders, rubric and ablation configs.
- `vision2code/data/`: Kaggle loading, manifest validation, split/source metadata.
- `vision2code/rendering/`: Python/Matplotlib, LaTeX, Excalidraw, sandbox, failure taxonomy.
- `vision2code/evaluation/`: dataset rubrics, generic rubric, rater prompts, parsing, aggregation, guardrails.
- `vision2code/generation/`: prompts, code normalization, API/local model generation wrappers.
- `vision2code/metrics/`: embedding similarity baselines and focus texts.
- `vision2code/ablations/`: self-training filters, test-time scaling, tool-use summaries.
- `vision2code/figures/`: table and figure builders.
- `results/paper_outputs/`: compact saved summaries used to regenerate tables.
- `data/fixture_kaggle/`: three-row synthetic Kaggle-layout fixture for CI and smoke checks.

## Full Reruns

Model reruns require provider API keys or local checkpoint paths in `.env`. Training reruns require GPU resources and explicit checkpoint/config paths.
