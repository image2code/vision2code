# Vision2Code

Code, configs, saved outputs, and scripts for running the Vision2Code benchmark.

## Setup

```bash
cd vision2code
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
cp .env.example .env
```

Open `.env` and add the keys you need:

```dotenv
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key
HF_TOKEN=your_huggingface_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
VISION2CODE_DATA_DIR=/path/to/vision2code_kaggle_dataset
```

Do not commit `.env`.

Local model inference needs the local/training extras:

```bash
pip install -e ".[train]"
```

## Data Loading

```bash
scripts/download_kaggle_data.sh --data_dir data/kaggle/vision2code
export VISION2CODE_DATA_DIR="$PWD/data/kaggle/vision2code"
python3 scripts/validate_release_data.py --data_dir "$VISION2CODE_DATA_DIR"
```

Expected dataset layout:

```text
manifest.csv
manifest.jsonl
images/
source_licenses_provenance.csv
croissant.json
```

## One-Sample OpenAI Smoke Test

This writes the same directory layout used by full benchmark runs.

```bash
python3 -m vision2code.benchmark.run_benchmark \
  --provider openai \
  --model gpt-5.4-mini \
  --model-slug gpt_5_4_mini_smoke \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test_mini \
  --num_samples 1 \
  --output_root results/outputs \
  --rater-provider openai \
  --rater-model gpt-5.4-mini \
  --env-file .env
```

Outputs are written under:

```text
results/outputs/gpt_5_4_mini_smoke/generations/benchmark/test_mini/
```

Each question folder contains the copied source image, `metadata.json`, `generated_code.py`,
`rendered_image.png` when rendering succeeds, `execution_error.txt` when rendering fails,
`result.json`, rating JSON, and the raw rater response.

## Full Test-Mini/Test Inference And Eval

OpenAI API model:

```bash
python3 -m vision2code.benchmark.run_benchmark \
  --provider openai \
  --model gpt-5.4-mini \
  --model-slug gpt_5_4_mini \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test_mini \
  --num_samples 0 \
  --output_root results/outputs \
  --rater-provider local_vllm \
  --rater-model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --rater-api-key EMPTY \
  --env-file .env
```

Local Hugging Face model:

```bash
python3 -m vision2code.benchmark.run_benchmark \
  --provider local \
  --model /path/to/local/checkpoint \
  --model-slug local_model_slug \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test \
  --num_samples 0 \
  --output_root results/outputs \
  --rater-provider local_vllm \
  --rater-model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --rater-api-key EMPTY \
  --env-file .env
```

Use `--split test_mini` for 539 examples and `--split test` for 2169 examples. Check row counts without model calls:

```bash
python3 -m vision2code.benchmark.run_benchmark \
  --provider openai \
  --model gpt-5.4-mini \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test_mini \
  --num_samples 0 \
  --dry-run
```

Aggregate files are written in the split directory:

```text
benchmark_inference.csv
benchmark_inference.json
benchmark_eval__<rater_slug>.csv
benchmark_eval__<rater_slug>.json
benchmark_summary__<rater_slug>.json
```

## Paper Results From Saved Outputs

These commands read saved outputs under `results/paper_outputs/` and write CSV tables under
`paper_assets/tables/`. They do not call model APIs.

```bash
scripts/reproduce_main_tables.sh
scripts/reproduce_error_analysis.sh
scripts/reproduce_human_correlation.sh
scripts/reproduce_ablations.sh
scripts/reproduce_figures.sh
```

`scripts/reproduce_figures.sh` only regenerates benchmark statistics. Static figure assets such as
`pipeline.png` and `self_improvement_pipeline.png` are kept under `paper_assets/figures/`.

## Directory Structure

```text
configs/                  release counts, model placeholders, rubric and ablation configs
data/                     fixture data and dataset notes
docs/                     dataset, compute, validation, and provenance notes
vision2code/benchmark/    benchmark inference, rendering, and evaluation runner
vision2code/data/         Kaggle loading and manifest validation
vision2code/rendering/    Python/Matplotlib, LaTeX, Excalidraw renderers
vision2code/evaluation/   dataset rubrics, generic rubric, parsing, guardrails
vision2code/generation/   benchmark prompts and code normalization
vision2code/metrics/      embedding similarity helpers and focus texts
vision2code/tables/       CSV table reproduction from saved outputs
vision2code/figures/      benchmark statistics only
results/paper_outputs/    saved outputs for table reproduction
paper_assets/tables/      reproduced CSV tables
paper_assets/figures/     benchmark/static figures
```
