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
RATER_BASE_URL=http://127.0.0.1:8000/v1
RATER_MODEL=Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
RATER_API_KEY=EMPTY
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

## Local Rater Setup

The default evaluator uses a local OpenAI-compatible vLLM server:

```text
Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
http://127.0.0.1:8000/v1
```

Install vLLM in the environment where you will run the rater. This is separate from the base install because vLLM is GPU- and platform-specific.

```bash
pip install vllm huggingface_hub
bash scripts/download_rater_model.sh
```

For a custom model cache, export the same cache setting before both download and server startup:

```bash
export HF_HOME=/path/to/hf_cache
bash scripts/download_rater_model.sh
```

Keep `HF_HOME` set in the terminal where you start vLLM.

Start the rater in one terminal:

```bash
bash scripts/start_local_rater.sh
```

Check it from another terminal:

```bash
python3 scripts/check_local_rater.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
```

If the model is already downloaded somewhere else, point vLLM at that path while keeping the served model name fixed:

```bash
RATER_MODEL_PATH=/path/to/model_or_snapshot \
bash scripts/start_local_rater.sh
```

## One-Sample OpenAI Smoke Test

This tests OpenAI generation plus the default local rater. Add `OPENAI_API_KEY` to `.env` first, and keep the local rater server running in a separate terminal.

```bash
bash scripts/run_openai_smoke_default_rater.sh \
  --data_dir "$VISION2CODE_DATA_DIR"
```

Equivalent explicit command:

```bash
python3 -m vision2code.benchmark.run_benchmark \
  --provider openai \
  --model gpt-5.4-mini \
  --model-slug gpt_5_4_mini_smoke \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test_mini \
  --num_samples 1 \
  --output_root results/outputs \
  --rater-provider local_vllm \
  --rater-model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --rater-base-url http://127.0.0.1:8000/v1 \
  --rater-api-key EMPTY \
  --env-file .env
```

For a quick API-only check without the local rater, use `--rater-provider openai --rater-model gpt-5.4-mini`.

Outputs are written under:

```text
results/outputs/gpt_5_4_mini_smoke/generations/benchmark/test_mini/
```

Each question folder contains the copied source image, `metadata.json`, `generated_code.py`,
`rendered_image.png` when rendering succeeds, `execution_error.txt` when rendering fails,
`result.json`, rating JSON, and the raw rater response.

A one-sample API smoke-test output is included for inspection:

```text
results/outputs/gpt_5_4_mini_api_smoke/generations/benchmark/test_mini/
```

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
  --rater-base-url http://127.0.0.1:8000/v1 \
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
  --rater-base-url http://127.0.0.1:8000/v1 \
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
scripts/reproduce_benchmark_stats.sh
```

Static figure assets such as `benchmark_statistics.png`, `pipeline.png`, and
`self_improvement_pipeline.png` are kept under `paper_assets/figures/`; they are not regenerated by
these scripts.

Optional ablation rerun entrypoints are provided under `vision2code/ablations/`:

```bash
bash scripts/prepare_self_training_data.sh --help
bash scripts/train_self_training_model.sh --help
bash scripts/run_test_time_scaling.sh --help
bash scripts/run_cosine_baselines.sh --help
bash scripts/run_tool_use_ablation.sh --help
bash scripts/render_tool_use_ablation.sh --help
bash scripts/evaluate_tool_use_ablation.sh --help
```

These require user-provided candidate outputs, manifests, API keys, or GPU/model resources depending on
the ablation. The saved paper summaries remain under `results/paper_outputs/`.

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
vision2code/ablations/    self-training, scaling, cosine, and tool-use ablation entrypoints
vision2code/tables/       CSV table and benchmark-stat reproduction
results/paper_outputs/    saved outputs for table reproduction
results/outputs/          included one-sample benchmark output
paper_assets/tables/      reproduced CSV tables
paper_assets/figures/     static paper figures
```
