# Reproducibility

## Level 1: Quick Smoke Test

Validates the Kaggle layout, loads a manifest, renders a tiny Matplotlib sample in a temp directory, and regenerates table inputs from saved outputs.

```bash
python3 scripts/validate_release_data.py --data_dir "$VISION2CODE_DATA_DIR"
scripts/run_smoke_test.sh --data_dir "$VISION2CODE_DATA_DIR" --num_samples 3
```

No Kaggle data available:

```bash
scripts/run_smoke_test.sh --data_dir data/fixture_kaggle --num_samples 3
```

## Level 2: Tables And Benchmark Statistics From Saved Outputs

```bash
scripts/reproduce_main_tables.sh
scripts/reproduce_error_analysis.sh
scripts/reproduce_human_correlation.sh
scripts/reproduce_ablations.sh
scripts/reproduce_benchmark_stats.sh
```

These commands read saved CSV/JSON files and write derived CSV files under `paper_assets/tables/`. They do not call model APIs.
Static figure files are kept under `paper_assets/figures/`.

## Level 3: Full Evaluation Rerun

Provide keys in `.env` and pass explicit output directories:

```bash
python3 -m vision2code.generation.run_model_generation --manifest "$VISION2CODE_DATA_DIR/manifest.csv" --model openai:gpt-4o --output results/rerun/model_outputs.jsonl
python3 -m vision2code.rendering.render_python results/rerun/example.py --output results/rerun/render.png
python3 -m vision2code.metrics.embedding_similarity --predictions results/rerun/render_manifest.csv --output results/rerun/embedding_scores.csv
python3 -m vision2code.evaluation.aggregate_scores results/rerun/rater_scores.csv --output results/rerun/final_scores.csv
```

Provider wrappers are templates. Proprietary model reproduction requires user-provided `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `TOGETHER_API_KEY`, or local Hugging Face credentials and checkpoint paths.

## Level 4: Training And Ablations

Self-training filter definitions are implemented in `vision2code.ablations.self_training.filters`:

- `all_valid`
- `R1 >= alpha and R2 >= R1`
- `R1 >= alpha and R2 < R1`
- `R1 < alpha`
- `R1 >= alpha`

Template entrypoints:

```bash
python3 -m vision2code.ablations.self_training.build_datasets --input trajectories.jsonl --output data/self_training --alpha 0.7
python3 -m vision2code.ablations.test_time_scaling.run_scaling --input generations.jsonl --output results/scaling
python3 -m vision2code.ablations.tool_use.summarize --root results/paper_outputs/ablations/tool_use --output paper_assets/tables/tool_use_ablation_summary.csv
```

Training commands intentionally use placeholders for checkpoint names, output directories, and compute resources.
