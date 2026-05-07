# Test-Time Scaling Ablation

`run_test_time_scaling.py` reruns staged image-to-code refinement. Stage 1 uses the standard benchmark prompt. Stages 2+ give the model the reference image, previous rendered image, and previous Python code, then rerender the revised code.

Stage 1 writes the same per-sample filenames as the main benchmark runner:
`generated_code.py`, `rendered_image.png`, `execution_error.txt`, and `result.json`.
Later stages write `stage2_generated_code.py`, `stage2_rendered_image.png`, and so on. A stage 2+ run can continue from an existing stage 1 benchmark output directory, or stage 1 and stage 2 can be run together.

Example one-sample OpenAI rerun:

```bash
python3 -m vision2code.ablations.test_time_scaling.run_test_time_scaling \
  --backend openai \
  --model gpt-5.4-mini \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test_mini \
  --stage 1 --stage 2 \
  --num_samples 1 \
  --output_dir results/ablations/test_time_scaling/gpt_5_4_mini/test_mini \
  --env-file .env
```

Expected files in each sample directory after `--stage 1 --stage 2`:

```text
generated_code.py
rendered_image.png
result.json
stage2_generated_code.py
stage2_rendered_image.png
stage2_result.json
```

Local model rerun:

```bash
python3 -m vision2code.ablations.test_time_scaling.run_test_time_scaling \
  --backend local \
  --model /path/to/local/qwen_checkpoint \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --split test_mini \
  --stage 1 --stage 2 \
  --num_samples 1 \
  --output_dir results/ablations/test_time_scaling/local_qwen/test_mini
```

The original paper summaries are in `results/paper_outputs/ablations/test_time_scaling_scores.csv`.
