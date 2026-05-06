# Self-Training Ablation

`prepare_data.py` builds five SFT data variants from generated samples that have paired self-improvement ratings.

Required input root:

```text
<root>/<sample_id>/
  result.json
  generated_code.py
  rendered_image.png
  ssl_stage_pair_rating_result__*.json
```

Filters match the experiment definitions:

- `all_valid`
- `r1_ge_alpha_r2_ge_r1`
- `r1_ge_alpha_r2_lt_r1`
- `r1_lt_alpha`
- `r1_ge_alpha`

Example:

```bash
python3 -m vision2code.ablations.self_training.prepare_data \
  build \
  --root results/outputs/self_training_candidates \
  --out-dir results/ablations/self_training_datasets \
  --threshold 4.0 \
  --sample-size 1412 \
  --dry-run
```

To write Hugging Face datasets for training, add `--write-hf`:

```bash
python3 -m vision2code.ablations.self_training.prepare_data \
  build \
  --root results/outputs/self_training_candidates \
  --out-dir results/ablations/self_training_datasets \
  --threshold 4.0 \
  --sample-size 1412 \
  --write-hf
```

Full fine-tuning uses the Qwen3-VL collator and TRL SFT trainer:

```bash
python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node 4 \
  -m vision2code.ablations.self_training.train_self_training_sft \
  --train-dataset results/ablations/self_training_datasets/r1_ge_alpha_r2_ge_r1/hf_train \
  --dev-dataset results/ablations/self_training_datasets/r1_ge_alpha_r2_ge_r1/hf_dev \
  --output-dir results/ablations/self_training_checkpoints/r1_ge_alpha_r2_ge_r1 \
  --model-id Qwen/Qwen3.5-9B \
  --deepspeed-config configs/ablations/deepspeed_zero3.json
```

Training requires the `train` extras plus a GPU/DeepSpeed environment. W&B is off by default; pass `--report-to wandb --wandb-project <name>` only if you want local experiment logging.
