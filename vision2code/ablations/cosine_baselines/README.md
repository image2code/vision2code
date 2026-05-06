# Cosine Similarity Baselines

The original paper summaries are saved under:

```text
results/paper_outputs/main_leaderboard/cosine_similarity_scores.csv
results/paper_outputs/main_leaderboard/cosine_similarity_scores_by_dataset.csv
```

`run_embeddings.py` recomputes per-sample cosine scores from a benchmark inference CSV. The paper baseline used Qwen3-VL-Embedding-8B embeddings with image-only and image-plus-text focus prompts. The script includes:

- `--provider pixel`: lightweight local smoke check with deterministic image vectors.
- `--provider qwen_vllm`: paper-style Qwen3-VL embedding runner using vLLM's pooling/embed API.

Smoke example:

```bash
python3 -m vision2code.ablations.cosine_baselines.run_embeddings \
  --provider pixel \
  --inference-csv results/outputs/gpt_5_4_mini_api_smoke/generations/benchmark/test_mini/benchmark_inference.csv \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --output-dir results/ablations/cosine_smoke
```

Qwen embedding example:

```bash
python3 -m vision2code.ablations.cosine_baselines.run_embeddings \
  --provider qwen_vllm \
  --embedding-model Qwen/Qwen3-VL-Embedding-8B \
  --inference-csv results/outputs/<model_slug>/generations/benchmark/test_mini/benchmark_inference.csv \
  --data_dir "$VISION2CODE_DATA_DIR" \
  --output-dir results/ablations/cosine_qwen/<model_slug>/test_mini \
  --batch-size 32 \
  --dtype bfloat16 \
  --max-model-len 32768
```

The Qwen path requires a GPU environment with `vllm`, `torch`, `transformers`, and access to the embedding model weights.
