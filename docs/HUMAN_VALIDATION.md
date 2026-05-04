# Human Validation

Processed human-validation files are stored under `results/paper_outputs/human_validation/`.

Included files:

- `human_alignment_correlations.csv`
- `human_alignment_bootstrap_deltas.csv`
- `human_alignment_summary.json`
- `human_alignment_joined.csv`
- `human_ratings.csv`

Task identifiers are synthetic, submission timestamps are redacted, and annotator names, emails, IP addresses, institution names, and private metadata are excluded.

To reproduce the human-correlation tables:

```bash
scripts/reproduce_human_correlation.sh
```

The script copies the compact saved summaries into `paper_assets/tables/`. Recomputing from raw annotation exports is not supported because raw exports can contain private annotator metadata.
