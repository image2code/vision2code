# Human Validation

Processed human-validation files are anonymized and stored under `results/paper_outputs/human_validation/`.

Included files:

- `human_alignment_correlations.csv`
- `human_alignment_bootstrap_deltas.csv`
- `human_alignment_summary.json`
- `human_alignment_joined_anonymized.csv`
- `human_ratings_anonymized.csv`

Task identifiers are synthetic, submission timestamps are redacted, and annotator names, emails, IP addresses, institution names, and private metadata are excluded.

To reproduce the paper correlation tables:

```bash
scripts/reproduce_human_correlation.sh
```

The script copies the compact saved summaries into `paper_assets/tables/`. Recomputing from raw annotation exports is intentionally not supported in this release because raw exports can contain private annotator metadata.
