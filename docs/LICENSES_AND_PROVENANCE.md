# Licenses And Provenance

Repository code is released under Apache-2.0. Dataset sources retain their upstream licenses and terms.

The Kaggle package includes `source_licenses_provenance.csv` with source URI, license metadata, split policy, and provenance notes. The same public source metadata is mirrored in `image2code.data.source_metadata` for validation and documentation generation.

Important release policy:

- No raw local-only data is included in this repository.
- No API keys, `.env` file, checkpoints, W&B runs, logs, caches, rendered image forests, or private launch scripts are included.
- Human-validation files are processed/anonymized summaries only.
- License conflicts or ambiguous upstream metadata are documented in the provenance table rather than normalized away.
