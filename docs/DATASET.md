# Dataset

The Kaggle package must contain:

```text
manifest.csv
manifest.jsonl
images/
source_licenses_provenance.csv
croissant.json
```

`manifest.csv` and `manifest.jsonl` are expected to describe the same rows. Image paths are relative to the Kaggle root and must point under `images/`.

## Splits

The internal benchmark split name is `test-mini`; the Kaggle package exposes it as `test_mini`. The final release counts are:

- `test`: 2169 rows
- `test_mini`: 539 rows
- combined Kaggle manifest: 2708 rows

The combined Kaggle manifest has 2708 rows because `test_mini` is duplicated as a separately loadable subset of `test`.

## Manifest Columns

Required columns:

- `sample_id`
- `split`
- `source_dataset`
- `image_path`

Optional but validated when present:

- `sha256`
- `question`
- `answer`
- `category`
- `license`
- `source_uri`

Validation checks reject absolute paths, missing images, private/local path strings, rejected examples, split count drift, source count drift, and hash mismatches when `sha256` is populated.

## Source Counts

`test-mini`: ChartQA 45, ChemVQA-2K 63, DocVQA 68, EEE-Bench 22, GEOQA_8K_R1V 21, Geoperception 34, Graph-Algorithms 39, GraphVQA-Swift 42, OlympiadBench 23, Physics 31, dvqa 39, figureqa 39, geometry3k 21, matplotlib 13, spatialvlm_qa 39.

`test`: ChartQA 195, ChemVQA-2K 288, DocVQA 292, EEE-Bench 67, GEOQA_8K_R1V 73, Geoperception 137, Graph-Algorithms 161, GraphVQA-Swift 175, OlympiadBench 81, Physics 121, dvqa 161, figureqa 161, geometry3k 65, matplotlib 31, spatialvlm_qa 161.

Source split policy and provenance metadata are in `image2code.data.source_metadata` and mirrored by the Kaggle `source_licenses_provenance.csv`.
