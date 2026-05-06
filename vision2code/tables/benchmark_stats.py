from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vision2code.data.source_metadata import CATEGORY_BY_DATASET
from vision2code.utils.io import write_csv
from vision2code.utils.paths import repo_root


def reproduce(output_dir: Path | None = None) -> list[Path]:
    root = repo_root()
    out = output_dir or root / "paper_assets" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for split, filename in [("test-mini", "test-mini-filtered.json"), ("test", "test-filtered.json")]:
        rows = json.loads((root / "configs" / "data" / filename).read_text(encoding="utf-8"))
        by_dataset = Counter(str(row.get("dataset") or "") for row in rows)
        by_domain: Counter[str] = Counter()
        for dataset, count in by_dataset.items():
            by_domain[CATEGORY_BY_DATASET.get(dataset, "Other")] += count
        dataset_path = out / f"benchmark_stats_by_dataset_{split}.csv"
        domain_path = out / f"benchmark_stats_by_domain_{split}.csv"
        write_csv(
            dataset_path,
            [
                {
                    "split": split,
                    "dataset": dataset,
                    "count": count,
                    "domain": CATEGORY_BY_DATASET.get(dataset, "Other"),
                }
                for dataset, count in sorted(by_dataset.items())
            ],
        )
        write_csv(
            domain_path,
            [{"split": split, "domain": domain, "count": count} for domain, count in sorted(by_domain.items())],
        )
        outputs.extend([dataset_path, domain_path])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate benchmark statistics CSV files.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in reproduce(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

