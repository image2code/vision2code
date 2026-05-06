from __future__ import annotations

import argparse
from pathlib import Path

from vision2code.utils.io import read_csv, write_csv
from vision2code.utils.paths import repo_root


GROUPS = {
    "main": {
        "source_dir": ("results", "paper_outputs", "main_leaderboard"),
        "files": {
            "paper_table_summary.csv": "paper_table_summary.csv",
            "main_benchmark_ratings.csv": "main_leaderboard.csv",
            "main_benchmark_ratings_by_dataset.csv": "main_leaderboard_by_dataset.csv",
            "generic_rubric_ratings.csv": "generic_rubric_table.csv",
            "cosine_similarity_scores.csv": "cosine_similarity_table.csv",
            "render_success_rates.csv": "render_success_table.csv",
        },
    },
    "error": {
        "source_dir": ("results", "paper_outputs", "render_failures"),
        "files": {
            "render_error_analysis_table.csv": "render_error_analysis_table.csv",
            "render_failure_type_counts_filtered_test.csv": "render_failure_type_counts_filtered_test.csv",
            "render_failure_type_by_model_filtered_test.csv": "render_failure_type_by_model_filtered_test.csv",
        },
    },
    "ablations": {
        "source_dir": ("results", "paper_outputs", "ablations"),
        "files": {
            "test_time_scaling_scores.csv": "test_time_scaling_scores.csv",
            "self_training_summary.csv": "self_training_summary.csv",
            "tool_use_ablation_summary.csv": "tool_use_ablation_summary.csv",
        },
    },
}


def reproduce(group: str = "all", output_dir: Path | None = None) -> list[Path]:
    root = repo_root()
    out = output_dir or root / "paper_assets" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    selected = list(GROUPS) if group == "all" else [group]
    outputs: list[Path] = []
    for name in selected:
        if name not in GROUPS:
            raise ValueError(f"unknown table group: {name}")
        spec = GROUPS[name]
        src = root.joinpath(*spec["source_dir"])
        for source_name, target_name in spec["files"].items():
            source_path = src / source_name
            if not source_path.exists():
                continue
            target = out / target_name
            write_csv(target, read_csv(source_path))
            outputs.append(target)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["all", *GROUPS], default="all")
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    for path in reproduce(args.group, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
