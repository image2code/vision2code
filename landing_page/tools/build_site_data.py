#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
SITE = Path(__file__).resolve().parents[1]
PLOTS = WORKSPACE / "annotation" / "plots" / "images"

DATA_DIR = SITE / "static" / "data"
FIGURE_DIR = SITE / "static" / "images" / "figures"
EXAMPLE_DIR = SITE / "static" / "images" / "examples"
VIEWER_DIR = SITE / "static" / "images" / "viewer"

CATEGORY_BY_DATASET = {
    "ChartQA": "Charts&Plots",
    "dvqa": "Charts&Plots",
    "figureqa": "Charts&Plots",
    "matplotlib": "Charts&Plots",
    "geometry3k": "Geometry",
    "GEOQA_8K_R1V": "Geometry",
    "Geoperception": "Geometry",
    "Graph-Algorithms": "Graphs",
    "GraphVQA-Swift": "Graphs",
    "ChemVQA-2K": "Science",
    "EEE-Bench": "Science",
    "OlympiadBench": "Geometry",
    "Physics": "Science",
    "DocVQA": "Documents",
    "spatialvlm_qa": "Spatial",
}

MODEL_LABELS = {
    "qwen35_9b": "Qwen3.5-9B",
    "kimi_k2_5": "Kimi-K2.5",
    "kimi_k2_5_reasoning": "Kimi-K2.5+R",
    "qwen35_397b_a17b": "Qwen3.5-397B-A17B",
    "qwen35_397b_a17b_reasoning": "Qwen3.5-397B-A17B+R",
    "gpt_5_4": "GPT-5.4",
    "gpt_5_4_mini": "GPT-5.4 Mini",
    "gpt_5_4_mini_reasoning_medium": "GPT-5.4 Mini+R",
    "gemini_3_1_flash_lite_preview": "Gemini-3.1 Flash Lite Preview",
}

SENSITIVE_PATTERNS = [
    "A" + "jay",
    "Vik" + "ram",
    "Peri" + "asami",
    "Jun" + "lin",
    "Bhu" + "wan",
    "Dhin" + "gra",
    "Du" + "ke",
    "image2" + "code",
]


def slugify(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "item"


def scrub(value):
    if isinstance(value, str):
        for pattern in SENSITIVE_PATTERNS:
            value = re.sub(re.escape(pattern), "[redacted]", value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scrub(data), handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def copy_asset(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(SITE)).replace("\\", "/")


def float_or_none(value: str):
    if value == "" or value is None:
        return None
    return float(value)


def build_figures() -> dict[str, str]:
    assets = {}
    for name in [
        "benchmark_examples.pdf",
        "benchmark_examples.png",
        "dataset_stats_horizontal.png",
        "rubric_scoring_pipeline.png",
        "correlation_1_heatmap.png",
        "render_success_rate.png",
    ]:
        src = PLOTS / name
        if src.exists():
            assets[Path(name).stem] = copy_asset(src, FIGURE_DIR / name)
    return assets


def build_leaderboard() -> None:
    main_rows = read_csv(ROOT / "paper_assets" / "tables" / "main_leaderboard.csv")
    by_dataset_rows = read_csv(ROOT / "paper_assets" / "tables" / "main_leaderboard_by_dataset.csv")

    dataset_scores: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in by_dataset_rows:
        split = row["split"]
        model = row["model_slug"]
        dataset = row["dataset"]
        dataset_scores[(split, model)].append(
            {
                "dataset": dataset,
                "domain": CATEGORY_BY_DATASET.get(dataset, "Other"),
                "score": float_or_none(row["benchmark_score_0_to_5"]),
                "samples": int(row["num_filtered_samples"]),
                "render_success_rate": (
                    int(row["num_render_success"]) / int(row["num_filtered_samples"])
                    if int(row["num_filtered_samples"])
                    else None
                ),
            }
        )

    rows = []
    for row in main_rows:
        samples = int(row["num_filtered_samples"])
        render_success = int(row["num_render_success"])
        rows.append(
            {
                "split": row["split"],
                "model_slug": row["model_slug"],
                "model": MODEL_LABELS.get(row["model_slug"], row["model_slug"]),
                "samples": samples,
                "rated": int(row["num_rated_final"]),
                "missing_ratings": int(row["num_missing_rating"]),
                "score": float_or_none(row["benchmark_score_0_to_5"]),
                "mean_final_rating": float_or_none(row["mean_final_rating_0_to_5"]),
                "mean_raw_score": float_or_none(row["mean_raw_score_0_to_5"]),
                "render_success_rate": render_success / samples if samples else None,
                "render_success_count": render_success,
                "datasets": sorted(
                    dataset_scores[(row["split"], row["model_slug"])],
                    key=lambda item: (item["domain"], item["dataset"]),
                ),
            }
        )

    write_json(
        DATA_DIR / "leaderboard_data.json",
        {
            "default_split": "filtered_test",
            "splits": sorted({row["split"] for row in rows}),
            "rows": rows,
        },
    )


def build_examples() -> None:
    examples = []
    for selection_path in sorted(PLOTS.glob("rating_examples_testmini_selected*_selection.json")):
        image_path = selection_path.with_name(selection_path.name.replace("_selection.json", ".png"))
        if not image_path.exists():
            continue
        with selection_path.open(encoding="utf-8") as handle:
            selection = json.load(handle)
        selected = selection["selected_sample"]
        dst_name = image_path.name
        rel_image = copy_asset(image_path, EXAMPLE_DIR / dst_name)
        dataset = selected["dataset"]
        examples.append(
            {
                "id": slugify(dst_name.removesuffix(".png")),
                "dataset": dataset,
                "domain": CATEGORY_BY_DATASET.get(dataset, "Other"),
                "question_folder": selected["question_folder"],
                "question": selected.get("question", ""),
                "score_median": selected.get("score_median"),
                "score_spread": selected.get("score_spread"),
                "rendered_count": selected.get("rendered_count"),
                "rated_count": selected.get("rated_count"),
                "image": rel_image,
                "scores": [
                    {
                        "model": MODEL_LABELS.get(model, model),
                        "score": score,
                        "rendered": selected.get("render_success_by_model", {}).get(model),
                    }
                    for model, score in selected.get("scores_by_model", {}).items()
                ],
            }
        )

    examples.sort(key=lambda item: (item["domain"], item["dataset"]))
    write_json(DATA_DIR / "examples_data.json", {"examples": examples})


def build_viewer() -> None:
    with (ROOT / "configs" / "data" / "test-filtered.json").open(encoding="utf-8") as handle:
        rows = json.load(handle)

    samples = []
    missing = []
    dataset_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        dataset = row["dataset"]
        domain = CATEGORY_BY_DATASET.get(dataset, "Other")
        dataset_counts[dataset] += 1
        domain_counts[domain] += 1

        image_entries = []
        question_dir = WORKSPACE / row["question_dir"]
        for image_name in row.get("saved_images", []):
            src = question_dir / image_name
            rel_dst = Path(slugify(dataset)) / row["question_folder"] / image_name
            dst = VIEWER_DIR / rel_dst
            if src.exists():
                rel_path = copy_asset(src, dst)
                image_entries.append({"src": rel_path, "name": image_name})
            else:
                missing.append(str(src))

        samples.append(
            {
                "id": index,
                "dataset": dataset,
                "domain": domain,
                "question_folder": row["question_folder"],
                "question": row.get("question", ""),
                "sample_id": row.get("sample_id", ""),
                "subset": row.get("subset", ""),
                "selection_rank_global": row.get("selection_rank_global"),
                "selection_rank_within_dataset": row.get("selection_rank_within_dataset"),
                "images": image_entries,
            }
        )

    datasets = [
        {
            "name": dataset,
            "domain": CATEGORY_BY_DATASET.get(dataset, "Other"),
            "count": count,
        }
        for dataset, count in sorted(dataset_counts.items())
    ]
    domains = [{"name": name, "count": count} for name, count in sorted(domain_counts.items())]

    write_json(
        DATA_DIR / "viewer_data.json",
        {
            "split": "test",
            "sample_count": len(samples),
            "image_count": sum(len(item["images"]) for item in samples),
            "domains": domains,
            "datasets": datasets,
            "samples": samples,
            "missing_images": missing,
        },
    )


def build_site_data(figures: dict[str, str]) -> None:
    domain_rows = read_csv(ROOT / "paper_assets" / "tables" / "benchmark_stats_by_domain_test.csv")
    dataset_rows = read_csv(ROOT / "paper_assets" / "tables" / "benchmark_stats_by_dataset_test.csv")
    human_rows = read_csv(ROOT / "paper_assets" / "tables" / "human_alignment_correlations.csv")

    human_alignment = None
    for row in human_rows:
        if row["group"] == "pooled" and row["group_value"] == "all" and row["metric"] == "ours_final":
            human_alignment = {
                "n": int(row["n"]),
                "pearson": float_or_none(row["pearson"]),
                "spearman": float_or_none(row["spearman"]),
                "mean_human": float_or_none(row["mean_human"]),
                "mean_metric": float_or_none(row["mean_metric"]),
            }
            break

    write_json(
        DATA_DIR / "site_data.json",
        {
            "title": "Vision2Code: A Multi-Domain Benchmark for Evaluating Image-to-Code Generation",
            "figures": figures,
            "benchmark": {
                "full_test_samples": 2169,
                "test_mini_samples": 539,
                "domains": [
                    {"name": row["domain"], "count": int(row["count"])}
                    for row in domain_rows
                ],
                "datasets": [
                    {
                        "name": row["dataset"],
                        "domain": row["domain"],
                        "count": int(row["count"]),
                    }
                    for row in dataset_rows
                ],
                "dataset_count": len(dataset_rows),
                "domain_count": len(domain_rows),
            },
            "evaluation": {
                "human_alignment": human_alignment,
                "rating_scale": "0 to 5",
                "primary_metric": "rubric-weighted rendered recreation score",
            },
        },
    )


def main() -> None:
    for directory in [DATA_DIR, FIGURE_DIR, EXAMPLE_DIR, VIEWER_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    figures = build_figures()
    build_site_data(figures)
    build_leaderboard()
    build_examples()
    build_viewer()

    print("Built site data")
    print(f"  site: {SITE}")
    print(f"  data: {DATA_DIR}")
    print(f"  viewer images: {VIEWER_DIR}")


if __name__ == "__main__":
    main()
