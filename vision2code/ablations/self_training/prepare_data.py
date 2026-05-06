from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from vision2code.utils.io import write_json, write_jsonl

VARIANT_ORDER = [
    "all_valid",
    "r1_ge_alpha_r2_ge_r1",
    "r1_ge_alpha_r2_lt_r1",
    "r1_lt_alpha",
    "r1_ge_alpha",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def first_existing_path(*values: Any) -> Path:
    for value in values:
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    return Path("")


def rating_values(pair_payload: Mapping[str, Any]) -> tuple[str | None, str | None, float | None, float | None]:
    row = pair_payload.get("row") if isinstance(pair_payload.get("row"), dict) else {}
    image01 = pair_payload.get("image0_image1") if isinstance(pair_payload.get("image0_image1"), dict) else {}
    image12 = pair_payload.get("image1_image2") if isinstance(pair_payload.get("image1_image2"), dict) else {}
    s1 = row.get("image0_image1_rating_status") or image01.get("rating_status")
    s2 = row.get("image1_image2_rating_status") or image12.get("rating_status")
    r1 = row.get("image0_image1_final_0_to_5")
    r2 = row.get("image1_image2_final_0_to_5")
    if r1 is None and isinstance(image01.get("rating"), dict):
        r1 = image01["rating"].get("final_rating_0_to_5")
    if r2 is None and isinstance(image12.get("rating"), dict):
        r2 = image12["rating"].get("final_rating_0_to_5")
    try:
        return str(s1) if s1 is not None else None, str(s2) if s2 is not None else None, float(r1), float(r2)
    except (TypeError, ValueError):
        return str(s1) if s1 is not None else None, str(s2) if s2 is not None else None, None, None


def build_record(
    *,
    sample_dir: Path,
    result_path: Path,
    result: Mapping[str, Any],
    rating_path: Path,
    r1: float,
    r2: float,
    max_code_chars: int,
) -> dict[str, Any] | None:
    rendered_image_path = first_existing_path(
        result.get("sft_input_image_path"),
        result.get("rendered_image_path"),
        sample_dir / "rendered_image.png",
    )
    code_path = first_existing_path(result.get("sft_code_path"), result.get("generated_code_path"), sample_dir / "generated_code.py")
    original_source_image_path = first_existing_path(
        result.get("source_image_path"),
        result.get("copied_source_image_path"),
        sample_dir / "source_image_1.png",
    )
    if not rendered_image_path.exists() or not code_path.exists():
        return None
    code_text = code_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not code_text:
        return None
    if max_code_chars > 0 and len(code_text) > max_code_chars:
        return None
    question_folder = str(result.get("question_folder") or sample_dir.name)
    code_hash = sha256_text(code_text)
    sample_uid = sha256_text(f"{question_folder}::{rendered_image_path}::{code_hash}")[:24]
    return {
        "sample_uid": sample_uid,
        "dataset": str(result.get("dataset") or "unknown"),
        "question_folder": question_folder,
        "source_image_path": str(rendered_image_path),
        "generation_json_path": str(result_path),
        "code_path": str(code_path),
        "code_text": code_text,
        "image_sha256": sha256_file(rendered_image_path),
        "code_sha256": code_hash,
        "snapshot_id": "base",
        "is_dev": False,
        "ssl_sample_dir": str(sample_dir),
        "original_source_image_path": str(original_source_image_path) if original_source_image_path.exists() else "",
        "rendered_image_path": str(rendered_image_path),
        "similarity": result.get("similarity"),
        "render_status": str(result.get("status") or ""),
        "model_slug": str(result.get("model_slug") or "unknown"),
        "dataset_variant": "base",
        "r1_score": r1,
        "r2_score": r2,
        "self_gap": r1 - r2,
        "rating_result_path": str(rating_path),
    }


def collect_base_records(root: Path, rating_glob: str, max_code_chars: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for sample_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name):
        stats["sample_dirs_seen"] += 1
        rating_paths = sorted(sample_dir.glob(rating_glob))
        if not rating_paths:
            stats["missing_pair_rating"] += 1
            continue
        rating_path = rating_paths[0]
        pair_payload = load_json(rating_path)
        if pair_payload is None:
            stats["unparseable_pair_rating"] += 1
            continue
        s1, s2, r1, r2 = rating_values(pair_payload)
        if s1 != "ok" or s2 != "ok" or r1 is None or r2 is None:
            stats["non_ok_or_missing_rating"] += 1
            continue
        result_path = sample_dir / "result.json"
        result = load_json(result_path)
        if result is None:
            stats["missing_or_unparseable_result"] += 1
            continue
        if result.get("render_success") is not True:
            stats["stage1_render_failed"] += 1
            continue
        record = build_record(
            sample_dir=sample_dir,
            result_path=result_path,
            result=result,
            rating_path=rating_path,
            r1=r1,
            r2=r2,
            max_code_chars=max_code_chars,
        )
        if record is None:
            stats["missing_stage1_assets"] += 1
            continue
        records.append(record)
        stats["eligible_records"] += 1
    return records, dict(stats)


def variant_predicates(alpha: float) -> dict[str, Callable[[Mapping[str, Any]], bool]]:
    return {
        "all_valid": lambda r: True,
        "r1_ge_alpha_r2_ge_r1": lambda r: float(r["r1_score"]) >= alpha and float(r["r2_score"]) >= float(r["r1_score"]),
        "r1_ge_alpha_r2_lt_r1": lambda r: float(r["r1_score"]) >= alpha and float(r["r2_score"]) < float(r["r1_score"]),
        "r1_lt_alpha": lambda r: float(r["r1_score"]) < alpha,
        "r1_ge_alpha": lambda r: float(r["r1_score"]) >= alpha,
    }


def clone_for_variant(record: Mapping[str, Any], variant_name: str) -> dict[str, Any]:
    out = dict(record)
    out["snapshot_id"] = variant_name
    out["dataset_variant"] = variant_name
    out["sample_uid"] = sha256_text(f"{variant_name}::{record['question_folder']}::{record['rendered_image_path']}::{record['code_sha256']}")[:24]
    out["is_dev"] = False
    return out


def select_variant_records(records: list[dict[str, Any]], variant_name: str, alpha: float, sample_size: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    predicate = variant_predicates(alpha)[variant_name]
    candidates = [record for record in records if predicate(record)]
    candidates.sort(key=lambda r: (str(r.get("question_folder", "")), str(r.get("sample_uid", ""))))
    random.Random(f"{seed}:{variant_name}").shuffle(candidates)
    if sample_size > 0 and len(candidates) < sample_size:
        raise RuntimeError(f"{variant_name} has only {len(candidates)} candidates; requested sample_size={sample_size}")
    selected_raw = candidates[:sample_size] if sample_size > 0 else candidates
    return [clone_for_variant(record, variant_name) for record in selected_raw], len(candidates)


def split_train_dev(records: list[dict[str, Any]], dev_ratio: float, seed: int, variant_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    indexes = list(range(len(records)))
    random.Random(f"{seed}:{variant_name}:dev").shuffle(indexes)
    n_dev = round(len(records) * dev_ratio)
    n_dev = max(1, min(n_dev, len(records) - 1)) if len(records) > 1 else 0
    dev_indexes = set(indexes[:n_dev])
    for index, record in enumerate(records):
        record["is_dev"] = index in dev_indexes
    return [r for r in records if not r["is_dev"]], [r for r in records if r["is_dev"]]


def write_variant(out_dir: Path, variant_name: str, records: list[dict[str, Any]], candidate_count: int, raw_stats: Mapping[str, int], args: argparse.Namespace) -> dict[str, Any]:
    train_records, dev_records = split_train_dev(records, args.dev_ratio, args.seed, variant_name)
    stats = {
        "dataset_variant": variant_name,
        "threshold": args.threshold,
        "sample_size": args.sample_size,
        "candidate_records": candidate_count,
        "total_records": len(records),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "rating_glob": args.rating_glob,
        "max_code_chars": args.max_code_chars,
        "raw_stats": dict(raw_stats),
        "by_dataset": dict(sorted(Counter(str(r.get("dataset", "unknown")) for r in records).items())),
    }
    if args.dry_run:
        return stats
    variant_dir = out_dir / variant_name
    write_jsonl(variant_dir / "manifest.jsonl", records)
    write_jsonl(variant_dir / "train.jsonl", train_records)
    write_jsonl(variant_dir / "dev.jsonl", dev_records)
    write_json(variant_dir / "stats.json", stats)
    if args.write_hf:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("Install `datasets` or pass --no-write-hf.") from exc
        Dataset.from_list(train_records).save_to_disk(str(variant_dir / "hf_train"))
        Dataset.from_list(dev_records).save_to_disk(str(variant_dir / "hf_dev"))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-training ablation dataset utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build the five self-training data variants.")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, default=Path("results/ablations/self_training_datasets"))
    build.add_argument("--threshold", type=float, default=4.0)
    build.add_argument("--sample-size", type=int, default=1412, help="0 means keep all candidates.")
    build.add_argument("--dev-ratio", type=float, default=0.10)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--rating-glob", default="ssl_stage_pair_rating_result__*.json")
    build.add_argument("--max-code-chars", type=int, default=0)
    build.add_argument("--write-hf", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> None:
    if not args.root.exists():
        raise RuntimeError(f"Input root does not exist: {args.root}")
    base_records, raw_stats = collect_base_records(args.root, args.rating_glob, args.max_code_chars)
    if not base_records:
        raise RuntimeError("No eligible records found.")
    summary: dict[str, Any] = {
        "threshold": args.threshold,
        "sample_size": args.sample_size,
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "root": str(args.root),
        "out_dir": str(args.out_dir),
        "dry_run": args.dry_run,
        "raw_stats": raw_stats,
        "variants": {},
    }
    for variant_name in VARIANT_ORDER:
        records, candidate_count = select_variant_records(base_records, variant_name, args.threshold, args.sample_size, args.seed)
        stats = write_variant(args.out_dir, variant_name, records, candidate_count, raw_stats, args)
        summary["variants"][variant_name] = {
            "candidate_records": candidate_count,
            "total_records": stats["total_records"],
            "train_records": stats["train_records"],
            "dev_records": stats["dev_records"],
        }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.dry_run:
        write_json(args.out_dir / "summary.json", summary)


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)


if __name__ == "__main__":
    main()

