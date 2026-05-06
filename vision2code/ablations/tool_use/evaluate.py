from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

from vision2code.benchmark.run_benchmark import summarize_ratings, validate_local_vllm_rater
from vision2code.evaluation.dataset_rubrics import (
    RATER_SYSTEM_PROMPT,
    aggregate_rating,
    build_rater_repair_prompt,
    build_rater_user_content,
    build_render_failure_rating,
    inspect_candidate_image,
    parse_json_object,
)
from vision2code.evaluation.generic_rubric import (
    GENERIC_RATER_SYSTEM_PROMPT,
    aggregate_generic_rating,
    build_generic_rater_repair_prompt,
    build_generic_rater_user_content,
    build_generic_render_failure_rating,
)
from vision2code.evaluation.rate_renders import _build_client, _call_rater_completion, _is_retryable_rater_exception
from vision2code.utils.env import load_env_file
from vision2code.utils.io import read_csv, write_csv, write_json

FIELDNAMES = [
    "task",
    "sample_id",
    "dataset",
    "question_folder",
    "model_id",
    "source_image_path",
    "rendered_image_path",
    "generated_artifact_path",
    "sample_dir",
    "render_success",
    "render_status",
    "rating_status",
    "rubric_dataset",
    "rater_provider",
    "rater_model",
    "rater_reasoning_enabled",
    "rater_slug",
    "rating_path",
    "rating_final_0_to_5",
    "rating_raw_score_0_to_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rendered tool-use ablation outputs.")
    parser.add_argument("--task", choices=["latex_docvqa", "excalidraw_json"], required=True)
    parser.add_argument("--render-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to render CSV parent.")
    parser.add_argument("--model-id", default="gpt-5.4-mini")
    parser.add_argument("--rubric-mode", choices=["auto", "dataset", "generic"], default="auto")
    parser.add_argument("--provider", choices=["openai", "local_vllm", "together"], default="local_vllm")
    parser.add_argument("--rater-model", default="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--rater-base-url", default="")
    parser.add_argument("--rater-api-key", default="")
    parser.add_argument("--rater-api-key-env", default="")
    parser.add_argument("--rater-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rater-temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--rater-api-timeout-sec", type=float, default=300.0)
    parser.add_argument("--rater-max-retries", type=int, default=4)
    parser.add_argument("--rater-repair-attempts", type=int, default=1)
    parser.add_argument("--rater-retry-sleep-sec", type=float, default=3.0)
    parser.add_argument("--rater-retry-jitter-sec", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--force-rerate", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "value"


def rubric_mode(args: argparse.Namespace) -> str:
    if args.rubric_mode != "auto":
        return str(args.rubric_mode)
    return "dataset" if args.task == "latex_docvqa" else "generic"


def eval_slug(args: argparse.Namespace) -> str:
    prefix = "dataset_rubric" if rubric_mode(args) == "dataset" else "generic_rubric"
    suffix = f"{args.provider}_{sanitize_slug(args.rater_model)}"
    if args.rater_reasoning:
        suffix += "_reasoning"
    return f"{prefix}__{suffix}"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def existing_rating_ok(path: Path, force: bool) -> bool:
    if force or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and "final_rating_0_to_5" in payload


def call_and_parse_rating(
    client: Any,
    args: argparse.Namespace,
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    repair_prompt_builder: Any,
) -> tuple[dict[str, Any], str]:
    attempt = 1
    while True:
        try:
            raw_text = _call_rater_completion(
                client,
                provider=args.provider,
                model=args.rater_model,
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=args.max_output_tokens,
                temperature=args.rater_temperature,
                timeout_sec=args.rater_api_timeout_sec,
                reasoning_enabled=args.rater_reasoning,
            )
            try:
                return parse_json_object(raw_text), raw_text
            except Exception as parse_error:
                repaired = raw_text
                for _ in range(args.rater_repair_attempts):
                    repair_prompt = repair_prompt_builder(raw_text, str(parse_error))
                    repaired = _call_rater_completion(
                        client,
                        provider=args.provider,
                        model=args.rater_model,
                        system_prompt=system_prompt,
                        user_content=list(user_content) + [{"type": "text", "text": repair_prompt}],
                        max_tokens=args.max_output_tokens,
                        temperature=0.0,
                        timeout_sec=args.rater_api_timeout_sec,
                        reasoning_enabled=args.rater_reasoning,
                    )
                    return parse_json_object(repaired), repaired
                raise ValueError(f"Failed to parse rater response: {parse_error}") from parse_error
        except Exception as exc:
            if not _is_retryable_rater_exception(exc) or attempt >= args.rater_max_retries:
                raise
            sleep_for = args.rater_retry_sleep_sec * (2 ** (attempt - 1)) + random.uniform(0.0, args.rater_retry_jitter_sec)
            print(f"[RATER-RETRY] attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
            time.sleep(sleep_for)
            attempt += 1


def build_client(args: argparse.Namespace) -> tuple[Any, str, str]:
    load_env_file(args.env_file, override=True)
    default_base_url = {
        "openai": "https://api.openai.com/v1",
        "local_vllm": "http://127.0.0.1:8000/v1",
        "together": "https://api.together.xyz/v1",
    }[args.provider]
    base_url = args.rater_base_url or default_base_url
    api_key_env = args.rater_api_key_env or ("TOGETHER_API_KEY" if args.provider == "together" else "OPENAI_API_KEY")
    api_key = args.rater_api_key or os.getenv(api_key_env) or ("EMPTY" if args.provider == "local_vllm" else "")
    if args.provider in {"openai", "together"} and not api_key:
        raise RuntimeError(f"Set {api_key_env} in the environment or in {args.env_file}.")
    client = _build_client(args.provider, base_url, api_key or "EMPTY", args.rater_api_timeout_sec)
    if args.provider == "local_vllm":
        validate_local_vllm_rater(client, model=args.rater_model, base_url=base_url)
    return client, base_url, api_key_env


def main() -> None:
    args = parse_args()
    rows = read_csv(args.render_csv)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    output_dir = args.output_dir or args.render_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = rubric_mode(args)
    slug = eval_slug(args)

    if args.dry_run:
        preview = {}
        if rows:
            sample_dir = Path(str(rows[0].get("sample_dir") or output_dir))
            preview = {
                "sample_dir": str(sample_dir),
                "rating_path": str(sample_dir / f"rating__{slug}.json"),
                "rubric_mode": mode,
            }
        print(json.dumps({"rows": len(rows), "preview": preview}, indent=2))
        return

    client, base_url, api_key_env = build_client(args)
    print(f"[INFO] rater_base_url={base_url}", flush=True)
    print(f"[INFO] rater_api_key_env={api_key_env}", flush=True)

    eval_rows: list[dict[str, Any]] = []
    rating_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample_id = str(row.get("sample_id") or f"sample_{index:06d}")
        dataset = str(row.get("dataset") or args.task)
        sample_dir = Path(str(row.get("sample_dir") or output_dir / sample_id))
        if not sample_dir.is_absolute() and not sample_dir.exists():
            sample_dir = output_dir / sample_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(str(row.get("source_image_path") or ""))
        rendered_path = Path(str(row.get("rendered_image_path") or ""))
        artifact_value = str(row.get("generated_artifact_path") or "")
        artifact_path = Path(artifact_value) if artifact_value else Path("")
        render_success = parse_bool(row.get("render_success")) and rendered_path.exists()
        render_status = "ok" if render_success else str(row.get("status") or "render_failed")
        rating_path = sample_dir / f"rating__{slug}.json"
        raw_path = sample_dir / f"rating_response_raw__{slug}.txt"

        if existing_rating_ok(rating_path, args.force_rerate):
            rating = json.loads(rating_path.read_text(encoding="utf-8"))
            rating_status = str(rating.get("status") or "ok")
        elif not render_success:
            if mode == "generic":
                rating = build_generic_render_failure_rating(
                    render_status,
                    dataset_name=dataset,
                    metadata=row,
                    reference_image_path=str(source_path),
                    candidate_image_path=str(rendered_path),
                )
            else:
                rating = build_render_failure_rating(
                    render_status,
                    dataset_name=dataset,
                    metadata=row,
                    reference_image_path=str(source_path),
                    candidate_image_path=str(rendered_path),
                )
            rating_status = "render_failed"
        else:
            candidate_inspection = inspect_candidate_image(rendered_path)
            if mode == "generic":
                user_content = build_generic_rater_user_content(
                    source_image_path=source_path,
                    rendered_image_path=rendered_path,
                    metadata=row,
                    dataset_name=dataset,
                )
                parsed, raw_text = call_and_parse_rating(
                    client,
                    args,
                    system_prompt=GENERIC_RATER_SYSTEM_PROMPT,
                    user_content=user_content,
                    repair_prompt_builder=build_generic_rater_repair_prompt,
                )
                rating = aggregate_generic_rating(
                    parsed,
                    dataset_name=dataset,
                    metadata=row,
                    execution_status="ok",
                    candidate_inspection=candidate_inspection,
                    reference_image_path=str(source_path),
                    candidate_image_path=str(rendered_path),
                )
            else:
                user_content = build_rater_user_content(
                    source_image_path=source_path,
                    rendered_image_path=rendered_path,
                    metadata=row,
                    dataset_name=dataset,
                )
                parsed, raw_text = call_and_parse_rating(
                    client,
                    args,
                    system_prompt=RATER_SYSTEM_PROMPT,
                    user_content=user_content,
                    repair_prompt_builder=lambda raw, err: build_rater_repair_prompt(raw, err, dataset_name=dataset),
                )
                rating = aggregate_rating(
                    parsed,
                    dataset_name=dataset,
                    metadata=row,
                    execution_status="ok",
                    candidate_inspection=candidate_inspection,
                    reference_image_path=str(source_path),
                    candidate_image_path=str(rendered_path),
                )
            raw_path.write_text(raw_text + "\n", encoding="utf-8")
            rating_status = "ok"

        write_json(rating_path, rating)
        if "final_rating_0_to_5" in rating:
            rating_rows.append({"dataset": dataset, "rating": rating})
        out = {
            "task": args.task,
            "sample_id": sample_id,
            "dataset": dataset,
            "question_folder": row.get("question_folder", ""),
            "model_id": args.model_id,
            "source_image_path": str(source_path),
            "rendered_image_path": str(rendered_path) if render_success else "",
            "generated_artifact_path": artifact_value,
            "sample_dir": str(sample_dir),
            "render_success": render_success,
            "render_status": render_status,
            "rating_status": rating_status,
            "rubric_dataset": rating.get("rubric_dataset", ""),
            "rater_provider": args.provider,
            "rater_model": args.rater_model,
            "rater_reasoning_enabled": args.rater_reasoning,
            "rater_slug": slug,
            "rating_path": str(rating_path),
            "rating_final_0_to_5": rating.get("final_rating_0_to_5", ""),
            "rating_raw_score_0_to_5": rating.get("raw_score_0_to_5", ""),
        }
        eval_rows.append(out)
        result_path = sample_dir / "result.json"
        payload = {}
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.update(out)
        payload["latest_tool_ablation_rating"] = rating
        write_json(result_path, payload)
        print(f"[DONE] {index}/{len(rows)} {out['question_folder']} rating_status={rating_status} final={out['rating_final_0_to_5']}", flush=True)

    summary = {
        "task": args.task,
        "render_csv": str(args.render_csv),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "rubric_mode": mode,
        "rater_provider": args.provider,
        "rater_model": args.rater_model,
        "rater_slug": slug,
        "num_rows": len(eval_rows),
        "num_render_success": sum(1 for row in eval_rows if row.get("render_success")),
        **summarize_ratings(rating_rows),
    }
    write_csv(output_dir / f"eval__{slug}.csv", eval_rows, FIELDNAMES)
    write_json(output_dir / f"eval__{slug}.json", {"summary": summary, "rows": eval_rows})
    write_json(output_dir / f"summary__{slug}.json", summary)


if __name__ == "__main__":
    main()
