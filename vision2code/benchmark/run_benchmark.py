from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from vision2code.data.load_kaggle_dataset import locate_data_dir, load_manifest_csv
from vision2code.evaluation.dataset_rubrics import (
    RATER_SYSTEM_PROMPT,
    RUBRIC_VERSION,
    aggregate_rating,
    build_rater_repair_prompt,
    build_rater_user_content,
    build_render_failure_rating,
    inspect_candidate_image,
    parse_json_object,
)
from vision2code.evaluation.generic_rubric import (
    GENERIC_RATER_SYSTEM_PROMPT,
    GENERIC_RUBRIC_VERSION,
    aggregate_generic_rating,
    build_generic_rater_repair_prompt,
    build_generic_rater_user_content,
    build_generic_render_failure_rating,
)
from vision2code.generation.normalize_code import normalize_generated_code
from vision2code.generation.prompts import SYSTEM_PROMPT, USER_PROMPT
from vision2code.rendering.render_python import render_matplotlib_code
from vision2code.utils.env import load_env_file
from vision2code.utils.io import write_csv, write_json


INFERENCE_FIELDNAMES = [
    "benchmark_version",
    "benchmark_split",
    "sample_id",
    "dataset",
    "subset",
    "source_filtered_split",
    "question_folder",
    "question",
    "source_image_path",
    "metadata_path",
    "copied_source_image_path",
    "generated_code_path",
    "rendered_image_path",
    "sample_dir",
    "render_success",
    "status",
    "similarity",
    "generated_code_chars",
]

EVAL_FIELDNAMES = [
    "benchmark_version",
    "benchmark_split",
    "inference_stage",
    "sample_id",
    "dataset",
    "subset",
    "source_filtered_split",
    "question_folder",
    "question",
    "source_image_path",
    "metadata_path",
    "copied_source_image_path",
    "generated_code_path",
    "rendered_image_path",
    "sample_dir",
    "render_success",
    "status",
    "generated_code_chars",
    "rater_provider",
    "rater_model",
    "rater_reasoning_enabled",
    "rater_slug",
    "rating_path",
    "rating_status",
    "rubric_dataset",
    "rating_final_0_to_5",
    "rating_raw_score_0_to_5",
]

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return slug.replace(".", "_").replace("-", "_") or "value"


def split_alias(split: str) -> str:
    cleaned = str(split).strip()
    if cleaned in {"test-mini", "test_mini"}:
        return "test_mini"
    if cleaned == "test":
        return "test"
    raise ValueError("--split must be one of: test_mini, test-mini, test")


def internal_split(split: str) -> str:
    return "test-mini" if split_alias(split) == "test_mini" else "test"


def model_slug(model: str, explicit: str = "") -> str:
    return sanitize_slug(explicit or model)


def rater_slug(provider: str, model: str, reasoning: bool) -> str:
    base = f"{provider}_{sanitize_slug(model).lower()}"
    return f"{base}_reasoning" if reasoning else base


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
            elif hasattr(item, "type") and getattr(item, "type", "") in {"text", "output_text"}:
                parts.append(str(getattr(item, "text", "") or ""))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def normalized_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.asarray(img_a.convert("RGB").resize((512, 512)), dtype=np.float32)
    b = np.asarray(img_b.convert("RGB").resize((512, 512)), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return max(0.0, min(1.0, 1.0 - (mse / (255.0**2))))


def selected_rows(data_dir: Path, split: str, num_samples: int) -> list[dict[str, str]]:
    split_value = split_alias(split)
    rows = [r for r in load_manifest_csv(data_dir) if str(r.get("split") or "") == split_value]
    if num_samples and num_samples > 0:
        rows = rows[:num_samples]
    return rows


def source_image_path(data_dir: Path, row: Mapping[str, Any]) -> Path:
    rel = Path(str(row.get("image_path") or ""))
    path = data_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing source image: {rel}")
    return path


def sample_dir_name(row: Mapping[str, Any], used: set[str]) -> str:
    raw = str(row.get("question_folder") or row.get("source_id") or row.get("source_record_id") or row.get("image_path") or "sample")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "sample"
    name = base
    if name in used:
        suffix = sanitize_slug(str(row.get("source_record_id") or row.get("image_path") or len(used)))
        name = f"{base}__{suffix}"
    counter = 2
    while name in used:
        name = f"{base}__{counter}"
        counter += 1
    used.add(name)
    return name


def copy_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def result_summary_base(
    *,
    benchmark_split: str,
    manifest_rows: int,
    model_kind: str,
    model_path: str,
    output_dir: Path,
    num_samples: int,
) -> dict[str, Any]:
    return {
        "benchmark_split": benchmark_split,
        "model_kind": model_kind,
        "model_path": model_path,
        "output_dir": str(output_dir),
        "num_manifest_rows": manifest_rows,
        "num_samples": num_samples,
    }


def is_retryable_openai_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in RETRYABLE_STATUS_CODES


def generate_openai(client: Any, args: argparse.Namespace, image_path: Path) -> str:
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_completion_tokens": args.max_new_tokens,
        "timeout": args.api_timeout_sec,
        "stream": False,
    }
    if args.reasoning_effort and args.reasoning_effort != "none":
        kwargs["reasoning_effort"] = args.reasoning_effort
    else:
        kwargs["temperature"] = 0

    attempt = 1
    while True:
        try:
            completion = client.chat.completions.create(**kwargs)
            return message_text(completion.choices[0].message.content)
        except (RateLimitError, APITimeoutError, APIError, APIConnectionError) as exc:
            if not is_retryable_openai_error(exc) or attempt >= args.max_retries:
                raise
            sleep_for = min(args.retry_sleep_sec * (2 ** (attempt - 1)), args.retry_max_sleep_sec) + random.uniform(
                0.0, args.retry_jitter_sec
            )
            print(f"[RETRY] sample={image_path.name} attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
            time.sleep(sleep_for)
            attempt += 1


def load_local_model(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError('Install local inference dependencies with: pip install -e ".[train]"') from exc
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


def generate_local(model: Any, processor: Any, args: argparse.Namespace, image: Image.Image) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_PROMPT}]},
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def build_rater_client(args: argparse.Namespace) -> Any:
    if args.rater_provider == "together":
        from together import Together

        return Together(
            api_key=args.rater_api_key,
            base_url=args.rater_base_url or "https://api.together.xyz/v1",
            timeout=args.rater_api_timeout_sec,
            max_retries=0,
        )

    from openai import OpenAI

    base_url = args.rater_base_url or ("http://127.0.0.1:8000/v1" if args.rater_provider == "local_vllm" else "https://api.openai.com/v1")
    return OpenAI(api_key=args.rater_api_key, base_url=base_url, timeout=args.rater_api_timeout_sec, max_retries=0)


def call_rater(
    client: Any,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    content: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    reasoning_enabled: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
        "temperature": temperature,
        "timeout": timeout_sec,
    }
    if provider == "openai":
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if provider == "local_vllm":
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": reasoning_enabled}}
    elif provider == "together":
        kwargs["reasoning"] = {"enabled": reasoning_enabled}
    completion = client.chat.completions.create(**kwargs)
    return message_text(completion.choices[0].message.content)


def is_retryable_rater_exception(exc: BaseException) -> bool:
    if isinstance(exc, ValueError):
        return True
    if getattr(exc, "status_code", None) in RETRYABLE_STATUS_CODES:
        return True
    return type(exc).__name__ in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "APIError"}


def rate_sample(
    *,
    client: Any,
    source_image: Path,
    rendered_image: Path,
    metadata: Mapping[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    candidate_inspection = inspect_candidate_image(rendered_image)
    if args.rubric == "generic":
        system_prompt = GENERIC_RATER_SYSTEM_PROMPT
        content = build_generic_rater_user_content(
            source_image_path=source_image,
            rendered_image_path=rendered_image,
            metadata=metadata,
            dataset_name=dataset_name,
        )
    else:
        system_prompt = RATER_SYSTEM_PROMPT
        content = build_rater_user_content(
            source_image_path=source_image,
            rendered_image_path=rendered_image,
            metadata=metadata,
            dataset_name=dataset_name,
        )

    attempt = 1
    while True:
        try:
            raw_text = call_rater(
                client,
                provider=args.rater_provider,
                model=args.rater_model,
                system_prompt=system_prompt,
                content=content,
                temperature=args.rater_temperature,
                max_tokens=args.rater_max_tokens,
                timeout_sec=args.rater_api_timeout_sec,
                reasoning_enabled=args.rater_reasoning,
            )
            try:
                parsed = parse_json_object(raw_text)
            except Exception as parse_error:
                repaired_text = raw_text
                for _ in range(args.rater_repair_attempts):
                    repair_prompt = (
                        build_generic_rater_repair_prompt(raw_text, str(parse_error))
                        if args.rubric == "generic"
                        else build_rater_repair_prompt(raw_text, str(parse_error), dataset_name=dataset_name)
                    )
                    repaired_text = call_rater(
                        client,
                        provider=args.rater_provider,
                        model=args.rater_model,
                        system_prompt=system_prompt,
                        content=list(content) + [{"type": "text", "text": repair_prompt}],
                        temperature=0.0,
                        max_tokens=args.rater_max_tokens,
                        timeout_sec=args.rater_api_timeout_sec,
                        reasoning_enabled=args.rater_reasoning,
                    )
                    parsed = parse_json_object(repaired_text)
                    raw_text = repaired_text
                    break
                else:
                    raise ValueError(f"Failed to parse rubric response: {parse_error}") from parse_error
            if args.rubric == "generic":
                return (
                    aggregate_generic_rating(
                        parsed,
                        dataset_name=dataset_name,
                        metadata=metadata,
                        execution_status="ok",
                        execution_error_present=False,
                        candidate_inspection=candidate_inspection,
                        reference_image_path=str(source_image),
                        candidate_image_path=str(rendered_image),
                    ),
                    raw_text,
                )
            return (
                aggregate_rating(
                    parsed,
                    dataset_name=dataset_name,
                    metadata=metadata,
                    execution_status="ok",
                    execution_error_present=False,
                    candidate_inspection=candidate_inspection,
                    reference_image_path=str(source_image),
                    candidate_image_path=str(rendered_image),
                ),
                raw_text,
            )
        except Exception as exc:
            if not is_retryable_rater_exception(exc) or attempt >= args.rater_max_retries:
                raise
            sleep_for = args.rater_retry_sleep_sec * (2 ** (attempt - 1)) + random.uniform(0.0, args.rater_retry_jitter_sec)
            print(f"[RATER-RETRY] attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
            time.sleep(sleep_for)
            attempt += 1


def rating_bin(score: float) -> str:
    if score < 1.0:
        return "0-1"
    if score < 2.0:
        return "1-2"
    if score < 3.0:
        return "2-3"
    if score < 4.0:
        return "3-4"
    return "4-5"


def summarize_ratings(rating_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rating_rows:
        return {
            "rated_samples": 0,
            "benchmark_score_0_to_5": 0.0,
            "mean_final_rating_0_to_5": 0.0,
            "mean_raw_score_0_to_5": 0.0,
            "macro_mean_final_rating_0_to_5": 0.0,
            "macro_mean_raw_score_0_to_5": 0.0,
            "final_rating_bins_0_to_5": {key: 0 for key in ("0-1", "1-2", "2-3", "3-4", "4-5")},
            "category_means_0_to_5": {},
            "by_dataset": {},
        }

    final_counter: Counter[str] = Counter()
    final_sum = 0.0
    raw_sum = 0.0
    category_totals: dict[str, float] = defaultdict(float)
    category_counts: dict[str, int] = defaultdict(int)
    dataset_rows: dict[str, list[float]] = defaultdict(list)
    dataset_raw_rows: dict[str, list[float]] = defaultdict(list)
    dataset_category_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    dataset_category_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rating_rows:
        rating = row["rating"]
        dataset = str(row.get("dataset", "unknown"))
        final = float(rating.get("final_rating_0_to_5", 0.0))
        raw = float(rating.get("raw_score_0_to_5", 0.0))
        final_counter[rating_bin(final)] += 1
        final_sum += final
        raw_sum += raw
        dataset_rows[dataset].append(final)
        dataset_raw_rows[dataset].append(raw)
        for category_id, category_payload in rating.get("category_scores", {}).items():
            score = float(category_payload.get("score_0_to_5", 0.0))
            composite_id = f"{dataset}.{category_id}"
            category_totals[composite_id] += score
            category_counts[composite_id] += 1
            dataset_category_totals[dataset][category_id] += score
            dataset_category_counts[dataset][category_id] += 1

    for key in ("0-1", "1-2", "2-3", "3-4", "4-5"):
        final_counter.setdefault(key, 0)

    by_dataset = {
        dataset: {
            "num_samples": len(values),
            "mean_final_rating_0_to_5": round(sum(values) / len(values), 4),
            "mean_raw_score_0_to_5": round(sum(dataset_raw_rows[dataset]) / len(dataset_raw_rows[dataset]), 4),
            "category_means_0_to_5": {
                category_id: round(dataset_category_totals[dataset][category_id] / dataset_category_counts[dataset][category_id], 4)
                for category_id in sorted(dataset_category_totals[dataset])
            },
        }
        for dataset, values in sorted(dataset_rows.items())
    }
    dataset_mean_values = [payload["mean_final_rating_0_to_5"] for payload in by_dataset.values()]
    dataset_raw_mean_values = [payload["mean_raw_score_0_to_5"] for payload in by_dataset.values()]
    macro_final = round(sum(dataset_mean_values) / len(dataset_mean_values), 4) if dataset_mean_values else 0.0
    macro_raw = round(sum(dataset_raw_mean_values) / len(dataset_raw_mean_values), 4) if dataset_raw_mean_values else 0.0

    return {
        "rated_samples": len(rating_rows),
        "benchmark_score_0_to_5": macro_final,
        "mean_final_rating_0_to_5": round(final_sum / len(rating_rows), 4),
        "mean_raw_score_0_to_5": round(raw_sum / len(rating_rows), 4),
        "macro_mean_final_rating_0_to_5": macro_final,
        "macro_mean_raw_score_0_to_5": macro_raw,
        "final_rating_bins_0_to_5": dict(sorted(final_counter.items())),
        "category_means_0_to_5": {
            category_id: round(category_totals[category_id] / category_counts[category_id], 4)
            for category_id in sorted(category_totals)
        },
        "by_dataset": by_dataset,
    }


def load_existing_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def should_reuse_inference(payload: dict[str, Any] | None, rendered_image_path: Path, force: bool) -> bool:
    if force or not payload:
        return False
    if "render_success" not in payload or "status" not in payload:
        return False
    if payload.get("render_success"):
        return rendered_image_path.exists() and rendered_image_path.stat().st_size > 0
    return True


def inference_row_from_payload(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    metadata_path: Path,
    source_copy_path: Path,
    generated_code_path: Path,
    rendered_image_path: Path,
    sample_dir: Path,
) -> dict[str, Any]:
    return {
        "benchmark_version": row.get("benchmark_version", "vision2code_kaggle_v1"),
        "benchmark_split": row.get("benchmark_split") or internal_split(str(row.get("split") or "")),
        "sample_id": row.get("sample_id") or row.get("source_record_id") or row.get("question_folder"),
        "dataset": row.get("source_dataset") or row.get("dataset", ""),
        "subset": row.get("source_subset", ""),
        "source_filtered_split": row.get("source_filtered_split", row.get("split", "")),
        "question_folder": row.get("question_folder", ""),
        "question": row.get("question", ""),
        "source_image_path": str(source_path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "copied_source_image_path": str(source_copy_path) if source_copy_path.exists() else "",
        "generated_code_path": str(generated_code_path) if generated_code_path.exists() else "",
        "rendered_image_path": str(rendered_image_path) if payload.get("render_success") else "",
        "sample_dir": str(sample_dir),
        "render_success": bool(payload.get("render_success")),
        "status": str(payload.get("status") or ""),
        "similarity": float(payload.get("similarity") or 0.0),
        "generated_code_chars": int(payload.get("generated_code_chars") or 0),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Vision2Code benchmark inference and evaluation.")
    ap.add_argument("--provider", choices=["openai", "local"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-slug", default="")
    ap.add_argument("--data_dir", type=Path)
    ap.add_argument("--split", choices=["test_mini", "test-mini", "test"], default="test_mini")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--output_root", type=Path, default=Path("results") / "outputs")
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-rating", action="store_true")
    ap.add_argument("--force-rerate", action="store_true")
    ap.add_argument("--copy-metadata", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--execution-timeout-sec", type=int, default=30)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-timeout-sec", type=float, default=300.0)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--retry-sleep-sec", type=float, default=10.0)
    ap.add_argument("--retry-max-sleep-sec", type=float, default=300.0)
    ap.add_argument("--retry-jitter-sec", type=float, default=5.0)
    ap.add_argument("--rubric", choices=["dataset", "generic"], default="dataset")
    ap.add_argument("--rater-provider", choices=["openai", "local_vllm", "together"], default="local_vllm")
    ap.add_argument("--rater-model", default="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
    ap.add_argument("--rater-base-url", default="")
    ap.add_argument("--rater-api-key-env", default="")
    ap.add_argument("--rater-api-key", default="")
    ap.add_argument("--rater-reasoning", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rater-max-tokens", type=int, default=2200)
    ap.add_argument("--rater-temperature", type=float, default=0.0)
    ap.add_argument("--rater-api-timeout-sec", type=float, default=300.0)
    ap.add_argument("--rater-max-retries", type=int, default=4)
    ap.add_argument("--rater-repair-attempts", type=int, default=1)
    ap.add_argument("--rater-retry-sleep-sec", type=float, default=3.0)
    ap.add_argument("--rater-retry-jitter-sec", type=float, default=1.5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file, override=True)
    data_dir = locate_data_dir(args.data_dir)
    split_value = split_alias(args.split)
    benchmark_split = internal_split(split_value)
    rows = selected_rows(data_dir, split_value, args.num_samples)
    full_rows = selected_rows(data_dir, split_value, 0)
    slug = model_slug(args.model, args.model_slug)
    output_dir = args.output_root / slug / "generations" / "benchmark" / split_value
    max_new_tokens = args.max_new_tokens or (8192 if args.provider == "openai" else 2048)
    args.max_new_tokens = max_new_tokens

    if args.dry_run:
        preview = rows[0] if rows else {}
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "model": args.model,
                    "model_slug": slug,
                    "split": split_value,
                    "benchmark_split": benchmark_split,
                    "data_dir": str(data_dir),
                    "output_dir": str(output_dir),
                    "num_rows": len(rows),
                    "full_split_rows": len(full_rows),
                    "first_question_folder": preview.get("question_folder", ""),
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": USER_PROMPT,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Set OPENAI_API_KEY in the environment or in {args.env_file}.")
    rater_api_key_env = args.rater_api_key_env or ("TOGETHER_API_KEY" if args.rater_provider == "together" else "OPENAI_API_KEY")
    args.rater_api_key = args.rater_api_key or os.getenv(rater_api_key_env) or "EMPTY"
    if args.rater_provider == "openai" and args.rater_api_key == "EMPTY":
        raise RuntimeError(f"Set OPENAI_API_KEY in the environment or in {args.env_file}.")
    if args.rater_provider == "together" and args.rater_api_key == "EMPTY":
        raise RuntimeError(f"Set TOGETHER_API_KEY in the environment or in {args.env_file}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    openai_client = None
    local_model = None
    local_processor = None
    if args.provider == "openai":
        from openai import OpenAI

        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=args.base_url, timeout=args.api_timeout_sec, max_retries=0)
    else:
        local_model, local_processor = load_local_model(args)

    rater_client = None if args.skip_rating else build_rater_client(args)
    rating_slug = rater_slug(args.rater_provider, args.rater_model, args.rater_reasoning)
    rubric_version = GENERIC_RUBRIC_VERSION if args.rubric == "generic" else RUBRIC_VERSION
    used_sample_names: set[str] = set()
    inference_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    rating_rows: list[dict[str, Any]] = []
    renders_ok = 0
    sim_sum = 0.0
    skipped_existing = 0
    run_start = time.monotonic()

    for index, row in enumerate(rows, start=1):
        sample_start = time.monotonic()
        dataset_name = str(row.get("source_dataset") or row.get("dataset") or "")
        question_folder = str(row.get("question_folder") or row.get("source_record_id") or f"sample_{index:06d}")
        print(f"[START] {index}/{len(rows)} dataset={dataset_name} question_folder={question_folder}", flush=True)

        source_path = source_image_path(data_dir, row)
        sample_dir = output_dir / sample_dir_name(row, used_sample_names)
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_copy_path = sample_dir / source_path.name
        metadata_path = sample_dir / "metadata.json"
        generated_code_path = sample_dir / "generated_code.py"
        rendered_image_path = sample_dir / "rendered_image.png"
        error_path = sample_dir / "execution_error.txt"
        result_path = sample_dir / "result.json"
        rating_path = sample_dir / f"rating__{rating_slug}.json"
        rating_raw_path = sample_dir / f"rating_response_raw__{rating_slug}.txt"

        metadata_payload = dict(row)
        existing_payload = load_existing_json(result_path)
        if should_reuse_inference(existing_payload, rendered_image_path, args.force):
            inference_row = inference_row_from_payload(
                row,
                existing_payload or {},
                source_path=source_path,
                metadata_path=metadata_path,
                source_copy_path=source_copy_path,
                generated_code_path=generated_code_path,
                rendered_image_path=rendered_image_path,
                sample_dir=sample_dir,
            )
            skipped_existing += 1
        else:
            copy_or_replace(source_path, source_copy_path)
            if args.copy_metadata:
                write_json(metadata_path, metadata_payload)
            for path in [error_path, rendered_image_path, generated_code_path, result_path, rating_path, rating_raw_path]:
                if path.exists() or path.is_symlink():
                    path.unlink()

            with Image.open(source_path) as source_image:
                source_rgb = source_image.convert("RGB")
                source_size = source_rgb.size
                if args.provider == "openai":
                    assert openai_client is not None
                    raw_code = generate_openai(openai_client, args, source_path)
                else:
                    assert local_model is not None and local_processor is not None
                    raw_code = generate_local(local_model, local_processor, args, source_rgb)
            code = normalize_generated_code(raw_code)
            write_text(generated_code_path, code + "\n")
            render_result = render_matplotlib_code(
                code,
                rendered_image_path,
                timeout_sec=args.execution_timeout_sec,
                target_size=source_size,
            )
            render_success = bool(render_result["render_success"])
            status = str(render_result["status"])
            similarity = 0.0
            if render_success:
                with Image.open(rendered_image_path) as rendered_image, Image.open(source_path) as source_image:
                    similarity = normalized_similarity(source_image.convert("RGB"), rendered_image.convert("RGB"))
            else:
                write_text(error_path, status + "\n")

            inference_row = inference_row_from_payload(
                row,
                {
                    "render_success": render_success,
                    "status": status,
                    "similarity": similarity,
                    "generated_code_chars": len(code),
                },
                source_path=source_path,
                metadata_path=metadata_path,
                source_copy_path=source_copy_path,
                generated_code_path=generated_code_path,
                rendered_image_path=rendered_image_path,
                sample_dir=sample_dir,
            )
            result_payload = dict(inference_row)
            result_payload["sample_metadata"] = metadata_payload
            result_payload["benchmark_row"] = dict(row)
            write_json(result_path, result_payload)

        inference_rows.append(inference_row)
        if inference_row["render_success"]:
            renders_ok += 1
            sim_sum += float(inference_row["similarity"] or 0.0)

        rating_payload: dict[str, Any] | None = None
        rating_status = ""
        if not args.skip_rating:
            existing_rating = load_existing_json(rating_path)
            if (
                existing_rating
                and not args.force_rerate
                and existing_rating.get("rubric_version") == rubric_version
                and "final_rating_0_to_5" in existing_rating
            ):
                rating_payload = existing_rating
                rating_status = str(existing_rating.get("status") or "ok")
            elif inference_row["render_success"]:
                assert rater_client is not None
                try:
                    rating_payload, raw_rating_text = rate_sample(
                        client=rater_client,
                        source_image=source_copy_path if source_copy_path.exists() else source_path,
                        rendered_image=rendered_image_path,
                        metadata=metadata_payload,
                        dataset_name=dataset_name,
                        args=args,
                    )
                    rating_status = "ok"
                    write_text(rating_raw_path, raw_rating_text + "\n")
                except Exception as exc:
                    rating_payload = {"status": "rating_failed", "error": str(exc)}
                    rating_status = "error"
            else:
                if args.rubric == "generic":
                    rating_payload = build_generic_render_failure_rating(
                        str(inference_row["status"]),
                        dataset_name=dataset_name,
                        metadata=metadata_payload,
                        reference_image_path=str(source_copy_path if source_copy_path.exists() else source_path),
                        candidate_image_path=str(rendered_image_path),
                    )
                else:
                    rating_payload = build_render_failure_rating(
                        str(inference_row["status"]),
                        dataset_name=dataset_name,
                        metadata=metadata_payload,
                        reference_image_path=str(source_copy_path if source_copy_path.exists() else source_path),
                        candidate_image_path=str(rendered_image_path),
                    )
                rating_status = "render_failed"

            write_json(rating_path, rating_payload)
            if "final_rating_0_to_5" in rating_payload:
                rating_rows.append({"dataset": dataset_name, "question_folder": question_folder, "rating": rating_payload})

            eval_row = {
                **{field: inference_row.get(field, "") for field in INFERENCE_FIELDNAMES},
                "inference_stage": 1,
                "rater_provider": args.rater_provider,
                "rater_model": args.rater_model,
                "rater_reasoning_enabled": args.rater_reasoning,
                "rater_slug": rating_slug,
                "rating_path": str(rating_path),
                "rating_status": rating_status,
                "rubric_dataset": rating_payload.get("rubric_dataset"),
                "rating_final_0_to_5": rating_payload.get("final_rating_0_to_5"),
                "rating_raw_score_0_to_5": rating_payload.get("raw_score_0_to_5"),
            }
            eval_rows.append(eval_row)
            result_payload = load_existing_json(result_path) or dict(inference_row)
            if not isinstance(result_payload.get("rubric_ratings"), dict):
                result_payload["rubric_ratings"] = {}
            result_payload["rubric_ratings"][rating_slug] = rating_payload
            if not isinstance(result_payload.get("rubric_rating_paths"), dict):
                result_payload["rubric_rating_paths"] = {}
            result_payload["rubric_rating_paths"][rating_slug] = str(rating_path)
            result_payload["latest_rubric_rating_slug"] = rating_slug
            write_json(result_path, result_payload)

        elapsed = time.monotonic() - sample_start
        avg_sec = (time.monotonic() - run_start) / index
        eta_min = (avg_sec * (len(rows) - index)) / 60.0
        print(
            f"[DONE] {index}/{len(rows)} dataset={dataset_name} question_folder={question_folder} "
            f"render_success={inference_row['render_success']} status={str(inference_row['status']).splitlines()[0][:120]} "
            f"rating_status={rating_status} final_rating={(rating_payload or {}).get('final_rating_0_to_5')} "
            f"sample_sec={elapsed:.1f} avg_sec={avg_sec:.1f} eta_min={eta_min:.1f}",
            flush=True,
        )

    inference_summary = result_summary_base(
        benchmark_split=benchmark_split,
        manifest_rows=len(full_rows),
        model_kind=args.provider,
        model_path=args.model,
        output_dir=output_dir,
        num_samples=len(inference_rows),
    )
    inference_summary.update(
        {
            "num_render_success": renders_ok,
            "render_success_rate": (renders_ok / len(inference_rows)) if inference_rows else 0.0,
            "mean_similarity": (sim_sum / renders_ok) if renders_ok else 0.0,
            "max_new_tokens": args.max_new_tokens,
            "reasoning_effort": args.reasoning_effort,
            "base_url": args.base_url if args.provider == "openai" else "",
            "num_skipped_existing_success": skipped_existing,
        }
    )
    write_json(output_dir / "benchmark_inference.json", {"summary": inference_summary, "rows": inference_rows})
    write_csv(output_dir / "benchmark_inference.csv", inference_rows, INFERENCE_FIELDNAMES)

    if not args.skip_rating:
        rubric_summary = summarize_ratings(rating_rows)
        eval_summary = dict(inference_summary)
        eval_summary.update(
            {
                "rubric_version": rubric_version,
                "rater_provider": args.rater_provider,
                "rater_model": args.rater_model,
                "rater_reasoning_enabled": args.rater_reasoning,
                "rater_slug": rating_slug,
                "num_samples_evaluated": len(eval_rows),
                "num_rated": len(rating_rows),
                "benchmark_score_0_to_5": rubric_summary.get("benchmark_score_0_to_5", 0.0),
                "mean_final_rating_0_to_5": rubric_summary.get("mean_final_rating_0_to_5", 0.0),
                "mean_raw_score_0_to_5": rubric_summary.get("mean_raw_score_0_to_5", 0.0),
                "macro_mean_final_rating_0_to_5": rubric_summary.get("macro_mean_final_rating_0_to_5", 0.0),
                "macro_mean_raw_score_0_to_5": rubric_summary.get("macro_mean_raw_score_0_to_5", 0.0),
                "final_rating_bins_0_to_5": rubric_summary.get("final_rating_bins_0_to_5", {}),
                "category_means_0_to_5": rubric_summary.get("category_means_0_to_5", {}),
                "by_dataset": rubric_summary.get("by_dataset", {}),
            }
        )
        eval_json = output_dir / f"benchmark_eval__{rating_slug}.json"
        eval_csv = output_dir / f"benchmark_eval__{rating_slug}.csv"
        summary_json = output_dir / f"benchmark_summary__{rating_slug}.json"
        write_json(eval_json, {"summary": eval_summary, "rows": eval_rows})
        write_csv(eval_csv, eval_rows, EVAL_FIELDNAMES)
        write_json(summary_json, eval_summary)
        print(f"[DONE] wrote: {eval_json}", flush=True)
        print(f"[DONE] wrote: {eval_csv}", flush=True)
        print(f"[DONE] wrote: {summary_json}", flush=True)

    print(f"[DONE] wrote: {output_dir / 'benchmark_inference.json'}", flush=True)
    print(f"[DONE] wrote: {output_dir / 'benchmark_inference.csv'}", flush=True)


if __name__ == "__main__":
    main()
