from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

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
from vision2code.utils.env import load_env_file
from vision2code.utils.io import read_jsonl, write_csv, write_json, write_jsonl


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts).strip()
    return str(content or "")


def _call_rater_completion(
    client: Any,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
    reasoning_enabled: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
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
    response = client.chat.completions.create(**kwargs)
    return _message_text(response.choices[0].message.content)


def _is_retryable_rater_exception(exc: BaseException) -> bool:
    if isinstance(exc, ValueError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
        return True
    name = type(exc).__name__
    return name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "APIError"}


def _build_client(provider: str, base_url: str, api_key: str, timeout_sec: float) -> Any:
    if provider == "together":
        try:
            from together import Together
        except ImportError as exc:
            raise RuntimeError("Install Together client to use --provider together.") from exc
        return Together(api_key=api_key, base_url=base_url or None, timeout=timeout_sec, max_retries=0)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Install the OpenAI client with: pip install -e ".[eval]"') from exc
    return OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout_sec, max_retries=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--provider", choices=["openai", "local_vllm", "together"], default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rubric", choices=["dataset", "generic"], default="dataset")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--rater-base-url", default="")
    ap.add_argument("--rater-api-key-env", default="")
    ap.add_argument("--rater-api-key", default="")
    ap.add_argument("--rater-reasoning", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rater-temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=2200)
    ap.add_argument("--rater-api-timeout-sec", type=float, default=300.0)
    ap.add_argument("--rater-max-retries", type=int, default=4)
    ap.add_argument("--rater-repair-attempts", type=int, default=1)
    ap.add_argument("--rater-retry-sleep-sec", type=float, default=3.0)
    ap.add_argument("--rater-retry-jitter-sec", type=float, default=1.5)
    args = ap.parse_args()

    load_env_file(args.env_file, override=True)
    default_base_url = {"openai": "https://api.openai.com/v1", "local_vllm": "http://127.0.0.1:8000/v1", "together": "https://api.together.xyz/v1"}[args.provider]
    api_key_env = args.rater_api_key_env or ("TOGETHER_API_KEY" if args.provider == "together" else "OPENAI_API_KEY")
    api_key = args.rater_api_key or os.getenv(api_key_env) or "EMPTY"
    if args.provider in {"openai", "together"} and api_key == "EMPTY":
        raise RuntimeError(f"Set {api_key_env} in the environment or in {args.env_file}.")
    client = _build_client(args.provider, args.rater_base_url or default_base_url, api_key, args.rater_api_timeout_sec)

    rows = read_jsonl(args.renders)
    if args.num_samples:
        rows = rows[: args.num_samples]

    render_root = args.renders.parent
    raw_dir = args.output.parent / "rating_raw"
    rating_dir = args.output.parent / "rating_json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rating_dir.mkdir(parents=True, exist_ok=True)

    rating_rows = []
    jsonl_rows = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"sample_{index:06d}")
        dataset_name = str(row.get("source_dataset") or row.get("dataset") or "")
        source_image = args.data_dir / str(row.get("image_path") or "")
        render_success = str(row.get("render_success")).lower() == "true" or row.get("render_success") is True
        candidate_image = _resolve_relative(render_root, str(row.get("rendered_image_path") or "")) if render_success else Path("")
        execution_status = "ok" if render_success else str(row.get("render_status") or "render_failed")

        if not render_success:
            if args.rubric == "generic":
                rating = build_generic_render_failure_rating(execution_status, dataset_name=dataset_name, metadata=row, reference_image_path=str(source_image), candidate_image_path="")
            else:
                rating = build_render_failure_rating(execution_status, dataset_name=dataset_name, metadata=row, reference_image_path=str(source_image), candidate_image_path="")
            raw_response = ""
        else:
            candidate_inspection = inspect_candidate_image(candidate_image)
            if args.rubric == "generic":
                system_prompt = GENERIC_RATER_SYSTEM_PROMPT
                user_content = build_generic_rater_user_content(source_image_path=source_image, rendered_image_path=candidate_image, metadata=row, dataset_name=dataset_name)
            else:
                system_prompt = RATER_SYSTEM_PROMPT
                user_content = build_rater_user_content(source_image_path=source_image, rendered_image_path=candidate_image, metadata=row, dataset_name=dataset_name)
            attempt = 1
            while True:
                try:
                    raw_response = _call_rater_completion(
                        client,
                        provider=args.provider,
                        model=args.model,
                        system_prompt=system_prompt,
                        user_content=user_content,
                        max_tokens=args.max_output_tokens,
                        temperature=args.rater_temperature,
                        timeout_sec=args.rater_api_timeout_sec,
                        reasoning_enabled=args.rater_reasoning,
                    )
                    try:
                        parsed = parse_json_object(raw_response)
                    except Exception as parse_error:
                        repaired_response = raw_response
                        for _ in range(args.rater_repair_attempts):
                            repair_prompt = (
                                build_generic_rater_repair_prompt(raw_response, str(parse_error))
                                if args.rubric == "generic"
                                else build_rater_repair_prompt(raw_response, str(parse_error), dataset_name=dataset_name)
                            )
                            repaired_response = _call_rater_completion(
                                client,
                                provider=args.provider,
                                model=args.model,
                                system_prompt=system_prompt,
                                user_content=list(user_content) + [{"type": "text", "text": repair_prompt}],
                                max_tokens=args.max_output_tokens,
                                temperature=0.0,
                                timeout_sec=args.rater_api_timeout_sec,
                                reasoning_enabled=args.rater_reasoning,
                            )
                            parsed = parse_json_object(repaired_response)
                            raw_response = repaired_response
                            break
                        else:
                            raise ValueError(f"Failed to parse rubric response: {parse_error}") from parse_error
                    break
                except Exception as exc:
                    if not _is_retryable_rater_exception(exc) or attempt >= args.rater_max_retries:
                        raise
                    sleep_for = args.rater_retry_sleep_sec * (2 ** (attempt - 1)) + random.uniform(0.0, args.rater_retry_jitter_sec)
                    print(f"[RATER-RETRY] attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
                    time.sleep(sleep_for)
                    attempt += 1
            if args.rubric == "generic":
                rating = aggregate_generic_rating(
                    parsed,
                    dataset_name=dataset_name,
                    metadata=row,
                    execution_status="ok",
                    candidate_inspection=candidate_inspection,
                    reference_image_path=str(source_image),
                    candidate_image_path=str(candidate_image),
                )
            else:
                rating = aggregate_rating(
                    parsed,
                    dataset_name=dataset_name,
                    metadata=row,
                    execution_status="ok",
                    candidate_inspection=candidate_inspection,
                    reference_image_path=str(source_image),
                    candidate_image_path=str(candidate_image),
                )

        raw_path = raw_dir / f"{sample_id}.txt"
        rating_path = rating_dir / f"{sample_id}.json"
        raw_path.write_text(raw_response, encoding="utf-8")
        write_json(rating_path, rating)

        flat = {
            "sample_id": sample_id,
            "row_index": row.get("row_index", index),
            "split": row.get("split", ""),
            "source_dataset": dataset_name,
            "image_path": row.get("image_path", ""),
            "model": row.get("model", ""),
            "rater_model": args.model,
            "rubric": args.rubric,
            "render_success": render_success,
            "execution_status": rating.get("execution_status", execution_status),
            "raw_score_0_to_5": rating.get("raw_score_0_to_5", ""),
            "final_rating_0_to_5": rating.get("final_rating_0_to_5", ""),
            "rating_json_path": rating_path.relative_to(args.output.parent).as_posix(),
            "raw_response_path": raw_path.relative_to(args.output.parent).as_posix(),
        }
        rating_rows.append(flat)
        jsonl_rows.append({**flat, "rating": rating})

    write_csv(args.output, rating_rows)
    write_jsonl(args.output.with_suffix(".jsonl"), jsonl_rows)
    print(args.output)


if __name__ == "__main__":
    main()
