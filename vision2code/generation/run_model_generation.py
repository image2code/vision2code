from __future__ import annotations

import argparse
import base64
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from vision2code.data.load_kaggle_dataset import load_manifest_csv
from vision2code.generation.normalize_code import normalize_generated_code
from vision2code.generation.prompts import SYSTEM_PROMPT, USER_PROMPT
from vision2code.utils.env import load_env_file
from vision2code.utils.io import write_json, write_jsonl


PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "local": "HF_TOKEN",
}


def _safe_id(row: dict[str, str], index: int) -> str:
    raw = row.get("question_folder") or row.get("source_id") or row.get("source_record_id") or f"sample_{index:06d}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._")
    return f"{index:06d}_{safe or 'sample'}"


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


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


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504, 529}


def _openai_generate(
    *,
    client: Any,
    model: str,
    image_path: Path,
    max_tokens: int,
    timeout_sec: float,
    reasoning_effort: str,
    max_retries: int,
    retry_sleep_sec: float,
    retry_max_sleep_sec: float,
    retry_jitter_sec: float,
) -> str:
    try:
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
    except ImportError as exc:
        raise RuntimeError('Install the OpenAI client with: pip install -e ".[eval]"') from exc

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_completion_tokens": max_tokens,
        "timeout": timeout_sec,
        "stream": False,
    }
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0

    attempt = 1
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            return _message_text(response.choices[0].message.content)
        except (RateLimitError, APITimeoutError, APIError, APIConnectionError) as exc:
            if not _is_retryable_openai_error(exc) or attempt >= max_retries:
                raise
            sleep_for = min(retry_sleep_sec * (2 ** (attempt - 1)), retry_max_sleep_sec) + random.uniform(0.0, retry_jitter_sec)
            print(f"[RETRY] sample={image_path.name} model={model} attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
            time.sleep(sleep_for)
            attempt += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PROVIDER_ENV), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--split", default="test_mini")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--api-timeout-sec", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--retry-sleep-sec", type=float, default=10.0)
    ap.add_argument("--retry-max-sleep-sec", type=float, default=300.0)
    ap.add_argument("--retry-jitter-sec", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env_file(args.env_file, override=True)
    rows = [r for r in load_manifest_csv(args.data_dir) if str(r.get("split")) == args.split]
    if args.num_samples:
        rows = rows[: args.num_samples]

    if args.dry_run:
        print({"provider": args.provider, "model": args.model, "rows": len(rows), "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT})
        return

    env = PROVIDER_ENV[args.provider]
    if args.provider != "local" and not os.getenv(env):
        raise RuntimeError(f"Set {env} in the environment or in {args.env_file}.")
    if args.provider != "openai":
        raise SystemExit(f"{args.provider} generation is configured but not implemented in this lightweight release script.")

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=args.base_url, timeout=args.api_timeout_sec, max_retries=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_dir = args.output_dir / "code"
    raw_dir = args.output_dir / "raw"
    code_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for index, row in enumerate(rows):
        sample_id = _safe_id(row, index)
        image_rel = Path(str(row.get("image_path") or ""))
        image_path = args.data_dir / image_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Missing source image for row {index}: {image_rel}")

        raw_text = _openai_generate(
            client=client,
            model=args.model,
            image_path=image_path,
            max_tokens=args.max_output_tokens,
            timeout_sec=args.api_timeout_sec,
            reasoning_effort=args.reasoning_effort.strip(),
            max_retries=args.max_retries,
            retry_sleep_sec=args.retry_sleep_sec,
            retry_max_sleep_sec=args.retry_max_sleep_sec,
            retry_jitter_sec=args.retry_jitter_sec,
        )
        code = normalize_generated_code(raw_text)

        code_path = code_dir / f"{sample_id}.py"
        raw_path = raw_dir / f"{sample_id}.txt"
        code_path.write_text(code + "\n", encoding="utf-8")
        raw_path.write_text(raw_text, encoding="utf-8")

        output_rows.append(
            {
                "sample_id": sample_id,
                "row_index": index,
                "split": row.get("split", ""),
                "benchmark_split": row.get("benchmark_split", ""),
                "source_dataset": row.get("source_dataset", row.get("dataset", "")),
                "question_folder": row.get("question_folder", ""),
                "image_path": image_rel.as_posix(),
                "provider": args.provider,
                "model": args.model,
                "code_path": code_path.relative_to(args.output_dir).as_posix(),
                "raw_response_path": raw_path.relative_to(args.output_dir).as_posix(),
            }
        )

    manifest_path = args.output_dir / "generations.jsonl"
    write_jsonl(manifest_path, output_rows)
    write_json(
        args.output_dir / "generation_summary.json",
        {
            "provider": args.provider,
            "model": args.model,
            "rows": len(output_rows),
            "manifest": manifest_path.name,
            "base_url": args.base_url,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
