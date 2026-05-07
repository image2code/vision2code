from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from vision2code.benchmark.run_benchmark import (
    image_data_url,
    internal_split,
    load_local_model,
    message_text,
    normalized_similarity,
    sample_dir_name,
    selected_rows,
    source_image_path,
    split_alias,
)
from vision2code.data.load_kaggle_dataset import locate_data_dir
from vision2code.generation.normalize_code import normalize_generated_code
from vision2code.generation.prompts import SYSTEM_PROMPT, USER_PROMPT
from vision2code.rendering.render_python import render_matplotlib_code
from vision2code.utils.env import load_env_file
from vision2code.utils.io import write_csv, write_json

TTS_VERSION = "benchmark_tts_v1"
MAX_TTS_STAGE = 4
REFINEMENT_SYSTEM_PROMPT = SYSTEM_PROMPT
REFINEMENT_PROMPT_TEMPLATE = """You are improving Python code that recreates a reference image.

The first image is the reference image to recreate.
The second image is the previous rendered output.

Previous Python code:
```python
{previous_code}
```

Rewrite the Python code so the rendered output better matches the reference image.
Save the image only to the Python variable OUTPUT_PATH.
Do not quote OUTPUT_PATH and do not append a file extension to it.
Return code only."""

FIELDNAMES = [
    "tts_version",
    "stage",
    "benchmark_split",
    "sample_id",
    "dataset",
    "subset",
    "source_filtered_split",
    "question_folder",
    "question",
    "source_image_path",
    "previous_stage",
    "previous_generated_code_path",
    "previous_rendered_image_path",
    "generated_code_path",
    "rendered_image_path",
    "execution_error_path",
    "result_path",
    "sample_dir",
    "render_success",
    "status",
    "similarity",
    "generated_code_chars",
    "model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-time scaling generation and rendering.")
    parser.add_argument("--backend", choices=["openai", "local"], default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--split", choices=["test_mini", "test-mini", "test"], default="test_mini")
    parser.add_argument("--stage", type=int, action="append", default=None)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--execution-timeout-sec", type=int, default=30)
    parser.add_argument("--api-timeout-sec", type=float, default=300.0)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-sleep-sec", type=float, default=10.0)
    parser.add_argument("--retry-max-sleep-sec", type=float, default=300.0)
    parser.add_argument("--retry-jitter-sec", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stages(raw: list[int] | None) -> list[int]:
    out = sorted(set(raw or [2]))
    if any(stage < 1 or stage > MAX_TTS_STAGE for stage in out):
        raise ValueError(f"Stages must be in [1, {MAX_TTS_STAGE}]")
    return out


def generate_openai(client: Any, args: argparse.Namespace, *, source_image: Path, previous_image: Path | None, previous_code: str | None) -> str:
    if previous_image is None:
        system_prompt = SYSTEM_PROMPT
        text_prompt = USER_PROMPT
        content = [{"type": "image_url", "image_url": {"url": image_data_url(source_image)}}, {"type": "text", "text": text_prompt}]
    else:
        system_prompt = REFINEMENT_SYSTEM_PROMPT
        text_prompt = REFINEMENT_PROMPT_TEMPLATE.format(previous_code=previous_code or "")
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(source_image)}},
            {"type": "image_url", "image_url": {"url": image_data_url(previous_image)}},
            {"type": "text", "text": text_prompt},
        ]
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
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
        except Exception as exc:
            if attempt >= args.max_retries:
                raise
            sleep_for = min(args.retry_sleep_sec * (2 ** (attempt - 1)), args.retry_max_sleep_sec) + random.uniform(0.0, args.retry_jitter_sec)
            print(f"[RETRY] attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
            time.sleep(sleep_for)
            attempt += 1


def generate_local_tts(
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    *,
    source_image: Image.Image,
    previous_image: Image.Image | None,
    previous_code: str | None,
) -> str:
    if previous_image is None:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": source_image}, {"type": "text", "text": USER_PROMPT}]},
        ]
        images = [source_image]
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": REFINEMENT_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": previous_image},
                    {"type": "text", "text": REFINEMENT_PROMPT_TEMPLATE.format(previous_code=previous_code or "")},
                ],
            },
        ]
        images = [source_image, previous_image]
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt_text], images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def existing_ok(result_path: Path, rendered_path: Path, force: bool) -> bool:
    if force or not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("render_success"):
        return rendered_path.exists() and rendered_path.stat().st_size > 0
    return "status" in payload


def row_base(row: Mapping[str, Any], stage: int, sample_dir: Path, source_path: Path) -> dict[str, Any]:
    return {
        "tts_version": TTS_VERSION,
        "stage": stage,
        "benchmark_split": row.get("benchmark_split") or internal_split(str(row.get("split") or "")),
        "sample_id": row.get("sample_id") or row.get("source_record_id") or row.get("question_folder"),
        "dataset": row.get("source_dataset") or row.get("dataset", ""),
        "subset": row.get("source_subset", ""),
        "source_filtered_split": row.get("source_filtered_split", row.get("split", "")),
        "question_folder": row.get("question_folder", ""),
        "question": row.get("question", ""),
        "source_image_path": str(source_path),
        "sample_dir": str(sample_dir),
    }


def stage_artifact_paths(sample_dir: Path, stage: int) -> dict[str, Path]:
    if stage == 1:
        return {
            "code": sample_dir / "generated_code.py",
            "rendered": sample_dir / "rendered_image.png",
            "result": sample_dir / "result.json",
            "error": sample_dir / "execution_error.txt",
        }
    prefix = f"stage{stage}"
    return {
        "code": sample_dir / f"{prefix}_generated_code.py",
        "rendered": sample_dir / f"{prefix}_rendered_image.png",
        "result": sample_dir / f"{prefix}_result.json",
        "error": sample_dir / f"{prefix}_execution_error.txt",
    }


def previous_stage_artifact_paths(sample_dir: Path, stage: int) -> dict[str, Path]:
    paths = stage_artifact_paths(sample_dir, stage - 1)
    legacy = {
        "code": sample_dir / f"tts_stage{stage - 1}_code.py",
        "rendered": sample_dir / f"tts_stage{stage - 1}_rendered.png",
    }
    for key, legacy_path in legacy.items():
        if not paths[key].exists() and legacy_path.exists():
            paths[key] = legacy_path
    return paths


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file, override=True)
    data_dir = locate_data_dir(args.data_dir)
    split_value = split_alias(args.split)
    rows = selected_rows(data_dir, split_value, args.num_samples)
    stage_values = stages(args.stage)
    if args.dry_run:
        print(json.dumps({"split": split_value, "num_rows": len(rows), "stages": stage_values, "output_dir": str(args.output_dir)}, indent=2))
        return
    client = None
    local_model = None
    local_processor = None
    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Set OPENAI_API_KEY in the environment or in {args.env_file}.")
    if args.backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=args.base_url, timeout=args.api_timeout_sec, max_retries=0)
    else:
        local_model, local_processor = load_local_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    sample_dirs: dict[str, Path] = {}
    for row in rows:
        sample_dirs[str(row.get("question_folder") or row.get("source_record_id"))] = args.output_dir / sample_dir_name(row, used_names)

    for stage in stage_values:
        out_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            key = str(row.get("question_folder") or row.get("source_record_id"))
            sample_dir = sample_dirs[key]
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_path = source_image_path(data_dir, row)
            with Image.open(source_path) as image:
                target_size = image.convert("RGB").size
            current_paths = stage_artifact_paths(sample_dir, stage)
            code_path = current_paths["code"]
            rendered_path = current_paths["rendered"]
            result_path = current_paths["result"]
            error_path = current_paths["error"]
            previous_paths = previous_stage_artifact_paths(sample_dir, stage) if stage > 1 else {}
            previous_code_path = previous_paths.get("code")
            previous_render_path = previous_paths.get("rendered")
            if existing_ok(result_path, rendered_path, args.force):
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                out_rows.append(payload)
                continue
            if stage > 1 and (
                previous_code_path is None
                or previous_render_path is None
                or not previous_code_path.exists()
                or not previous_render_path.exists()
            ):
                payload = row_base(row, stage, sample_dir, source_path)
                payload.update(
                    {
                        "previous_stage": stage - 1,
                        "previous_generated_code_path": str(previous_code_path or ""),
                        "previous_rendered_image_path": str(previous_render_path or ""),
                        "generated_code_path": "",
                        "rendered_image_path": "",
                        "execution_error_path": "",
                        "result_path": str(result_path),
                        "render_success": False,
                        "status": "missing_previous_stage",
                        "similarity": 0.0,
                        "generated_code_chars": 0,
                        "model": args.model,
                    }
                )
                write_json(result_path, payload)
                out_rows.append(payload)
                continue
            previous_code = previous_code_path.read_text(encoding="utf-8") if stage > 1 and previous_code_path else None
            if args.backend == "openai":
                raw_code = generate_openai(
                    client,
                    args,
                    source_image=source_path,
                    previous_image=previous_render_path if stage > 1 else None,
                    previous_code=previous_code,
                )
            else:
                assert local_model is not None and local_processor is not None
                with Image.open(source_path) as source_image_handle:
                    source_image_rgb = source_image_handle.convert("RGB")
                previous_image_rgb = None
                if stage > 1 and previous_render_path is not None:
                    with Image.open(previous_render_path) as previous_image_handle:
                        previous_image_rgb = previous_image_handle.convert("RGB")
                raw_code = generate_local_tts(
                    local_model,
                    local_processor,
                    args,
                    source_image=source_image_rgb,
                    previous_image=previous_image_rgb,
                    previous_code=previous_code,
                )
            code = normalize_generated_code(raw_code)
            code_path.write_text(code + "\n", encoding="utf-8")
            render_result = render_matplotlib_code(code, rendered_path, timeout_sec=args.execution_timeout_sec, target_size=target_size)
            similarity = 0.0
            if render_result["render_success"]:
                with Image.open(source_path) as source_image, Image.open(rendered_path) as rendered_image:
                    similarity = normalized_similarity(source_image.convert("RGB"), rendered_image.convert("RGB"))
            else:
                error_path.write_text(str(render_result["status"]) + "\n", encoding="utf-8")
            payload = row_base(row, stage, sample_dir, source_path)
            payload.update(
                {
                    "previous_stage": stage - 1 if stage > 1 else "",
                    "previous_generated_code_path": str(previous_code_path or "") if stage > 1 else "",
                    "previous_rendered_image_path": str(previous_render_path or "") if stage > 1 else "",
                    "generated_code_path": str(code_path),
                    "rendered_image_path": str(rendered_path) if render_result["render_success"] else "",
                    "execution_error_path": str(error_path) if not render_result["render_success"] else "",
                    "result_path": str(result_path),
                    "render_success": bool(render_result["render_success"]),
                    "status": str(render_result["status"]),
                    "similarity": similarity,
                    "generated_code_chars": len(code),
                    "model": args.model,
                }
            )
            write_json(result_path, payload)
            out_rows.append(payload)
            print(f"[DONE] stage={stage} {index}/{len(rows)} {payload['question_folder']} status={payload['status']}", flush=True)
        write_csv(args.output_dir / f"benchmark_tts_stage{stage}.csv", out_rows, FIELDNAMES)
        write_json(args.output_dir / f"benchmark_tts_stage{stage}.json", {"rows": out_rows, "summary": {"stage": stage, "num_rows": len(out_rows)}})


if __name__ == "__main__":
    main()
