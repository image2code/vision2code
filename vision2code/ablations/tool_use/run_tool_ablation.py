from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from vision2code.utils.env import load_env_file
from vision2code.utils.io import read_jsonl, write_csv, write_json

TASK_TO_ARTIFACT = {
    "latex_docvqa": "generated.tex",
    "excalidraw_json": "scene.excalidraw.json",
}

FIELDNAMES = [
    "task",
    "sample_id",
    "dataset",
    "subset",
    "source_split",
    "question_folder",
    "question",
    "model",
    "source_image_path",
    "copied_source_image_path",
    "generated_artifact_path",
    "sample_dir",
    "status",
    "generated_chars",
]

LATEX_SYSTEM_PROMPT = """\
You write LaTeX source that recreates document images.
Return only a complete standalone .tex file. Do not use markdown fences.
Do not include external images or network resources. Do not use shell escape.
"""

LATEX_USER_PROMPT = """\
Recreate the provided document page as a single standalone LaTeX file.

Requirements:
- Output LaTeX source only.
- Use standard packages such as geometry, xcolor, graphicx, array, tabularx, tikz, textpos, fontenc, inputenc, and lmodern if useful.
- Do not reference external files, URLs, or images.
- Match the visible page structure, text, tables, forms, spacing, and approximate typography as closely as possible.
- Preserve readable visible text. If tiny text is unreadable, approximate layout and use plausible short placeholders.
- Size the page/aspect ratio to match the image dimensions when possible.
- The question below is context only; recreate the full document page, not just an answer.

Question/context:
{question}
"""

EXCALIDRAW_SYSTEM_PROMPT = """\
You write strict Excalidraw scene JSON for diagram recreation.
Return one valid JSON object only. Do not use markdown fences, comments, HTML, SVG, Mermaid, canvas code, or prose.
"""

EXCALIDRAW_USER_PROMPT = """\
Recreate the provided Excalidraw-style image as an Excalidraw scene JSON object.

Requirements:
- Output JSON only.
- Use a top-level object compatible with Excalidraw, including an "elements" array.
- Use only these Excalidraw element.type values: rectangle, diamond, ellipse, arrow, line, freedraw, text, frame.
- Do not use non-Excalidraw or unsupported element types such as roundRect, path, triangle, circle, polygon, group, icon, label, image, embeddable, html, svg, or mermaid.
- For rounded boxes, use type "rectangle" with a "roundness" field such as {{"type": 3}}; never use "roundRect".
- For triangles, brackets, braces, icons, and irregular shapes, approximate them with supported line, arrow, freedraw, rectangle, diamond, ellipse, and text elements.
- Use standard Excalidraw element fields: id, type, x, y, width, height, angle, strokeColor, backgroundColor, fillStyle, strokeWidth, strokeStyle, roughness, opacity, points, text, fontSize, fontFamily, textAlign, verticalAlign, seed, version, versionNonce, isDeleted.
- Include appState with viewBackgroundColor when useful.
- Do not reference external images, URLs, or files. Do not include image elements or files data.
- Approximate hand-drawn shapes, arrows, connectors, labels, layout, and colors.
- Keep coordinates within an image-sized canvas close to {width}x{height}.
- The JSON must be complete and parseable; every object and array must be closed.
"""

EXCALIDRAW_ALLOWED_ELEMENT_TYPES = {"rectangle", "diamond", "ellipse", "arrow", "line", "freedraw", "text", "frame"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI generation for tool-use ablations.")
    parser.add_argument("--task", choices=sorted(TASK_TO_ARTIFACT), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-sleep-sec", type=float, default=10.0)
    parser.add_argument("--retry-max-sleep-sec", type=float, default=300.0)
    parser.add_argument("--retry-jitter-sec", type=float, default=5.0)
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def strip_model_fences(text: str, language_hint: str = "") -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> str:
    stripped = strip_model_fences(text)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    return stripped[start : end + 1]


def validate_excalidraw_scene_json(scene_text: str) -> str:
    payload = json.loads(scene_text)
    if not isinstance(payload, dict):
        raise ValueError("Excalidraw output must be a JSON object.")
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Excalidraw output must contain an elements array.")
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise ValueError(f"Excalidraw element {index} must be an object.")
        element_type = element.get("type")
        if element_type not in EXCALIDRAW_ALLOWED_ELEMENT_TYPES:
            raise ValueError(f"Unsupported Excalidraw element type at index {index}: {element_type!r}")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def prompts_for_sample(task: str, sample: Mapping[str, Any], image_path: Path) -> tuple[str, str]:
    if task == "latex_docvqa":
        return LATEX_SYSTEM_PROMPT, LATEX_USER_PROMPT.format(question=sample.get("question", ""))
    if task == "excalidraw_json":
        width = int(sample.get("width") or 1024)
        height = int(sample.get("height") or 768)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            pass
        return EXCALIDRAW_SYSTEM_PROMPT, EXCALIDRAW_USER_PROMPT.format(width=width, height=height)
    raise ValueError(f"Unsupported task: {task}")


def source_image_path(sample: Mapping[str, Any], manifest: Path, data_dir: Path | None) -> Path:
    raw = str(sample.get("image_path") or sample.get("source_image_path") or "")
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in [data_dir, manifest.parent]:
        if base is None:
            continue
        candidate = base / path
        if candidate.exists():
            return candidate
    return manifest.parent / path


def sample_dir_name(sample: Mapping[str, Any], index: int) -> str:
    raw = str(sample.get("question_folder") or sample.get("sample_id") or f"sample_{index:06d}")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or f"sample_{index:06d}"


def completion_text(completion: Any) -> str:
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    return str(content or "").strip()


def generate(client: Any, args: argparse.Namespace, sample: Mapping[str, Any], image_path: Path) -> str:
    system_prompt, user_prompt = prompts_for_sample(args.task, sample, image_path)
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    {"type": "text", "text": user_prompt},
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
    if args.task == "excalidraw_json":
        kwargs["response_format"] = {"type": "json_object"}
    attempt = 1
    while True:
        try:
            return completion_text(client.chat.completions.create(**kwargs))
        except Exception as exc:
            if attempt >= args.max_retries:
                raise
            sleep_for = min(args.retry_sleep_sec * (2 ** (attempt - 1)), args.retry_max_sleep_sec) + random.uniform(0.0, args.retry_jitter_sec)
            print(f"[RETRY] sample={sample.get('sample_id')} attempt={attempt} sleep={sleep_for:.1f}s err={type(exc).__name__}", flush=True)
            time.sleep(sleep_for)
            attempt += 1


def normalize_artifact(task: str, raw_text: str) -> str:
    if task == "latex_docvqa":
        return strip_model_fences(raw_text, "latex").strip() + "\n"
    if task == "excalidraw_json":
        return validate_excalidraw_scene_json(extract_json_object(raw_text))
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file, override=True)
    samples = read_jsonl(args.manifest)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    if args.dry_run:
        print(json.dumps({"task": args.task, "manifest": str(args.manifest), "samples": len(samples), "output_dir": str(args.output_dir)}, indent=2))
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Set OPENAI_API_KEY in the environment or in {args.env_file}.")
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=args.base_url, timeout=args.api_timeout_sec, max_retries=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        sample_dir = args.output_dir / sample_dir_name(sample, index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = source_image_path(sample, args.manifest, args.data_dir)
        copied_image = sample_dir / image_path.name
        artifact_path = sample_dir / TASK_TO_ARTIFACT[args.task]
        raw_path = sample_dir / "raw_response.txt"
        if artifact_path.exists() and not args.force:
            status = "cached"
        else:
            shutil.copy2(image_path, copied_image)
            raw_text = generate(client, args, sample, image_path)
            raw_path.write_text(raw_text + "\n", encoding="utf-8")
            artifact_path.write_text(normalize_artifact(args.task, raw_text), encoding="utf-8")
            status = "ok"
        row = {
            "task": args.task,
            "sample_id": sample.get("sample_id") or sample.get("source_record_id") or sample.get("question_folder") or index,
            "dataset": sample.get("dataset") or sample.get("source_dataset") or "",
            "subset": sample.get("subset") or sample.get("source_subset") or "",
            "source_split": sample.get("source_split") or sample.get("split") or "",
            "question_folder": sample.get("question_folder") or sample_dir.name,
            "question": sample.get("question", ""),
            "model": args.model,
            "source_image_path": str(image_path),
            "copied_source_image_path": str(copied_image),
            "generated_artifact_path": str(artifact_path),
            "sample_dir": str(sample_dir),
            "status": status,
            "generated_chars": artifact_path.stat().st_size if artifact_path.exists() else 0,
        }
        rows.append(row)
        write_json(sample_dir / "result.json", row)
        print(f"[DONE] {index}/{len(samples)} {row['question_folder']} status={status}", flush=True)
    write_csv(args.output_dir / f"inference_{args.task}.csv", rows, FIELDNAMES)
    write_json(args.output_dir / f"inference_{args.task}.json", {"rows": rows, "summary": {"task": args.task, "num_rows": len(rows)}})


if __name__ == "__main__":
    main()

