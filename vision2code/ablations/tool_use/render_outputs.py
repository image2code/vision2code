from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from vision2code.rendering.render_excalidraw import render_excalidraw
from vision2code.rendering.render_latex import render_latex
from vision2code.utils.io import read_csv, write_csv, write_json

TASK_TO_ARTIFACT = {
    "latex_docvqa": "generated.tex",
    "excalidraw_json": "scene.excalidraw.json",
}

FIELDNAMES = [
    "task",
    "sample_id",
    "dataset",
    "question_folder",
    "source_image_path",
    "generated_artifact_path",
    "rendered_pdf_path",
    "rendered_image_path",
    "sample_dir",
    "render_success",
    "status",
    "similarity",
    "renderer",
    "render_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LaTeX or Excalidraw tool-use ablation outputs.")
    parser.add_argument("--task", choices=sorted(TASK_TO_ARTIFACT), required=True)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory containing per-sample folders. Defaults to inference CSV parent.")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--latex-engine", choices=["auto", "latexmk", "pdflatex"], default="auto")
    parser.add_argument("--renderer-dir", type=Path, default=None)
    parser.add_argument("--chrome-bin", default="")
    parser.add_argument("--max-render-width", type=int, default=2400)
    parser.add_argument("--max-render-height", type=int, default=1800)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: str, *, data_dir: Path | None, base_dir: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    if data_dir is not None:
        candidate = data_dir / path
        if candidate.exists():
            return candidate
    return base_dir / path


def normalized_similarity(path_a: Path, path_b: Path) -> float:
    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        a = np.asarray(image_a.convert("RGB").resize((512, 512)), dtype=np.float32)
        b = np.asarray(image_b.convert("RGB").resize((512, 512)), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return max(0.0, min(1.0, 1.0 - (mse / (255.0**2))))


def sample_dir_for_row(row: Mapping[str, Any], output_dir: Path) -> Path:
    value = str(row.get("sample_dir") or "")
    if value:
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        return output_dir / path
    name = str(row.get("question_folder") or row.get("sample_id") or "sample").strip()
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._") or "sample"
    return output_dir / safe


def main() -> None:
    args = parse_args()
    rows = read_csv(args.inference_csv)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    output_dir = args.output_dir or args.inference_csv.parent

    if args.dry_run:
        preview = {}
        if rows:
            sample_dir = sample_dir_for_row(rows[0], output_dir)
            preview = {
                "sample_dir": str(sample_dir),
                "artifact": str(sample_dir / TASK_TO_ARTIFACT[args.task]),
                "rendered_image": str(sample_dir / "rendered_image.png"),
            }
        print(json.dumps({"rows": len(rows), "preview": preview}, indent=2))
        return

    render_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        start = time.monotonic()
        sample_dir = sample_dir_for_row(row, output_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        artifact_value = str(row.get("generated_artifact_path") or "")
        artifact_path = Path(artifact_value) if artifact_value else sample_dir / TASK_TO_ARTIFACT[args.task]
        if artifact_value and not artifact_path.is_absolute() and not artifact_path.exists():
            candidate = args.inference_csv.parent / artifact_path
            artifact_path = candidate if candidate.exists() else sample_dir / TASK_TO_ARTIFACT[args.task]
        rendered_image = sample_dir / "rendered_image.png"
        rendered_pdf = sample_dir / "rendered.pdf"
        source_image = resolve_path(str(row.get("source_image_path") or ""), data_dir=args.data_dir, base_dir=args.inference_csv.parent)

        if rendered_image.exists() and rendered_image.stat().st_size > 0 and not args.force:
            result = {"render_success": True, "status": "existing_render", "output_path": str(rendered_image)}
        elif args.task == "latex_docvqa":
            result = render_latex(
                artifact_path,
                rendered_image,
                output_pdf=rendered_pdf,
                timeout_sec=args.timeout_sec,
                dpi=args.dpi,
                engine=args.latex_engine,
            )
        else:
            result = render_excalidraw(
                artifact_path,
                rendered_image,
                renderer_dir=args.renderer_dir,
                source_image=source_image if source_image.exists() else None,
                timeout_sec=args.timeout_sec,
                max_render_width=args.max_render_width,
                max_render_height=args.max_render_height,
                chrome_bin=args.chrome_bin,
            )

        render_success = bool(result.get("render_success"))
        similarity = 0.0
        if render_success and source_image.exists() and rendered_image.exists():
            try:
                similarity = normalized_similarity(source_image, rendered_image)
            except Exception:
                similarity = 0.0

        out = {
            "task": args.task,
            "sample_id": row.get("sample_id", ""),
            "dataset": row.get("dataset", ""),
            "question_folder": row.get("question_folder", ""),
            "source_image_path": str(source_image),
            "generated_artifact_path": str(artifact_path) if artifact_path.exists() else "",
            "rendered_pdf_path": str(rendered_pdf) if rendered_pdf.exists() else "",
            "rendered_image_path": str(rendered_image) if render_success else "",
            "sample_dir": str(sample_dir),
            "render_success": render_success,
            "status": result.get("status", ""),
            "similarity": similarity,
            "renderer": "latex" if args.task == "latex_docvqa" else "official_excalidraw_chrome",
            "render_seconds": round(time.monotonic() - start, 3),
        }
        render_rows.append(out)
        result_path = sample_dir / "result.json"
        payload = {}
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.update(out)
        payload["render_result"] = dict(result)
        write_json(result_path, payload)
        print(f"[DONE] {index}/{len(rows)} {out['question_folder']} render_success={render_success} status={out['status']}", flush=True)

    prefix = "render_latex" if args.task == "latex_docvqa" else "render_excalidraw"
    write_csv(output_dir / f"{prefix}.csv", render_rows, FIELDNAMES)
    write_json(
        output_dir / f"{prefix}.json",
        {
            "summary": {
                "task": args.task,
                "num_rows": len(render_rows),
                "num_render_success": sum(1 for row in render_rows if row.get("render_success")),
                "render_success_rate": (sum(1 for row in render_rows if row.get("render_success")) / len(render_rows)) if render_rows else 0.0,
            },
            "rows": render_rows,
        },
    )


if __name__ == "__main__":
    main()
