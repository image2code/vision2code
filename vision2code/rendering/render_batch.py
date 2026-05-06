from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import numpy as np

from vision2code.rendering.render_python import render_matplotlib_code
from vision2code.utils.io import read_jsonl, write_csv, write_jsonl


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def normalized_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.asarray(img_a.convert("RGB").resize((512, 512)), dtype=np.float32)
    b = np.asarray(img_b.convert("RGB").resize((512, 512)), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return max(0.0, min(1.0, 1.0 - (mse / (255.0 ** 2))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path)
    ap.add_argument("--data_dir", type=Path)
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=30)
    args = ap.parse_args()

    rows = read_jsonl(args.generations)
    if args.num_samples:
        rows = rows[: args.num_samples]

    generation_root = args.generations.parent
    output_dir = args.output_dir or generation_root
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"sample_{index:06d}")
        code_path = _resolve_relative(generation_root, str(row.get("code_path") or ""))
        rendered_path = render_dir / f"{sample_id}.png"
        target_size = (512, 512)
        source_path = None
        if args.data_dir and row.get("image_path"):
            source_path = args.data_dir / str(row["image_path"])
            if source_path.exists():
                with Image.open(source_path) as image:
                    target_size = image.size
        result = render_matplotlib_code(code_path.read_text(encoding="utf-8"), rendered_path, timeout_sec=args.timeout_sec, target_size=target_size)
        rendered_value = rendered_path.relative_to(output_dir).as_posix() if result["render_success"] else ""
        similarity = 0.0
        if result["render_success"] and source_path and source_path.exists():
            with Image.open(source_path) as source_image, Image.open(rendered_path) as rendered_image:
                similarity = normalized_similarity(source_image, rendered_image)
        output_rows.append(
            {
                **row,
                "render_success": bool(result["render_success"]),
                "render_status": str(result["status"]),
                "rendered_image_path": rendered_value,
                "similarity": similarity,
            }
        )

    jsonl_path = output_dir / "render_manifest.jsonl"
    csv_path = output_dir / "render_manifest.csv"
    write_jsonl(jsonl_path, output_rows)
    write_csv(csv_path, output_rows)
    print(jsonl_path)


if __name__ == "__main__":
    main()
