from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageStat

ALLOWED_ELEMENT_TYPES = {
    "rectangle",
    "diamond",
    "ellipse",
    "arrow",
    "line",
    "freedraw",
    "text",
    "frame",
}


def _inspect_image(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"readable": False, "width": 0, "height": 0, "is_blank": True}
    try:
        with Image.open(path) as image:
            image.load()
            gray = image.convert("L")
            stat = ImageStat.Stat(gray)
            extrema = gray.getextrema()
            return {
                "readable": True,
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
                "is_blank": bool(extrema and extrema[0] == extrema[1]),
                "mean_luma": float(stat.mean[0]) if stat.mean else 0.0,
            }
    except Exception as exc:
        return {"readable": False, "width": 0, "height": 0, "is_blank": True, "error": str(exc)}


def default_renderer_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "ablations" / "tool_use" / "vendor" / "excalidraw_renderer"


def validate_scene_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Excalidraw scene must be a JSON object.")
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Excalidraw scene must contain an elements array.")
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise ValueError(f"Excalidraw element at index {index} must be an object.")
        element_type = str(element.get("type") or "")
        if element_type not in ALLOWED_ELEMENT_TYPES:
            allowed = ", ".join(sorted(ALLOWED_ELEMENT_TYPES))
            raise ValueError(f"Unsupported Excalidraw element type {element_type!r} at index {index}; allowed: {allowed}")
    files = payload.get("files", {})
    if files and not isinstance(files, dict):
        raise ValueError("Excalidraw files field must be an object when present.")
    if files:
        raise ValueError("Excalidraw tool-use ablation scenes must not include embedded files or image data.")
    return payload


def validate_scene(scene_path: Path) -> dict[str, Any]:
    return validate_scene_payload(json.loads(scene_path.read_text(encoding="utf-8")))


def scaled_render_size(source_width: int, source_height: int, max_width: int, max_height: int) -> tuple[int, int, float]:
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    scale = min(max_width / source_width, max_height / source_height, 1.0)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale)), scale


def scale_scene_payload(scene: dict[str, Any], scale: float) -> dict[str, Any]:
    if abs(scale - 1.0) < 1e-9:
        return scene
    scaled = copy.deepcopy(scene)
    for element in scaled.get("elements", []):
        if not isinstance(element, dict):
            continue
        for key in ("x", "y", "width", "height"):
            if isinstance(element.get(key), (int, float)):
                element[key] = element[key] * scale
        for key in ("fontSize", "strokeWidth"):
            if isinstance(element.get(key), (int, float)):
                element[key] = element[key] * scale
        points = element.get("points")
        if isinstance(points, list):
            for point in points:
                if isinstance(point, list) and len(point) >= 2:
                    if isinstance(point[0], (int, float)):
                        point[0] = point[0] * scale
                    if isinstance(point[1], (int, float)):
                        point[1] = point[1] * scale
    return scaled


def _write_scene_for_render(scene_path: Path, scene: dict[str, Any], scale: float) -> Path:
    if abs(scale - 1.0) < 1e-9:
        return scene_path
    scaled_path = scene_path.with_name(".render_scaled_scene.excalidraw.json")
    scaled_path.write_text(
        json.dumps(scale_scene_payload(scene, scale), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return scaled_path


def _source_size(source_image: Path | None, width: int | None, height: int | None) -> tuple[int, int]:
    if source_image and source_image.exists():
        try:
            with Image.open(source_image) as image:
                return int(image.width), int(image.height)
        except Exception:
            pass
    return int(width or 1024), int(height or 768)


def _render_with_chrome(
    scene_path: Path,
    output_png: Path,
    renderer_dir: Path,
    width: int,
    height: int,
    *,
    timeout_sec: int,
    chrome_bin: str = "",
) -> tuple[bool, str]:
    html_path = renderer_dir / "render_scene.html"
    bundle_path = renderer_dir / "renderer_bundle.js"
    if not html_path.exists():
        return False, f"missing_renderer_html: {html_path}"
    if not bundle_path.exists():
        return False, f"missing_renderer_bundle: {bundle_path}"
    chrome = chrome_bin or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        return False, "chrome_not_found"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_png.exists():
        output_png.unlink()
    scene_url = scene_path.resolve().as_uri()
    page_url = f"{html_path.resolve().as_uri()}?scene={quote(scene_url, safe=':/')}&width={width}&height={height}"
    cmd = [
        chrome,
        "--headless",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-crash-reporter",
        "--disable-crashpad",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--allow-file-access-from-files",
        f"--window-size={width},{height}",
        f"--screenshot={output_png}",
        "--virtual-time-budget=10000",
        page_url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return False, "chrome_timeout"
    if proc.returncode != 0:
        return False, f"chrome_failed: {(proc.stderr or proc.stdout or '').strip()[:1200]}"
    inspection = _inspect_image(output_png)
    if not inspection.get("readable") or inspection.get("is_blank"):
        return False, f"rendered_png_invalid: {inspection}"
    return True, "ok"


def render_excalidraw(
    scene_json: str | Path,
    output_png: str | Path,
    *,
    renderer_dir: str | Path | None = None,
    timeout_sec: int = 60,
    source_image: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    max_render_width: int = 2400,
    max_render_height: int = 1800,
    chrome_bin: str = "",
) -> dict[str, object]:
    output_path = Path(output_png)
    scene_text = str(scene_json)
    looks_like_path = isinstance(scene_json, Path) or ("\n" not in scene_text and len(scene_text) < 240)
    if looks_like_path and Path(scene_text).exists():
        scene_path = Path(scene_json)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path = output_path.with_name("scene.excalidraw.json")
        scene_path.write_text(str(scene_json), encoding="utf-8")
    if not scene_path.exists() or scene_path.stat().st_size <= 0:
        return {"render_success": False, "status": "missing_scene_json", "output_path": ""}

    renderer = Path(renderer_dir) if renderer_dir else default_renderer_dir()
    source_path = Path(source_image) if source_image else None
    try:
        scene = validate_scene(scene_path)
        source_width, source_height = _source_size(source_path, width, height)
        render_width, render_height, scale = scaled_render_size(
            source_width,
            source_height,
            max_render_width,
            max_render_height,
        )
        render_scene_path = _write_scene_for_render(scene_path, scene, scale)
        success, status = _render_with_chrome(
            render_scene_path,
            output_path,
            renderer,
            render_width,
            render_height,
            timeout_sec=timeout_sec,
            chrome_bin=chrome_bin,
        )
        return {
            "render_success": success,
            "status": status,
            "output_path": str(output_path) if success else "",
            "renderer_dir": str(renderer),
            "render_width": render_width,
            "render_height": render_height,
            "render_scale": scale,
        }
    except Exception as exc:
        return {"render_success": False, "status": f"excalidraw_render_failed: {type(exc).__name__}: {exc}", "output_path": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an Excalidraw scene JSON file to PNG.")
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-dir", type=Path, default=None)
    parser.add_argument("--source-image", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-render-width", type=int, default=2400)
    parser.add_argument("--max-render-height", type=int, default=1800)
    parser.add_argument("--chrome-bin", default="")
    args = parser.parse_args()
    print(
        render_excalidraw(
            args.scene_path,
            args.output,
            renderer_dir=args.renderer_dir,
            source_image=args.source_image,
            timeout_sec=args.timeout_sec,
            max_render_width=args.max_render_width,
            max_render_height=args.max_render_height,
            chrome_bin=args.chrome_bin,
        )
    )


if __name__ == "__main__":
    main()
