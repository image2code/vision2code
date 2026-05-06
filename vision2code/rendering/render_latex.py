from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], *, cwd: Path, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


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


def _latexmk_command(tex_name: str) -> list[str]:
    binary = _which("latexmk")
    if not binary:
        raise RuntimeError("latexmk not found in PATH")
    return [
        binary,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-pdflatex=pdflatex %O -interaction=nonstopmode -halt-on-error -no-shell-escape %S",
        tex_name,
    ]


def _pdflatex_command(tex_name: str) -> list[str]:
    binary = _which("pdflatex")
    if not binary:
        raise RuntimeError("pdflatex not found in PATH")
    return [binary, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", tex_name]


def _compile_tex(
    tex_path: Path,
    output_pdf: Path,
    log_path: Path,
    *,
    engine: str,
    timeout_sec: int,
) -> tuple[bool, str, str]:
    attempts: list[tuple[str, list[str]]] = []
    if engine in {"auto", "latexmk"}:
        try:
            attempts.append(("latexmk", _latexmk_command("generated.tex")))
        except RuntimeError:
            if engine == "latexmk":
                raise
    if engine in {"auto", "pdflatex"}:
        attempts.append(("pdflatex", _pdflatex_command("generated.tex")))
    if not attempts:
        return False, "latex_engine_not_found", engine

    logs: list[str] = []
    last_engine = engine
    with tempfile.TemporaryDirectory(prefix="vision2code_latex_") as td:
        work_dir = Path(td)
        shutil.copy2(tex_path, work_dir / "generated.tex")
        for engine_name, cmd in attempts:
            last_engine = engine_name
            try:
                proc = _run(cmd, cwd=work_dir, timeout_sec=timeout_sec)
            except subprocess.TimeoutExpired:
                logs.append(f"===== {engine_name} timeout after {timeout_sec}s =====\n")
                continue
            logs.append(
                f"===== command: {' '.join(cmd)} =====\n"
                f"returncode={proc.returncode}\n"
                f"----- stdout -----\n{proc.stdout}\n"
                f"----- stderr -----\n{proc.stderr}\n"
            )
            if engine_name == "pdflatex" and proc.returncode == 0:
                second = _run(cmd, cwd=work_dir, timeout_sec=timeout_sec)
                logs.append(
                    "===== pdflatex second pass =====\n"
                    f"returncode={second.returncode}\n"
                    f"----- stdout -----\n{second.stdout}\n"
                    f"----- stderr -----\n{second.stderr}\n"
                )
                proc = second
            generated_pdf = work_dir / "generated.pdf"
            if proc.returncode == 0 and generated_pdf.exists() and generated_pdf.stat().st_size > 0:
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(generated_pdf, output_pdf)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("\n".join(logs), encoding="utf-8")
                return True, "ok", engine_name

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(logs), encoding="utf-8")
    return False, "latex_compile_failed", last_engine


def _pdf_to_png(pdf_path: Path, output_png: Path, *, dpi: int, timeout_sec: int) -> tuple[bool, str]:
    binary = _which("pdftoppm")
    if not binary:
        return False, "pdftoppm_not_found"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_png.exists():
        output_png.unlink()
    prefix = output_png.with_suffix("")
    cmd = [binary, "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(prefix)]
    try:
        proc = _run(cmd, cwd=output_png.parent, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, "pdf_to_png_timeout"
    if proc.returncode != 0:
        return False, f"pdf_to_png_failed: {(proc.stderr or proc.stdout or '').strip()[:1000]}"
    inspection = _inspect_image(output_png)
    if not inspection.get("readable") or inspection.get("is_blank"):
        return False, f"rendered_png_invalid: {inspection}"
    return True, "ok"


def render_latex(
    tex_source: str | Path,
    output_png: str | Path,
    *,
    timeout_sec: int = 60,
    dpi: int = 144,
    engine: str = "auto",
    output_pdf: str | Path | None = None,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Compile a standalone LaTeX document and rasterize the first page to PNG."""
    if engine not in {"auto", "latexmk", "pdflatex"}:
        raise ValueError("engine must be one of: auto, latexmk, pdflatex")
    output_png_path = Path(output_png)
    output_pdf_path = Path(output_pdf) if output_pdf else output_png_path.with_suffix(".pdf")
    compile_log_path = Path(log_path) if log_path else output_png_path.with_name("latex_compile.log")

    source_text = str(tex_source)
    looks_like_path = isinstance(tex_source, Path) or ("\n" not in source_text and len(source_text) < 240)
    if looks_like_path and Path(source_text).exists():
        tex_path = Path(tex_source)
    else:
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path = output_png_path.with_name("generated.tex")
        tex_path.write_text(str(tex_source), encoding="utf-8")

    if not tex_path.exists() or tex_path.stat().st_size <= 0:
        return {"render_success": False, "status": "missing_generated_tex", "output_path": ""}

    try:
        compiled, status, compile_engine = _compile_tex(
            tex_path,
            output_pdf_path,
            compile_log_path,
            engine=engine,
            timeout_sec=timeout_sec,
        )
        if not compiled:
            return {
                "render_success": False,
                "status": status,
                "output_path": "",
                "rendered_pdf_path": "",
                "compile_engine": compile_engine,
                "compile_log_path": str(compile_log_path),
            }
        rendered, status = _pdf_to_png(output_pdf_path, output_png_path, dpi=dpi, timeout_sec=timeout_sec)
        return {
            "render_success": rendered,
            "status": status,
            "output_path": str(output_png_path) if rendered else "",
            "rendered_pdf_path": str(output_pdf_path),
            "compile_engine": compile_engine,
            "compile_log_path": str(compile_log_path),
        }
    except subprocess.TimeoutExpired:
        return {"render_success": False, "status": "timeout", "output_path": ""}
    except Exception as exc:
        return {"render_success": False, "status": f"latex_render_failed: {type(exc).__name__}: {exc}", "output_path": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a standalone LaTeX file to PNG.")
    parser.add_argument("tex_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--engine", choices=["auto", "latexmk", "pdflatex"], default="auto")
    args = parser.parse_args()
    print(
        render_latex(
            args.tex_path,
            args.output,
            output_pdf=args.output_pdf,
            timeout_sec=args.timeout_sec,
            dpi=args.dpi,
            engine=args.engine,
        )
    )


if __name__ == "__main__":
    main()
