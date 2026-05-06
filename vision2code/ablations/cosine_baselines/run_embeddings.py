from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from vision2code.metrics.embedding_similarity import cosine_similarity, scale_cosine_to_0_to_5
from vision2code.metrics.focus_texts import FOCUS_TEXT_STRATEGY, focus_text_for_dataset
from vision2code.utils.io import read_csv, write_csv, write_json

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
EMBEDDING_INSTRUCTION = "Represent the input for comparing the visual fidelity of a recreated figure to its source."

SAMPLE_FIELDNAMES = [
    "benchmark_split",
    "dataset",
    "subset",
    "sample_id",
    "question_folder",
    "model_slug",
    "source_image_path",
    "generated_image_path",
    "render_available",
    "render_success",
    "focus_text_strategy",
    "focus_text",
    "embedding_model",
    "embedding_instruction",
    "cosine_image_only",
    "cosine_image_only_0_to_5",
    "cosine_image_plus_text",
    "cosine_image_plus_text_0_to_5",
    "status",
]

SUMMARY_FIELDNAMES = [
    "benchmark_split",
    "model_slug",
    "dataset",
    "total_rows",
    "scored_rows",
    "missing_render_rows",
    "missing_sample_dir_rows",
    "embed_error_rows",
    "mean_cosine_image_only",
    "mean_cosine_image_only_0_to_5",
    "mean_cosine_image_only_0_to_5_all_rows",
    "mean_cosine_image_plus_text",
    "mean_cosine_image_plus_text_0_to_5",
    "mean_cosine_image_plus_text_0_to_5_all_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image-only and image+text embedding-similarity baselines.")
    parser.add_argument("--provider", choices=["pixel", "qwen_vllm"], default="pixel")
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-instruction", default=EMBEDDING_INSTRUCTION)
    parser.add_argument("--model-slug", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: str, data_dir: Path | None) -> Path:
    text = str(value or "")
    if text.startswith("<VISION2CODE_DATA_DIR>/"):
        if data_dir is None:
            raise RuntimeError("CSV uses <VISION2CODE_DATA_DIR>; pass --data_dir.")
        return data_dir / text.removeprefix("<VISION2CODE_DATA_DIR>/")
    path = Path(text)
    return path if path.is_absolute() else Path.cwd() / path


def image_vector(path: Path, size: int = 32) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError(f"Invalid image vector for {path}")
    return arr / norm


def text_vector(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def combined_image_text_vector(path: Path, focus_text: str) -> np.ndarray:
    return np.concatenate([image_vector(path), text_vector(focus_text)])


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Encountered zero-norm embedding vector")
    return vector / norm


class QwenVllmEmbedder:
    """Qwen3-VL embedding runner using vLLM's pooling/embed API."""

    def __init__(self, *, model_id: str, dtype: str, max_model_len: int, instruction: str, batch_size: int) -> None:
        try:
            from vllm import EngineArgs, LLM
        except ImportError as exc:
            raise RuntimeError("Install vLLM in a GPU environment to use --provider qwen_vllm.") from exc

        self.instruction = instruction
        self.batch_size = batch_size
        try:
            engine_args = EngineArgs(
                model=model_id,
                runner="pooling",
                dtype=dtype,
                trust_remote_code=True,
                max_model_len=max_model_len,
            )
        except TypeError:
            engine_args = EngineArgs(
                model=model_id,
                task="embed",
                dtype=dtype,
                trust_remote_code=True,
                max_model_len=max_model_len,
            )
        self.llm = LLM(**vars(engine_args))

    def _prepare_input(self, image_path: Path, text: str) -> dict[str, Any]:
        content: list[dict[str, str]] = [{"type": "image", "image": "file://" + str(image_path.resolve())}]
        if text:
            content.append({"type": "text", "text": text})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {"role": "user", "content": content},
        ]
        prompt_text = self.llm.llm_engine.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(image_path) as handle:
            image_obj = handle.convert("RGB")
        return {"prompt": prompt_text, "multi_modal_data": {"image": image_obj}}

    def _extract_embedding(self, output: Any) -> np.ndarray:
        if hasattr(output, "outputs") and hasattr(output.outputs, "embedding"):
            values = output.outputs.embedding
        elif hasattr(output, "embedding"):
            values = output.embedding
        elif isinstance(output, dict) and "embedding" in output:
            values = output["embedding"]
        else:
            raise RuntimeError(f"Unable to extract embedding from vLLM output: {type(output)!r}")
        return l2_normalize(np.asarray(values, dtype=np.float32))

    def embed(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        results: list[Any] = [None] * len(items)

        def run_indexes(indexes: Sequence[int]) -> None:
            inputs = [self._prepare_input(Path(str(items[idx]["image_path"])), str(items[idx].get("text") or "")) for idx in indexes]
            outputs = self.llm.embed(inputs)
            if len(outputs) != len(indexes):
                raise RuntimeError(f"Expected {len(indexes)} embeddings, got {len(outputs)}")
            for idx, output in zip(indexes, outputs):
                results[idx] = self._extract_embedding(output)

        def run_chunk(indexes: Sequence[int]) -> None:
            try:
                run_indexes(indexes)
            except Exception as exc:
                if len(indexes) == 1:
                    results[indexes[0]] = exc
                    return
                midpoint = len(indexes) // 2
                run_chunk(indexes[:midpoint])
                run_chunk(indexes[midpoint:])

        for start in range(0, len(items), self.batch_size):
            run_chunk(list(range(start, min(start + self.batch_size, len(items)))))
        return results


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize(rows: list[Mapping[str, Any]], include_dataset: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("benchmark_split") or ""), str(row.get("model_slug") or ""))
        if include_dataset:
            key += (str(row.get("dataset") or ""),)
        groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        scored = [row for row in group if row.get("status") == "scored"]
        image_only = [value for value in (parse_float(row.get("cosine_image_only")) for row in scored) if value is not None]
        image_only_scaled = [value for value in (parse_float(row.get("cosine_image_only_0_to_5")) for row in scored) if value is not None]
        image_text = [value for value in (parse_float(row.get("cosine_image_plus_text")) for row in scored) if value is not None]
        image_text_scaled = [value for value in (parse_float(row.get("cosine_image_plus_text_0_to_5")) for row in scored) if value is not None]
        total = len(group) or 1
        summaries.append(
            {
                "benchmark_split": key[0],
                "model_slug": key[1],
                "dataset": key[2] if include_dataset else "",
                "total_rows": len(group),
                "scored_rows": len(scored),
                "missing_render_rows": sum(1 for row in group if row.get("status") == "missing_render"),
                "missing_sample_dir_rows": sum(1 for row in group if row.get("status") == "missing_sample_dir"),
                "embed_error_rows": sum(1 for row in group if row.get("status") == "embed_error"),
                "mean_cosine_image_only": mean(image_only),
                "mean_cosine_image_only_0_to_5": mean(image_only_scaled),
                "mean_cosine_image_only_0_to_5_all_rows": sum(float(row.get("cosine_image_only_0_to_5") or 0.0) for row in group) / total,
                "mean_cosine_image_plus_text": mean(image_text),
                "mean_cosine_image_plus_text_0_to_5": mean(image_text_scaled),
                "mean_cosine_image_plus_text_0_to_5_all_rows": sum(float(row.get("cosine_image_plus_text_0_to_5") or 0.0) for row in group) / total,
            }
        )
    return summaries


def inferred_model_slug(inference_csv: Path, explicit_slug: str) -> str:
    if explicit_slug:
        return explicit_slug
    parts = inference_csv.parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def row_paths(row: Mapping[str, Any], data_dir: Path | None) -> tuple[Path, Path]:
    source = resolve_path(str(row.get("source_image_path") or row.get("reference_image_path") or ""), data_dir)
    rendered = resolve_path(str(row.get("rendered_image_path") or row.get("generated_image_path") or ""), data_dir)
    return source, rendered


def cache_key(model: str, image_path: Path, text: str) -> str:
    payload = f"{model}\n{image_path.resolve()}\n{text}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def cache_path(cache_dir: Path, model: str, image_path: Path, text: str) -> Path:
    return cache_dir / f"{cache_key(model, image_path, text)}.npy"


def qwen_embedding(
    embedder: QwenVllmEmbedder,
    *,
    cache_dir: Path,
    model: str,
    image_path: Path,
    text: str,
    force: bool,
) -> np.ndarray:
    path = cache_path(cache_dir, model, image_path, text)
    if path.exists() and not force:
        return l2_normalize(np.load(path))
    result = embedder.embed([{"image_path": str(image_path), "text": text}])[0]
    if isinstance(result, Exception):
        raise result
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    return result


def main() -> None:
    args = parse_args()
    rows = read_csv(args.inference_csv)
    if args.dry_run:
        print({"rows": len(rows), "provider": args.provider, "output_dir": str(args.output_dir)})
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = inferred_model_slug(args.inference_csv, args.model_slug)
    cache_dir = args.cache_dir or (args.output_dir / "_embedding_cache")
    embedder: QwenVllmEmbedder | None = None
    if args.provider == "qwen_vllm":
        embedder = QwenVllmEmbedder(
            model_id=args.embedding_model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            instruction=args.embedding_instruction,
            batch_size=args.batch_size,
        )

    sample_rows: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row.get("dataset") or row.get("source_dataset") or "")
        question_folder = str(row.get("question_folder") or "")
        focus_text = focus_text_for_dataset(dataset)
        out = {
            "benchmark_split": row.get("benchmark_split", row.get("split", "")),
            "dataset": dataset,
            "subset": row.get("subset", ""),
            "sample_id": row.get("sample_id", ""),
            "question_folder": question_folder,
            "model_slug": model_slug,
            "source_image_path": row.get("source_image_path", ""),
            "generated_image_path": row.get("rendered_image_path", row.get("generated_image_path", "")),
            "render_available": False,
            "render_success": parse_bool(row.get("render_success")),
            "focus_text_strategy": FOCUS_TEXT_STRATEGY,
            "focus_text": focus_text,
            "embedding_model": args.embedding_model if args.provider == "qwen_vllm" else "pixel_downsample_rgb_32",
            "embedding_instruction": args.embedding_instruction,
            "cosine_image_only": "",
            "cosine_image_only_0_to_5": "",
            "cosine_image_plus_text": "",
            "cosine_image_plus_text_0_to_5": "",
            "status": "",
        }
        try:
            source, rendered = row_paths(row, args.data_dir)
            if not rendered.exists() or (row.get("render_success") not in ("", None) and not parse_bool(row.get("render_success"))):
                out["status"] = "missing_render"
            elif not source.exists():
                out["status"] = "missing_source_image"
            else:
                out["render_available"] = True
                if args.provider == "pixel":
                    source_image_only = image_vector(source)
                    rendered_image_only = image_vector(rendered)
                    source_image_text = combined_image_text_vector(source, focus_text)
                    rendered_image_text = combined_image_text_vector(rendered, focus_text)
                else:
                    assert embedder is not None
                    source_image_only = qwen_embedding(
                        embedder,
                        cache_dir=cache_dir,
                        model=args.embedding_model,
                        image_path=source,
                        text="",
                        force=args.force,
                    )
                    rendered_image_only = qwen_embedding(
                        embedder,
                        cache_dir=cache_dir,
                        model=args.embedding_model,
                        image_path=rendered,
                        text="",
                        force=args.force,
                    )
                    source_image_text = qwen_embedding(
                        embedder,
                        cache_dir=cache_dir,
                        model=args.embedding_model,
                        image_path=source,
                        text=focus_text,
                        force=args.force,
                    )
                    rendered_image_text = qwen_embedding(
                        embedder,
                        cache_dir=cache_dir,
                        model=args.embedding_model,
                        image_path=rendered,
                        text=focus_text,
                        force=args.force,
                    )
                image_only = cosine_similarity(source_image_only, rendered_image_only)
                image_text = cosine_similarity(source_image_text, rendered_image_text)
                out.update(
                    {
                        "cosine_image_only": image_only,
                        "cosine_image_only_0_to_5": scale_cosine_to_0_to_5(image_only),
                        "cosine_image_plus_text": image_text,
                        "cosine_image_plus_text_0_to_5": scale_cosine_to_0_to_5(image_text),
                        "status": "scored",
                    }
                )
        except Exception as exc:
            out["status"] = f"embed_error: {type(exc).__name__}: {exc}"
        sample_rows.append(out)

    write_csv(args.output_dir / "benchmark_embedding_similarity.csv", sample_rows, SAMPLE_FIELDNAMES)
    write_csv(args.output_dir / "benchmark_embedding_similarity_by_model.csv", summarize(sample_rows, include_dataset=False), SUMMARY_FIELDNAMES)
    write_csv(args.output_dir / "benchmark_embedding_similarity_by_model_dataset.csv", summarize(sample_rows, include_dataset=True), SUMMARY_FIELDNAMES)
    write_json(
        args.output_dir / "summary.json",
        {
            "provider": args.provider,
            "embedding_model": args.embedding_model if args.provider == "qwen_vllm" else "pixel_downsample_rgb_32",
            "rows": len(sample_rows),
            "scored_rows": sum(1 for row in sample_rows if row.get("status") == "scored"),
            "output_dir": str(args.output_dir),
        },
    )


if __name__ == "__main__":
    main()
