from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_DEEPSPEED_CONFIG = Path("configs/ablations/deepspeed_zero3.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tuning for Vision2Code self-training ablations.")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--dev-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED_CONFIG)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none", help='Use "wandb" only when configured locally.')
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def patch_deepspeed_stage3_norm() -> None:
    """Reduce ZeRO-3 grad-norm peak memory for smaller GPU memory budgets."""
    try:
        from deepspeed.runtime.zero import stage3 as ds_stage3
        import torch
    except Exception:
        return

    if getattr(ds_stage3.DeepSpeedZeroOptimizer_Stage3, "_vision2code_norm_patch", False):
        return

    buffer_size = int(os.environ.get("DS_STAGE3_NORM_BUFFER_SIZE", "25000000"))

    def _constant_buffered_norm2_fp32(self, input, buffer_size=buffer_size):
        norm = None
        for part in input.view(-1).split(buffer_size):
            part_norm = part.data.float().norm(2) ** 2.0
            norm = part_norm if norm is None else norm + part_norm
        if norm is None:
            return torch.tensor(0.0, device=input.device)
        return norm**0.5

    ds_stage3.DeepSpeedZeroOptimizer_Stage3._constant_buffered_norm2 = _constant_buffered_norm2_fp32
    ds_stage3.DeepSpeedZeroOptimizer_Stage3._vision2code_norm_patch = True


def latest_checkpoint_dir(checkpoints_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except Exception:
            continue
        candidates.append((step, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def best_checkpoint_from_log_history(log_history: list[dict[str, Any]], checkpoints_dir: Path) -> Path | None:
    best_step: int | None = None
    best_loss: float | None = None
    for row in log_history:
        if "eval_loss" not in row:
            continue
        eval_loss = row.get("eval_loss")
        step = row.get("step")
        if not isinstance(eval_loss, (int, float)) or not isinstance(step, (int, float)):
            continue
        if best_loss is None or float(eval_loss) < best_loss:
            best_loss = float(eval_loss)
            best_step = int(step)
    if best_step is None:
        return None
    candidate = checkpoints_dir / f"checkpoint-{best_step}"
    return candidate if candidate.exists() else None


def export_zero3_checkpoint(checkpoint_dir: Path, export_dir: Path, processor: Any) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"):
        for src in checkpoint_dir.glob(pattern):
            if src.name in {"trainer_state.json", "training_args.bin", "scheduler.pt"}:
                continue
            if src.is_file():
                shutil.copy2(src, export_dir / src.name)
    processor.save_pretrained(str(export_dir))
    converter = checkpoint_dir / "zero_to_fp32.py"
    output_bin = export_dir / "pytorch_model.bin"
    if converter.exists():
        subprocess.run([sys.executable, str(converter), str(checkpoint_dir), str(output_bin)], check=True)
    return output_bin


def main() -> None:
    args = parse_args()
    if not args.train_dataset.exists() or not args.dev_dataset.exists():
        raise RuntimeError("Train/dev dataset paths are required and must exist.")

    from datasets import load_from_disk

    train_ds = load_from_disk(str(args.train_dataset))
    dev_ds = load_from_disk(str(args.dev_dataset))
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_dev_samples is not None:
        dev_ds = dev_ds.select(range(min(args.max_dev_samples, len(dev_ds))))

    print(f"[INFO] model id: {args.model_id}", flush=True)
    print(f"[INFO] train samples: {len(train_ds)} | dev samples: {len(dev_ds)}", flush=True)
    if args.dry_run:
        sample = train_ds[0]
        print("[DRY-RUN] sample keys:", sorted(sample.keys()), flush=True)
        print("[DRY-RUN] source image:", sample.get("source_image_path", ""), flush=True)
        print("[DRY-RUN] code chars:", len(str(sample.get("code_text", ""))), flush=True)
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    from vision2code.ablations.self_training.collator_qwen3vl import Qwen3VLCollator

    if args.report_to == "wandb" and args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    patch_deepspeed_stage3_norm()
    sft_kwargs = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "report_to": [] if args.report_to in {"", "none"} else args.report_to,
        "run_name": args.wandb_run_name or args.output_dir.name,
        "seed": args.seed,
        "deepspeed": str(args.deepspeed_config) if args.deepspeed_config else None,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    signature = inspect.signature(SFTConfig)
    if "max_length" in signature.parameters:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in signature.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    else:
        raise RuntimeError("SFTConfig does not support max_length/max_seq_length in this TRL version.")

    training_args = SFTConfig(**sft_kwargs)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    collator = Qwen3VLCollator(processor=processor, max_length=args.max_seq_length)

    class LocalSFTTrainer(SFTTrainer):
        def create_model_card(self, *trainer_args, **trainer_kwargs):
            return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trainer = LocalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor,
        data_collator=collator,
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    best_checkpoint_dir = None
    state_best = getattr(trainer.state, "best_model_checkpoint", None)
    if state_best:
        candidate = Path(state_best)
        if candidate.exists():
            best_checkpoint_dir = candidate
    if best_checkpoint_dir is None:
        best_checkpoint_dir = best_checkpoint_from_log_history(trainer.state.log_history, checkpoints_dir)
    final_checkpoint_dir = latest_checkpoint_dir(checkpoints_dir)

    final_model_dir = args.output_dir / "final_model"
    best_model_dir = None
    if trainer.is_world_process_zero():
        final_model_dir.mkdir(parents=True, exist_ok=True)
        if final_checkpoint_dir is not None:
            export_zero3_checkpoint(final_checkpoint_dir, final_model_dir, processor)
        else:
            trainer.save_model(str(final_model_dir))
            processor.save_pretrained(str(final_model_dir))
        if best_checkpoint_dir is not None:
            best_model_dir = args.output_dir / "best_model"
            export_zero3_checkpoint(best_checkpoint_dir, best_model_dir, processor)
        run_meta = {
            "model_id": args.model_id,
            "train_dataset": str(args.train_dataset),
            "dev_dataset": str(args.dev_dataset),
            "output_dir": str(args.output_dir),
            "report_to": args.report_to,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name or args.output_dir.name,
            "final_model_dir": str(final_model_dir),
            "best_model_dir": str(best_model_dir) if best_model_dir is not None else None,
            "best_checkpoint": str(best_checkpoint_dir) if best_checkpoint_dir is not None else None,
            "final_checkpoint": str(final_checkpoint_dir) if final_checkpoint_dir is not None else None,
            "best_metric_name": "eval_loss",
        }
        (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[DONE] final model: {final_model_dir}", flush=True)


if __name__ == "__main__":
    main()
