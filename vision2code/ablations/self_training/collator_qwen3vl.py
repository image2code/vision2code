from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from vision2code.generation.prompts import SYSTEM_PROMPT, USER_PROMPT


def build_messages_for_sample(image_obj: Image.Image, code_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": USER_PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": code_text}]},
    ]


@dataclass
class Qwen3VLCollator:
    processor: Any
    max_length: int | None = None

    def _image_token_ids(self) -> list[int]:
        tokenizer = self.processor.tokenizer
        ids: list[int] = []
        special = getattr(tokenizer, "additional_special_tokens", []) or []
        special_ids = getattr(tokenizer, "additional_special_tokens_ids", []) or []
        for token, token_id in zip(special, special_ids):
            if "image" in str(token).lower():
                ids.append(int(token_id))
        return ids

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []
        for example in examples:
            image = Image.open(Path(str(example["source_image_path"]))).convert("RGB")
            messages = build_messages_for_sample(image, str(example["code_text"]))
            prompt_messages = messages[:-1]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
            images.append(image)

        kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True}
        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = int(self.max_length)

        batch = self.processor(text=texts, images=images, **kwargs)
        labels = batch["input_ids"].clone()

        prompt_batch = self.processor(text=prompt_texts, images=images, **kwargs)
        if "attention_mask" in prompt_batch:
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        else:
            prompt_lengths = [len(ids) for ids in prompt_batch["input_ids"]]
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for image_token_id in self._image_token_ids():
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch
