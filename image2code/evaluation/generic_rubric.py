from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from image2code.evaluation.dataset_rubrics import inspect_candidate_image, parse_json_object

GENERIC_RUBRIC_VERSION = "generic_recreation_rubric_v1_equal_weight"
SCORE_MIN = 0.0
SCORE_MAX = 5.0


def _category(category_id: str, label: str, description: str) -> Dict[str, Any]:
    return {
        "id": category_id,
        "label": label,
        "weight": 0.25,
        "description": description,
    }


GENERIC_RUBRIC = {
    "rubric_dataset": "generic",
    "categories": [
        _category(
            "core_information_fidelity",
            "Core Information Fidelity",
            "Preserves important visible information, values, objects, relations, or semantics.",
        ),
        _category(
            "structure_layout_fidelity",
            "Structure and Layout Fidelity",
            "Preserves overall layout, geometry, chart or diagram structure, document structure, and spatial arrangement.",
        ),
        _category(
            "text_annotation_accuracy",
            "Text and Annotation Accuracy",
            "Preserves readable labels, numbers, symbols, legends, annotations, and text.",
        ),
        _category(
            "visual_completeness_cleanliness",
            "Visual Completeness and Cleanliness",
            "The output is complete, unclipped, readable, nonblank, and visually clean.",
        ),
    ],
}


GENERIC_RATER_SYSTEM_PROMPT = """\
You are a rigorous but calibrated evaluator for image recreation quality.
Compare the reference image against the candidate rendered image.
Judge only what is visibly present in the images.

Use the rubric provided by the user message. It applies to every dataset.
Be strict about semantically wrong recreations, but do not over-focus on small cosmetic differences.

Scoring anchors on a 0.0 to 5.0 scale:
- 5.0 = near-exact or clearly excellent
- 4.0 = strong recreation with minor issues
- 3.0 = good but with noticeable structural, text, or semantic problems
- 2.0 = partial recreation with important errors
- 1.0 = major mismatch
- 0.0 = missing, unreadable, blank, or fundamentally broken

Return JSON only. Do not include markdown fences.
"""


def encode_image_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".gif":
        mime = "image/gif"
    elif suffix == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64,{b64}"


def generic_rubric_markdown() -> str:
    lines = ["Rubric: generic image recreation baseline"]
    for category in GENERIC_RUBRIC["categories"]:
        lines.append(f"- {category['id']} (weight {category['weight']:.2f}): {category['description']}")
    return "\n".join(lines)


def required_category_ids() -> List[str]:
    return [str(category["id"]) for category in GENERIC_RUBRIC["categories"]]


def build_generic_rater_user_content(
    *,
    source_image_path: Path,
    rendered_image_path: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    metadata_payload = dict(metadata or {})
    prompt_text = (
        "Evaluate the candidate recreation against the reference image using this rubric.\n\n"
        f"Rubric version: {GENERIC_RUBRIC_VERSION}\n"
        "Rubric dataset: generic\n"
        f"Source dataset name, if known: {dataset_name or ''}\n\n"
        "Categories:\n"
        f"{generic_rubric_markdown()}\n\n"
        "Reference metadata:\n"
        f"{json.dumps(metadata_payload, indent=2, ensure_ascii=False)}\n\n"
        "Return JSON with exactly these top-level keys:\n"
        '1) "category_scores"\n'
        '2) "rationales"\n'
        '3) "strengths"\n'
        '4) "issues"\n'
        '5) "overall_summary"\n\n'
        "Rules:\n"
        "- 'category_scores' must contain every category id with a numeric score from 0.0 to 5.0.\n"
        "- Use decimals in increments of 0.1 when needed.\n"
        "- 'rationales' must contain the same keys, with one short rationale per category.\n"
        "- 'strengths' must be a short list of 1 to 3 bullets.\n"
        "- 'issues' must be a short list of 1 to 5 bullets.\n"
        "- 'overall_summary' must be 1 to 3 concise sentences.\n"
        "- Use the same four generic categories for every dataset.\n"
        "- Do not compute the final 0.0 to 5.0 rating yourself; provide category scores only.\n"
    )

    return [
        {"type": "text", "text": "Reference image:"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_to_data_url(source_image_path)},
        },
        {"type": "text", "text": "Candidate rendered image:"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_to_data_url(rendered_image_path)},
        },
        {"type": "text", "text": prompt_text},
    ]


def build_generic_rater_repair_prompt(raw_response: str, error_message: str) -> str:
    expected_keys = ", ".join(required_category_ids())
    return (
        "The previous evaluation response was invalid.\n"
        f"Validation error: {error_message}\n\n"
        "Return corrected JSON only with these top-level keys:\n"
        '  "category_scores", "rationales", "strengths", "issues", "overall_summary"\n'
        "The 'category_scores' object must include every category id exactly once.\n"
        "Each category score must be numeric in the range 0.0 to 5.0, preferably in 0.1 increments.\n"
        f"Required category ids: {expected_keys}\n\n"
        "Previous response:\n"
        f"{raw_response}"
    )


def _round_tenth(value: float) -> float:
    return round(float(value) + 1e-8, 1)


def _coerce_decimal_score(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean score for {key}.")
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str) and value.strip():
        score = float(value.strip())
    else:
        raise ValueError(f"Missing score for {key}.")

    if not math.isfinite(score):
        raise ValueError(f"Invalid score for {key}: {value!r}")
    if score < SCORE_MIN or score > SCORE_MAX:
        raise ValueError(f"Score for {key} must be between {SCORE_MIN:.1f} and {SCORE_MAX:.1f} (got {score}).")
    return _round_tenth(score)


def aggregate_generic_rating(
    parsed: Dict[str, Any],
    *,
    dataset_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    execution_status: str = "ok",
    execution_error_present: bool = False,
    candidate_inspection: Optional[Mapping[str, Any]] = None,
    reference_image_path: str = "",
    candidate_image_path: str = "",
) -> Dict[str, Any]:
    scores_obj = parsed.get("category_scores")
    if not isinstance(scores_obj, dict):
        scores_obj = parsed.get("subscores")
    rationales_obj = parsed.get("rationales")
    if not isinstance(scores_obj, dict):
        raise ValueError("Response is missing a 'category_scores' object.")
    if not isinstance(rationales_obj, dict):
        raise ValueError("Response is missing a 'rationales' object.")

    category_results: Dict[str, Any] = {}
    raw_score = 0.0

    for category in GENERIC_RUBRIC["categories"]:
        category_id = str(category["id"])
        score = _coerce_decimal_score(scores_obj.get(category_id), category_id)
        rationale = str(rationales_obj.get(category_id, "")).strip()
        weighted = score * float(category["weight"])
        raw_score += weighted
        category_results[category_id] = {
            "label": category["label"],
            "description": category["description"],
            "weight": category["weight"],
            "score_0_to_5": score,
            "weighted_score_0_to_5": round(weighted, 4),
            "rationale": rationale,
            "is_critical": False,
        }

    strengths = [str(item).strip() for item in parsed.get("strengths", []) if str(item).strip()]
    issues = [str(item).strip() for item in parsed.get("issues", []) if str(item).strip()]
    overall_summary = str(parsed.get("overall_summary", "")).strip()
    final_rating = _round_tenth(max(SCORE_MIN, min(SCORE_MAX, raw_score)))

    if execution_status != "ok":
        issues = [f"Render failed before visual evaluation: {execution_status}"]
        overall_summary = "The generated code did not produce a valid rendered image, so the sample receives the minimum rating."
        raw_score = 0.0
        final_rating = 0.0

    return {
        "rubric_version": GENERIC_RUBRIC_VERSION,
        "rubric_dataset": "generic",
        "source_dataset": dataset_name or dict(metadata or {}).get("dataset", ""),
        "status": "ok" if execution_status == "ok" else "render_failed",
        "execution_status": execution_status,
        "execution_detail": "" if execution_status == "ok" else execution_status,
        "execution_error_present": execution_error_present,
        "reference_image_path": reference_image_path,
        "candidate_image_path": candidate_image_path,
        "candidate_inspection": dict(candidate_inspection or {}),
        "category_scores": category_results,
        "raw_score_0_to_5": round(raw_score, 4),
        "raw_visual_score_0_to_5": round(raw_score, 4),
        "provisional_rating_0_to_5": final_rating,
        "applied_caps": [],
        "final_rating_0_to_5": final_rating,
        "strengths": strengths[:3],
        "issues": issues[:5],
        "overall_summary": overall_summary,
    }


def build_generic_render_failure_rating(
    status: str,
    *,
    dataset_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    execution_error_present: bool = True,
    reference_image_path: str = "",
    candidate_image_path: str = "",
    candidate_inspection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    category_scores: Dict[str, Any] = {}
    for category in GENERIC_RUBRIC["categories"]:
        category_scores[str(category["id"])] = {
            "label": category["label"],
            "description": category["description"],
            "weight": category["weight"],
            "score_0_to_5": 0.0,
            "weighted_score_0_to_5": 0.0,
            "rationale": f"Candidate image unavailable because generation or render failed: {status}",
            "is_critical": False,
        }

    return {
        "rubric_version": GENERIC_RUBRIC_VERSION,
        "rubric_dataset": "generic",
        "source_dataset": dataset_name or dict(metadata or {}).get("dataset", ""),
        "status": "render_failed",
        "execution_status": "render_failed",
        "execution_detail": status,
        "execution_error_present": execution_error_present,
        "reference_image_path": reference_image_path,
        "candidate_image_path": candidate_image_path,
        "candidate_inspection": dict(candidate_inspection or {}),
        "category_scores": category_scores,
        "raw_score_0_to_5": 0.0,
        "raw_visual_score_0_to_5": 0.0,
        "provisional_rating_0_to_5": 0.0,
        "applied_caps": [],
        "final_rating_0_to_5": 0.0,
        "strengths": [],
        "issues": [f"Render failed before visual evaluation: {status}"],
        "overall_summary": "The generated code did not produce a valid rendered image, so the sample receives the minimum rating.",
    }


def validate_generic_rating_payload(payload: Mapping[str, Any]) -> bool:
    if payload.get("rubric_version") != GENERIC_RUBRIC_VERSION:
        return False
    if "final_rating_0_to_5" not in payload:
        return False
    category_scores = payload.get("category_scores")
    if not isinstance(category_scores, Mapping):
        return False
    return set(category_scores) == set(required_category_ids())
