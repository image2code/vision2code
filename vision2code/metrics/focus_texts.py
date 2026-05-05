from __future__ import annotations

from typing import Dict

FOCUS_TEXT_STRATEGY = "per_dataset_rubric_summary_v1"

DATASET_EMBEDDING_FOCUS_TEXTS: Dict[str, str] = {
    "ChartQA": "chart type, plotted values, axis labels, legend entries, overall layout",
    "ChemVQA-2K": "molecular structure, chemical symbols, bond lines, annotations, color-coded semantics",
    "DocVQA": "text accuracy, document layout, table structure, labels, visual organization",
    "EEE-Bench": "circuit topology, component identity, labels, values, symbols, current or voltage direction",
    "GEOQA_8K_R1V": "geometry structure, point labels, lengths, angles, markers, annotation placement",
    "Geoperception": "geometry structure, labels, markers, relationships, annotation placement",
    "Graph-Algorithms": "graph topology, node labels, edge directions, edge weights, ordering",
    "GraphVQA-Swift": "graph structure, node labels, edge labels, attributes, label binding",
    "OlympiadBench": "geometry structure, labels, givens, markers, precise relationships",
    "Physics": "object placement, vectors, labels, numerical givens, physically correct structure",
    "dvqa": "chart structure, plotted values, readable labels, legend entries, information content",
    "figureqa": "chart structure, plotted trends, labels, legend entries, visual relationships",
    "geometry3k": "geometry shapes, point labels, markers, numerical annotations, exact relations",
    "matplotlib": "plot layout, data traces, axes, titles, legends, annotations, figure styling",
    "spatialvlm_qa": "object identity, spatial relations, depth cues, layout, scene completeness",
}


def focus_text_for_dataset(dataset_name: str) -> str:
    try:
        return DATASET_EMBEDDING_FOCUS_TEXTS[dataset_name]
    except KeyError as exc:
        raise KeyError(f"Missing embedding focus text for dataset: {dataset_name}") from exc
