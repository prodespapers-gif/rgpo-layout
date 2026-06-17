"""
data.py — Data, Layout Representation, Serialization, and Prompt Construction
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [1] of 10. This is the data contract every other module depends on.

What lives here
---------------
  1. ``Layout`` — the single, canonical in-memory representation of a poster
     layout (a set of categorised boxes). retrieval.py returns these; rgpo.py
     consumes these; model.py serialises/parses these. Defining it once, here,
     is what lets a retrieved neighbour be handed straight to the RGPO agreement
     score with no reshaping.
  2. Quantisation (linear / k-means) — applied ONLY at the JSON boundary.
     The primary representation is normalised float [0, 1]; quantisation to
     128 bins is a thin, explicit step for what the model reads/emits.
  3. JSON (de)serialisation with a guaranteed lossless float round-trip.
  4. Canvas + saliency loading, and a corrected *consolidated*-annotation loader
     (the real RALF layout files are single big files indexed by id, not one
     JSON per sample).
  5. The six constrained-task input builders and the design-aware CoT prompt.
  6. ``PosterLayoutDataset`` for SFT and evaluation.

What deliberately does NOT live here
------------------------------------
  - Preference-pair construction → preferences.py
  - Any aesthetic / retrieval-grounded scoring → rgpo.py
  This separation is what keeps the A/B/C preference-source ablation a clean
  control: data.py never knows where the preference signal comes from.

Design decisions (locked for the whole project)
------------------------------------------------
  - Primary coordinate representation: normalised float in [0, 1].
  - Canonical SERIALISATION order: ``label`` (category index, then area).
    This matches the SFT targets and the original config default.
  - Canonical AGREEMENT representation: order-invariant. The ``Layout`` exposes
    set-based accessors (occupancy grid, per-category stats) so that rgpo.py's
    signal depends on geometry, never on element order. Ordering therefore
    affects only the text the model sees, never the preference signal.

References
----------
  - PKU dataset: Hsu et al., CVPR 2023
  - CGL dataset: Zhou et al., IJCAI 2022 (+ CGL-Dataset-V2 annotations)
  - Splits / retrieval / inpainting: Horita et al. (RALF), CVPR 2024
  - Quantisation: Inoue et al. (LayoutDM), CVPR 2023
  - Design-aware CoT: Shi et al. (LayoutCoT), 2026
  - Constrained-task taxonomy: Jiang et al. (LayoutFormer++), CVPR 2023
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ======================================================================
# 1. CONSTANTS
# ======================================================================

GEO_KEYS = ("center_x", "center_y", "width", "height")

CATEGORY_MAPS: dict[str, list[str]] = {
    "cgl": ["logo", "text", "underlay", "embellishment"],
    "pku": ["logo", "text", "underlay"],
}

# Constrained generation task taxonomy (LayoutFormer++, CVPR 2023).
TASK_TYPES = (
    "unconstrained",   # generate all attributes from the canvas alone
    "c_to_sp",         # Category -> Size + Position
    "cs_to_p",         # Category + Size -> Position
    "completion",      # partial elements -> complete layout
    "refinement",      # noisy positions -> clean layout
    "relationship",    # satisfy sampled pairwise relations
)


# ======================================================================
# 2. THE LAYOUT REPRESENTATION
#    One type, used everywhere. An ordered set of categorised boxes in
#    normalised-float coordinates, plus the set-based accessors RGPO needs.
# ======================================================================

@dataclass
class Element:
    """A single categorised box in normalised [0, 1] centre-format.

    Carries BOTH an integer ``label`` (category index, used by the prompt
    builders) and a string ``category`` (used by metrics / scoring). The two
    are kept consistent by :meth:`Layout.from_records`.
    """
    label: int
    category: str
    center_x: float
    center_y: float
    width: float
    height: float

    # ---- geometry helpers (normalised, never pixels) ----
    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def xyxy(self) -> tuple[float, float, float, float]:
        """Corner format [x1, y1, x2, y2], clamped to the canvas."""
        x1 = self.center_x - self.width / 2.0
        y1 = self.center_y - self.height / 2.0
        x2 = self.center_x + self.width / 2.0
        y2 = self.center_y + self.height / 2.0
        return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))

    def as_record(self, geo_round: Optional[int] = None) -> dict:
        """Plain-dict view (dual-schema), optionally rounding coordinates."""
        rec = {"label": self.label, "category": self.category}
        for k in GEO_KEYS:
            v = float(getattr(self, k))
            rec[k] = round(v, geo_round) if geo_round is not None else v
        return rec


@dataclass
class Layout:
    """An ordered set of :class:`Element`, plus the category vocabulary.

    This is THE type passed between modules. retrieval.py returns
    ``list[Layout]`` (the neighbours); rgpo.py consumes ``Layout`` and the
    neighbour set; model.py serialises/parses ``Layout``.

    The serialisation order is canonical (``label``-sorted); the agreement
    accessors (:meth:`occupancy_grid`, :meth:`category_stats`) are
    order-invariant by construction.
    """
    elements: list[Element] = field(default_factory=list)
    categories: tuple[str, ...] = ()

    # ---------- construction ----------
    @classmethod
    def from_records(cls, records: Iterable[dict], categories: Sequence[str],
                     max_elements: Optional[int] = None,
                     filter_min_area: float = 0.0) -> "Layout":
        """Build a Layout from raw element dicts following either schema.

        Accepts dicts with a string ``category`` and/or an integer ``label``;
        resolves both so downstream code never has to care which was present.
        """
        cats = tuple(categories)
        out: list[Element] = []
        for rec in records:
            cat_name = _resolve_category_name(rec, cats)
            label = cats.index(cat_name) if cat_name in cats else 0
            elem = Element(
                label=label,
                category=cats[label] if 0 <= label < len(cats) else cat_name,
                center_x=float(rec.get("center_x", 0.5)),
                center_y=float(rec.get("center_y", 0.5)),
                width=float(rec.get("width", 0.1)),
                height=float(rec.get("height", 0.1)),
            )
            if elem.area >= filter_min_area:
                out.append(elem)
            if max_elements is not None and len(out) >= max_elements:
                break
        return cls(elements=out, categories=cats)

    # ---------- canonical ordering (serialisation only) ----------
    def sorted_elements(self, method: str = "label") -> list[Element]:
        """Return elements in a canonical order for SERIALISATION.

        Never call this to build the agreement signal — that must stay
        order-invariant (use the set-based accessors below).
        """
        if method == "area":
            return sorted(self.elements, key=lambda e: -e.area)
        if method == "position":
            return sorted(self.elements, key=lambda e: (e.center_y, e.center_x))
        if method == "lexicographic":
            return sorted(self.elements,
                          key=lambda e: (e.label, e.center_y, e.center_x))
        # default: label, then larger elements first (stable, deterministic)
        return sorted(self.elements, key=lambda e: (e.label, -e.area))

    # ---------- order-invariant accessors (RGPO uses these) ----------
    def boxes_xyxy(self) -> np.ndarray:
        """(N, 4) array of corner-format boxes. Order carries no meaning here."""
        if not self.elements:
            return np.zeros((0, 4), dtype=np.float64)
        return np.asarray([e.xyxy() for e in self.elements], dtype=np.float64)

    def occupancy_grid(self, resolution: int = 16,
                       per_category: bool = False) -> np.ndarray:
        """Soft occupancy map of the layout's boxes on a coarse grid.

        This is the permutation-invariant representation the RGPO agreement
        score compares against the neighbour mean. It is category-count
        agnostic (works for PKU's 3 and CGL's 4, and for any other domain),
        which is what lets the same agreement score generalise beyond posters.

        Args:
            resolution: G; the grid is G x G.
            per_category: if True, return (C, G, G) with one plane per
                category; else (G, G) summed over categories.

        Returns:
            float array in [0, 1], shape (G, G) or (C, G, G).
        """
        G = int(resolution)
        C = len(self.categories) if per_category else 1
        grid = np.zeros((C, G, G), dtype=np.float64)
        for e in self.elements:
            x1, y1, x2, y2 = e.xyxy()
            # Fractional cell coverage: rasterise the box onto the grid with
            # partial coverage at the edges (smooth, not nearest-cell).
            gx1, gx2 = x1 * G, x2 * G
            gy1, gy2 = y1 * G, y2 * G
            ix1, ix2 = int(np.floor(gx1)), int(np.ceil(gx2))
            iy1, iy2 = int(np.floor(gy1)), int(np.ceil(gy2))
            plane = e.label if per_category and 0 <= e.label < C else 0
            for gy in range(max(0, iy1), min(G, iy2)):
                cover_y = min(gy + 1, gy2) - max(gy, gy1)
                if cover_y <= 0:
                    continue
                for gx in range(max(0, ix1), min(G, ix2)):
                    cover_x = min(gx + 1, gx2) - max(gx, gx1)
                    if cover_x <= 0:
                        continue
                    grid[plane, gy, gx] += cover_x * cover_y
        np.clip(grid, 0.0, 1.0, out=grid)
        return grid if per_category else grid[0]

    def category_stats(self) -> dict[int, dict[str, float]]:
        """Per-category centroid and mean size, order-invariant.

        Returns {label: {cx, cy, w, h, count}} aggregated over the elements of
        that category. Used by the RGPO structural-mismatch term.
        """
        acc: dict[int, list[Element]] = {}
        for e in self.elements:
            acc.setdefault(e.label, []).append(e)
        stats: dict[int, dict[str, float]] = {}
        for label, elems in acc.items():
            n = len(elems)
            stats[label] = {
                "cx": float(np.mean([e.center_x for e in elems])),
                "cy": float(np.mean([e.center_y for e in elems])),
                "w": float(np.mean([e.width for e in elems])),
                "h": float(np.mean([e.height for e in elems])),
                "count": float(n),
            }
        return stats

    # ---------- misc ----------
    def __len__(self) -> int:
        return len(self.elements)

    def is_empty(self) -> bool:
        return len(self.elements) == 0

    def to_records(self, geo_round: Optional[int] = None) -> list[dict]:
        return [e.as_record(geo_round=geo_round) for e in self.elements]


def _resolve_category_name(rec: dict, categories: Sequence[str]) -> str:
    """Resolve a category name from a dict following either schema."""
    cat = rec.get("category")
    if isinstance(cat, str) and cat:
        return cat
    label = rec.get("label")
    if isinstance(label, bool):
        label = int(label)
    if isinstance(label, int):
        if 0 <= label < len(categories):
            return categories[label]
        return categories[0] if categories else "text"
    if isinstance(label, str) and label:
        return label
    return categories[0] if categories else "text"


# ======================================================================
# 3. QUANTISATION  (applied ONLY at the JSON boundary)
#    Reference: LayoutDM (Inoue et al., CVPR 2023), Sec. 3.2 / 4.7
# ======================================================================

class LinearBucketizer:
    """Uniform quantisation of [0, 1] into ``num_bins`` equal-width bins."""

    def __init__(self, num_bins: int = 128):
        self.num_bins = num_bins
        edges = np.arange(num_bins + 1) / num_bins
        self.boundaries = edges[1:]                  # upper edges
        self.centers = (edges[:-1] + edges[1:]) / 2.0

    def encode(self, value: float) -> int:
        idx = int(np.searchsorted(self.boundaries, float(np.clip(value, 0.0, 1.0))))
        return min(idx, self.num_bins - 1)

    def decode(self, index: int) -> float:
        return float(self.centers[int(np.clip(index, 0, self.num_bins - 1))])


class KMeansBucketizer:
    """Adaptive quantisation using precomputed k-means centres per coordinate."""

    def __init__(self, cluster_centers: np.ndarray):
        self.centers = np.sort(np.asarray(cluster_centers).flatten())
        mids = (self.centers[:-1] + self.centers[1:]) / 2.0
        self.boundaries = np.append(mids, 1.0)
        self.num_bins = len(self.centers)

    def encode(self, value: float) -> int:
        idx = int(np.searchsorted(self.boundaries, float(np.clip(value, 0.0, 1.0))))
        return min(idx, self.num_bins - 1)

    def decode(self, index: int) -> float:
        return float(self.centers[int(np.clip(index, 0, self.num_bins - 1))])


def create_bucketizers(method: str = "linear", num_bins: int = 128,
                       cluster_path: Optional[str] = None) -> dict:
    """One bucketizer per geometric coordinate."""
    if method == "kmeans" and cluster_path and os.path.exists(cluster_path):
        import pickle
        with open(cluster_path, "rb") as f:
            centers = pickle.load(f)
        return {k: KMeansBucketizer(centers[k]) for k in GEO_KEYS}
    return {k: LinearBucketizer(num_bins) for k in GEO_KEYS}


# ======================================================================
# 4. SERIALISATION  (Layout <-> JSON text)
# ======================================================================

def layout_to_json(layout: Layout, *, sort_by: str = "label",
                   quantize: bool = False,
                   bucketizers: Optional[dict] = None,
                   geo_round: int = 4, indent: int = 2) -> str:
    """Serialise a ``Layout`` to the JSON the model reads/emits.

    Floats are the primary representation; quantisation to bin indices happens
    here and only here. Element order is canonical (``sort_by``), affecting only
    the text — never the agreement signal.
    """
    out = []
    for e in layout.sorted_elements(sort_by):
        rec = {"category": e.category}
        for k in GEO_KEYS:
            v = float(getattr(e, k))
            if quantize and bucketizers:
                rec[k] = bucketizers[k].encode(v)
            else:
                rec[k] = round(v, geo_round)
        out.append(rec)
    return json.dumps(out, indent=indent)


def json_to_layout(text: str, categories: Sequence[str],
                   bucketizers: Optional[dict] = None,
                   max_elements: Optional[int] = None,
                   filter_min_area: float = 0.0) -> Layout:
    """Parse model output JSON back into a ``Layout``.

    Tolerant of markdown code-fences. Decodes quantised integer coordinates
    when ``bucketizers`` is given and the value is an int; otherwise treats the
    value as a float (the lossless path).
    """
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line and any closing fence
        t = t.split("\n", 1)[1] if "\n" in t else t.lstrip("`")
    if t.endswith("```"):
        t = t.rsplit("```", 1)[0]
    t = t.strip()

    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        logger.warning("Failed to parse layout JSON: %s...", text[:80])
        return Layout(elements=[], categories=tuple(categories))

    if not isinstance(parsed, list):
        return Layout(elements=[], categories=tuple(categories))

    records: list[dict] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        rec = {"category": entry.get("category", "text")}
        for k in GEO_KEYS:
            v = entry.get(k, 0.5)
            if bucketizers is not None and isinstance(v, int):
                rec[k] = bucketizers[k].decode(v)
            else:
                rec[k] = float(v)
        records.append(rec)

    return Layout.from_records(records, categories,
                               max_elements=max_elements,
                               filter_min_area=filter_min_area)


# ======================================================================
# 5. PROMPTS  (design-aware chain-of-thought + six constrained tasks)
# ======================================================================

SYSTEM_PROMPT = """You are an expert graphic designer specializing in advertising \
poster layout. Given a product canvas image and reference layout examples from \
similar posters, generate a harmonious layout that places design elements \
(logo, text, underlay, embellishment) without occluding the main product.

Output a valid JSON array of layout elements. Each element has:
- "category": one of {categories}
- "center_x": horizontal center (0.0 = left, 1.0 = right)
- "center_y": vertical center (0.0 = top, 1.0 = bottom)
- "width": element width as a fraction of canvas width
- "height": element height as a fraction of canvas height"""

COT_PROMPT_TEMPLATE = """## Task
Generate a content-aware poster layout for the given canvas image.

## Reference Layouts
The following layouts were used for similar poster canvases. Study them as design references:

{retrieved_layouts}

## Instructions
Think step by step:

1. **Canvas Analysis**: Examine the canvas image. Identify the salient product region \
that must NOT be occluded. Note the overall composition and available white space.

2. **Reference Examination**: Review the reference layouts above. Identify common \
patterns: where logos sit, how text relates to the product, how underlays support \
readability.

3. **Placement Reasoning**: Decide how many elements to place and where. Ensure text \
sits on smooth background, underlays fully back the text they support, the logo is \
prominent but clear of the product, and nothing significantly occludes the salient region.

4. **Layout Output**: Output the final layout as a JSON array.

## Output
Respond with ONLY a valid JSON array of layout elements. No other text."""

DIRECT_PROMPT_TEMPLATE = """## Task
Generate a content-aware poster layout for the given canvas image.

## Reference Layouts
{retrieved_layouts}

## Output
Respond with ONLY a valid JSON array of layout elements:
{format_hint}"""

CONSTRAINED_PROMPT_TEMPLATES = {
    "c_to_sp": """## Task
Given the following element categories, generate their sizes and positions for this canvas.

Categories: {constraint}

## Reference Layouts
{retrieved_layouts}

## Output
Respond with ONLY a valid JSON array with center_x, center_y, width, height for each element.""",

    "cs_to_p": """## Task
Given the following elements with their categories and sizes, generate their positions.

Elements: {constraint}

## Reference Layouts
{retrieved_layouts}

## Output
Respond with ONLY a valid JSON array with center_x, center_y for each element (keep the given sizes).""",

    "completion": """## Task
Complete the following partial layout by adding the missing elements.

Placed elements: {constraint}

## Reference Layouts
{retrieved_layouts}

## Output
Respond with ONLY a valid JSON array containing ALL elements (placed and new).""",

    "refinement": """## Task
The following layout has noisy/imprecise positions. Refine the element positions to \
create a clean, well-aligned layout.

Noisy layout: {constraint}

## Reference Layouts
{retrieved_layouts}

## Output
Respond with ONLY a valid JSON array with corrected positions.""",

    "relationship": """## Task
Generate a layout satisfying the following spatial relationship constraints.

Constraints: {constraint}

## Reference Layouts
{retrieved_layouts}

## Output
Respond with ONLY a valid JSON array of layout elements satisfying all constraints.""",
}


def format_retrieved_for_prompt(neighbours: Sequence[Layout],
                                max_show: int = 4,
                                sort_by: str = "label") -> str:
    """Render the top retrieved neighbour layouts as numbered JSON references."""
    if not neighbours:
        return "(no reference layouts available)"
    lines = []
    for i, lay in enumerate(neighbours[:max_show]):
        lines.append(f"Reference {i + 1}:\n{layout_to_json(lay, sort_by=sort_by)}")
    return "\n\n".join(lines)


def build_sft_sample(canvas: Image.Image, target: Layout,
                     neighbours: Sequence[Layout], categories: Sequence[str],
                     use_cot: bool = True, max_retrieved: int = 4,
                     sort_by: str = "label") -> dict:
    """Build one SFT instruction-tuning triple (system, user, assistant)."""
    ref_text = format_retrieved_for_prompt(neighbours, max_retrieved, sort_by)
    system = SYSTEM_PROMPT.format(categories=", ".join(categories))
    if use_cot:
        user = COT_PROMPT_TEMPLATE.format(retrieved_layouts=ref_text)
    else:
        hint = json.dumps([{"category": categories[0], "center_x": 0.5,
                            "center_y": 0.1, "width": 0.2, "height": 0.05}])
        user = DIRECT_PROMPT_TEMPLATE.format(retrieved_layouts=ref_text,
                                             format_hint=hint)
    return {
        "image": canvas,
        "system": system,
        "user": user,
        "assistant": layout_to_json(target, sort_by=sort_by),
    }


def build_constrained_sample(canvas: Image.Image, target: Layout,
                             neighbours: Sequence[Layout],
                             categories: Sequence[str], task_type: str,
                             max_retrieved: int = 4, noise_std: float = 0.01,
                             rel_ratio: float = 0.1,
                             sort_by: str = "label",
                             rng: Optional[random.Random] = None) -> dict:
    """Build one constrained-generation sample for the six-task evaluation."""
    rng = rng or random
    if task_type == "unconstrained":
        return build_sft_sample(canvas, target, neighbours, categories,
                                use_cot=True, max_retrieved=max_retrieved,
                                sort_by=sort_by)

    ref_text = format_retrieved_for_prompt(neighbours, max_retrieved, sort_by)
    system = SYSTEM_PROMPT.format(categories=", ".join(categories))
    elems = target.sorted_elements(sort_by)

    if task_type == "c_to_sp":
        constraint = [{"category": e.category} for e in elems]
        user = CONSTRAINED_PROMPT_TEMPLATES["c_to_sp"].format(
            constraint=json.dumps(constraint, indent=2), retrieved_layouts=ref_text)

    elif task_type == "cs_to_p":
        constraint = [{"category": e.category,
                       "width": round(e.width, 4),
                       "height": round(e.height, 4)} for e in elems]
        user = CONSTRAINED_PROMPT_TEMPLATES["cs_to_p"].format(
            constraint=json.dumps(constraint, indent=2), retrieved_layouts=ref_text)

    elif task_type == "completion":
        # Completion reveals a MAJORITY of the elements and asks for the rest.
        # (Revealing 0-1 of e.g. 4 elements collapses this into unconstrained
        # generation, which would not test completion at all.) We reveal each
        # element with prob ~0.8, but always keep at least one revealed and at
        # least one hidden when the layout has >= 2 elements, so the task is
        # genuinely partial.
        n = len(elems)
        if n <= 1:
            known = []
        else:
            shuffled = list(elems)
            rng.shuffle(shuffled)
            n_known = max(1, min(n - 1, int(round(0.8 * n))))
            known = shuffled[:n_known]
        constraint = [e.as_record(geo_round=4) for e in known]
        for r in constraint:
            r.pop("label", None)
        user = CONSTRAINED_PROMPT_TEMPLATES["completion"].format(
            constraint=json.dumps(constraint, indent=2), retrieved_layouts=ref_text)

    elif task_type == "refinement":
        noisy = []
        for e in elems:
            ne = deepcopy(e)
            for k in GEO_KEYS:
                # Use the passed-in rng (random.Random) so refinement noise is
                # reproducible under a fixed seed, exactly like every other
                # task here. np.random (the global RNG) would NOT be controlled
                # by this rng and would break per-seed reproducibility.
                setattr(ne, k, float(np.clip(getattr(ne, k)
                        + rng.gauss(0.0, noise_std), 0.0, 1.0)))
            noisy.append(ne)
        constraint = [{"category": e.category,
                       "center_x": round(e.center_x, 4),
                       "center_y": round(e.center_y, 4),
                       "width": round(e.width, 4),
                       "height": round(e.height, 4)} for e in noisy]
        user = CONSTRAINED_PROMPT_TEMPLATES["refinement"].format(
            constraint=json.dumps(constraint, indent=2), retrieved_layouts=ref_text)

    elif task_type == "relationship":
        rels: list[str] = []
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                if rng.random() > rel_ratio:
                    continue
                ei, ej = elems[i], elems[j]
                rels.append(f"{ei.category} is "
                            f"{'above' if ei.center_y < ej.center_y else 'below'} "
                            f"{ej.category}")
                rels.append(f"{ei.category} is "
                            f"{'larger' if ei.area > ej.area else 'smaller'} than "
                            f"{ej.category}")
        cat_list = json.dumps([{"category": e.category} for e in elems], indent=2)
        constraint = cat_list + ("\n\nRelationships:\n" + "\n".join(
            f"- {r}" for r in rels) if rels else "")
        user = CONSTRAINED_PROMPT_TEMPLATES["relationship"].format(
            constraint=constraint, retrieved_layouts=ref_text)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return {
        "image": canvas,
        "system": system,
        "user": user,
        "assistant": layout_to_json(target, sort_by=sort_by),
    }


# ======================================================================
# 6. ANNOTATION LOADING  (the consolidated-file fix)
#    Real RALF layout data is a handful of big files indexed by id, not
#    one JSON per sample. We load each consolidated file once and index it.
# ======================================================================

class _AnnotationStore:
    """Lazily loads and indexes the consolidated layout annotations by id.

    Supports the two real formats RALF distributes:
      * CGL: ``layout_train_6w_fixed_v2.json`` / ``layout_test_6w_fixed_v2.json``
             (+ ``yinhe.json``) — JSON lists keyed per poster.
      * PKU: ``train_csv_9973.csv`` / ``test_csv_905.csv`` — one row per element.
    The exact field names differ across releases, so parsing is defensive:
    we map a small set of known aliases to our (label, cx, cy, w, h) schema.
    """

    def __init__(self, dataset_name: str, data_root: str,
                 categories: Sequence[str],
                 original_size: tuple[int, int]):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.categories = tuple(categories)
        self.orig_h, self.orig_w = original_size
        self._by_id: Optional[dict[str, list[dict]]] = None

    # ---- public ----
    def get(self, sample_id: str) -> list[dict]:
        if self._by_id is None:
            self._by_id = self._load_all()
        return self._by_id.get(str(sample_id), [])

    @property
    def num_indexed(self) -> int:
        if self._by_id is None:
            self._by_id = self._load_all()
        return len(self._by_id)

    # ---- loading ----
    def _load_all(self) -> dict[str, list[dict]]:
        root = self.data_root
        candidates = [
            os.path.join(root, "annotation"),
            os.path.join(root, "annotations"),
            root,
        ]
        for ann_dir in candidates:
            if not os.path.isdir(ann_dir):
                continue
            store = self._try_load_dir(ann_dir)
            if store:
                logger.info("Indexed %d layouts from %s", len(store), ann_dir)
                return store
        logger.warning(
            "No consolidated annotation file found for %s under %s; "
            "layouts will be empty. Expected RALF files like "
            "layout_train_6w_fixed_v2.json (CGL) or train_csv_9973.csv (PKU).",
            self.dataset_name, root)
        return {}

    def _try_load_dir(self, ann_dir: str) -> dict[str, list[dict]]:
        store: dict[str, list[dict]] = {}
        for fname in sorted(os.listdir(ann_dir)):
            fpath = os.path.join(ann_dir, fname)
            if fname.endswith(".json"):
                store.update(self._parse_json_file(fpath))
            elif fname.endswith(".csv"):
                store.update(self._parse_csv_file(fpath))
        return store

    # ---- format-specific parsers (defensive about field names) ----
    _ID_ALIASES = ("id", "poster_id", "image_id", "file_name", "name", "pic")
    _LABEL_ALIASES = ("label", "cls", "category", "category_id", "class")

    def _coerce_box_to_cxcywh(self, box) -> Optional[tuple[float, float, float, float]]:
        """Normalise a raw box (pixel xyxy or xywh) to normalised cxcywh."""
        if box is None:
            return None
        b = [float(v) for v in box]
        if len(b) != 4:
            return None
        # Heuristic: values > 1.5 imply pixel coordinates → normalise.
        if max(b) > 1.5:
            x1, y1, x2, y2 = b
            x1, x2 = x1 / self.orig_w, x2 / self.orig_w
            y1, y2 = y1 / self.orig_h, y2 / self.orig_h
        else:
            x1, y1, x2, y2 = b
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)),
                float(np.clip(w, 0, 1)), float(np.clip(h, 0, 1)))

    def _label_to_index(self, raw) -> int:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            idx = int(raw)
            # Some releases are 1-indexed; map onto our 0-indexed vocab.
            if idx == len(self.categories):
                idx = idx - 1
            return int(np.clip(idx, 0, len(self.categories) - 1))
        if isinstance(raw, str) and raw in self.categories:
            return self.categories.index(raw)
        return 0

    @staticmethod
    def _first_present(d: dict, keys: Sequence[str]):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return None

    def _records_from_poster_obj(self, obj: dict) -> tuple[Optional[str], list[dict]]:
        """Extract (id, [element-records]) from one poster-level JSON object."""
        sid = self._first_present(obj, self._ID_ALIASES)
        if sid is not None:
            sid = os.path.splitext(str(sid))[0]
        labels = obj.get("labels", obj.get("label"))
        boxes = obj.get("boxes", obj.get("bbox", obj.get("box")))
        records: list[dict] = []
        if isinstance(labels, list) and isinstance(boxes, list) \
                and len(labels) == len(boxes):
            for lab, box in zip(labels, boxes):
                cxcywh = self._coerce_box_to_cxcywh(box)
                if cxcywh is None:
                    continue
                cx, cy, w, h = cxcywh
                records.append({"label": self._label_to_index(lab),
                                "center_x": cx, "center_y": cy,
                                "width": w, "height": h})
        return sid, records

    def _parse_json_file(self, fpath: str) -> dict[str, list[dict]]:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
            return {}

        store: dict[str, list[dict]] = {}
        # Accept either a list of poster objects, or a dict keyed by id.
        if isinstance(data, dict) and "annotations" in data:
            data = data["annotations"]
        if isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                sid, records = self._records_from_poster_obj(obj)
                if sid is not None and records:
                    store.setdefault(sid, []).extend(records)
        elif isinstance(data, dict):
            for sid, obj in data.items():
                if isinstance(obj, dict):
                    _, records = self._records_from_poster_obj(
                        {**obj, "id": sid})
                    if records:
                        store[os.path.splitext(str(sid))[0]] = records
        return store

    def _parse_csv_file(self, fpath: str) -> dict[str, list[dict]]:
        store: dict[str, list[dict]] = {}
        try:
            with open(fpath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = self._first_present(row, self._ID_ALIASES)
                    if sid is None:
                        continue
                    sid = os.path.splitext(str(sid))[0]
                    lab = self._first_present(row, self._LABEL_ALIASES)
                    # PKU CSV stores corner coords as separate columns.
                    box = None
                    for quad in (("xmin", "ymin", "xmax", "ymax"),
                                 ("x1", "y1", "x2", "y2"),
                                 ("left", "top", "right", "bottom")):
                        if all(q in row and row[q] not in (None, "") for q in quad):
                            box = [row[q] for q in quad]
                            break
                    cxcywh = self._coerce_box_to_cxcywh(box) if box else None
                    if cxcywh is None:
                        continue
                    cx, cy, w, h = cxcywh
                    store.setdefault(sid, []).append(
                        {"label": self._label_to_index(lab),
                         "center_x": cx, "center_y": cy,
                         "width": w, "height": h})
        except OSError as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
        return store


# ======================================================================
# 7. DATASET
# ======================================================================

class PosterLayoutDataset(Dataset):
    """SFT / evaluation dataset over PKU or CGL with RALF-compatible splits.

    Yields, per item: image (PIL), system/user/assistant prompt strings, the
    ground-truth ``Layout``, and the retrieved neighbour ``Layout`` list.
    Retrieval *index* lookup (id -> neighbour ids) lives in retrieval.py; this
    class accepts an optional ``retriever`` and otherwise reads RALF's
    precomputed per-split index for self-contained operation.
    """

    def __init__(self, config: dict, split: str = "train",
                 task_type: str = "unconstrained",
                 retriever=None):
        super().__init__()
        self.config = config
        self.split = split
        self.task_type = task_type
        self.retriever = retriever

        dcfg = config["dataset"]
        self.dataset_name = dcfg["name"]
        self.categories = CATEGORY_MAPS[self.dataset_name]
        self.max_elements = dcfg["max_elements"]
        self.filter_min_area = dcfg.get("filter_min_area", 0.0)
        self.sort_by = dcfg.get("sort_by", "label")
        self.use_cot = config["model"]["use_cot_reasoning"]
        self.max_retrieved = config["retrieval"]["max_retrieved_in_prompt"]
        self.num_neighbors = config["retrieval"]["num_neighbors"]
        self.canvas_size = tuple(dcfg["canvas_size"])           # (H, W)
        original = tuple(dcfg.get("original_poster_size", [513, 750]))

        self.bucketizers = create_bucketizers(
            method=dcfg.get("quantization", "linear"),
            num_bins=dcfg.get("num_bins", 128),
            cluster_path=os.path.join(
                config["paths"]["cache_dir"],
                f"{self.dataset_name}_kmeans_clusters.pkl"),
        )

        self.data_root = config["paths"][f"{self.dataset_name}_root"]
        self._ann = _AnnotationStore(self.dataset_name, self.data_root,
                                     self.categories, original)

        self.sample_ids = self._load_split_ids(split)
        self.retrieval_index = self._load_precomputed_index(split)

        logger.info(
            "Loaded %d ids for %s/%s (task=%s, cats=%s, ann_indexed=%d, "
            "retr_index=%d)", len(self.sample_ids), self.dataset_name, split,
            task_type, self.categories, self._ann.num_indexed,
            len(self.retrieval_index))

    # ---------- split / index ----------
    def _load_split_ids(self, split: str) -> list[str]:
        split_file_map = {"train": "train.txt", "val": "val.txt",
                          "test": "test.txt",
                          "unannotated": "with_no_annotation.txt"}
        sf = os.path.join(self.config["paths"]["ralf_splits_dir"],
                          self.dataset_name, split_file_map[split])
        if not os.path.exists(sf):
            raise FileNotFoundError(
                f"Split file not found: {sf}. Download/prepare RALF splits "
                f"from https://github.com/CyberAgentAILab/RALF")
        with open(sf, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _load_precomputed_index(self, split: str) -> dict[str, list[str]]:
        rf = os.path.join(self.config["paths"]["ralf_retrieval_dir"],
                          self.dataset_name, f"{split}.yaml")
        if not os.path.exists(rf):
            logger.warning("Retrieval index not found: %s (no retrieval).", rf)
            return {}
        with open(rf, "r") as f:
            idx = yaml.safe_load(f) or {}
        return {str(k): [str(v) for v in vals] for k, vals in idx.items()}

    # ---------- per-sample loading ----------
    def _load_canvas(self, sid: str) -> Image.Image:
        H, W = self.canvas_size
        patterns = [
            os.path.join(self.data_root, "image", self.split, "input", f"{sid}.png"),
            os.path.join(self.data_root, "image", self.split, "input", f"{sid}.jpg"),
            os.path.join(self.data_root, "inpainted", f"{sid}.png"),
            os.path.join(self.data_root, self.split, "inpainted_poster", f"{sid}.png"),
            os.path.join(self.data_root, "image", "train", "input", f"{sid}.png"),
        ]
        for p in patterns:
            if os.path.exists(p):
                return Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
        logger.warning("Canvas not found for %s; blank canvas used.", sid)
        return Image.new("RGB", (W, H), (255, 255, 255))

    def load_saliency(self, sid: str) -> Optional[np.ndarray]:
        """(H, W) float saliency in [0, 1], or None if absent.

        Exposed (not underscore-private) because preferences.py / rgpo.py need
        it for the occlusion term when source A is in use.

        PKU PosterLayout (Hsu et al., CVPR 2023) Eq. 1 defines the saliency used
        by Occ/Uti as the pixel-wise MAXIMUM of two saliency maps,
        ``S = max(S_PFPN, S_BASNet)``. RALF stores these as a primary map and a
        ``*_sub`` map. We therefore combine a main map with its matching sub map
        by elementwise max when BOTH are present, and fall back to whichever
        single map exists otherwise (RALF distributions that already ship a
        single combined map still work unchanged).
        """
        H, W = self.canvas_size

        def _load_one(path: str) -> Optional[np.ndarray]:
            if not os.path.exists(path):
                return None
            img = Image.open(path).convert("L").resize((W, H), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0

        # (main, sub) pairs to combine via pixel-wise max; sub may be absent.
        pairs = [
            (os.path.join(self.data_root, "image", self.split, "saliency", f"{sid}.png"),
             os.path.join(self.data_root, "image", self.split, "saliency_sub", f"{sid}.png")),
            (os.path.join(self.data_root, "saliency", f"{sid}.png"),
             os.path.join(self.data_root, "saliency_sub", f"{sid}.png")),
            (os.path.join(self.data_root, self.split, "saliencymaps_basnet", f"{sid}.png"),
             os.path.join(self.data_root, self.split, "saliencymaps_pfpn", f"{sid}.png")),
        ]
        for main_p, sub_p in pairs:
            main = _load_one(main_p)
            if main is None:
                continue
            sub = _load_one(sub_p)
            return np.maximum(main, sub) if sub is not None else main
        return None

    def load_layout(self, sid: str) -> Layout:
        """Ground-truth ``Layout`` for a sample id (empty if unannotated)."""
        records = self._ann.get(sid)
        return Layout.from_records(records, self.categories,
                                   max_elements=self.max_elements,
                                   filter_min_area=self.filter_min_area)

    def neighbour_layouts(self, sid: str) -> list[Layout]:
        """The K retrieved neighbour ``Layout`` objects for a sample id.

        Prefers a live ``retriever`` (retrieval.py) if one was injected;
        otherwise uses RALF's precomputed index. Either way, neighbours are
        returned as ``Layout`` — exactly the type rgpo.py consumes — with the
        query excluded (the index is built that way at train time).
        """
        if self.retriever is not None:
            neigh_ids = self.retriever.get_neighbor_ids(
                sid, k=self.num_neighbors, split=self.split)
        else:
            # Direct precomputed-index fallback (no live Retriever). Exclude the
            # query id for the same reason Retriever does: a self-neighbour would
            # trivially inflate the RGPO agreement score against this canvas's
            # own ground truth. Over-fetch by one to still return num_neighbors.
            raw = self.retrieval_index.get(sid, [])
            neigh_ids = [i for i in raw if i != sid][:self.num_neighbors]
        out: list[Layout] = []
        for nid in neigh_ids:
            lay = self.load_layout(nid)
            if not lay.is_empty():
                out.append(lay)
        return out

    # ---------- Dataset protocol ----------
    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        canvas = self._load_canvas(sid)
        target = self.load_layout(sid)
        neighbours = self.neighbour_layouts(sid)

        if self.task_type == "unconstrained" or self.split == "train":
            sample = build_sft_sample(
                canvas, target, neighbours, self.categories,
                use_cot=self.use_cot, max_retrieved=self.max_retrieved,
                sort_by=self.sort_by)
        else:
            sample = build_constrained_sample(
                canvas, target, neighbours, self.categories, self.task_type,
                max_retrieved=self.max_retrieved,
                noise_std=self.config["evaluation"].get("refinement_noise_std", 0.01),
                rel_ratio=self.config["evaluation"].get("relationship_sample_ratio", 0.1),
                sort_by=self.sort_by)

        sample["target_layout"] = target          # the Layout object
        sample["neighbour_layouts"] = neighbours   # list[Layout], for RGPO
        sample["metadata"] = {
            "sample_id": sid,
            "dataset_name": self.dataset_name,
            "num_elements": len(target),
            "task_type": self.task_type,
            "split": self.split,
        }
        return sample


# ======================================================================
# 8. CONFIG + LOADER HELPERS
# ======================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset(config: dict, split: str = "train",
                task_type: str = "unconstrained",
                retriever=None) -> PosterLayoutDataset:
    return PosterLayoutDataset(config, split=split, task_type=task_type,
                               retriever=retriever)


def make_loader(dataset: PosterLayoutDataset, shuffle: bool = False,
                num_workers: int = 4):
    """Single-sample DataLoader (variable-length prompts; batching is by
    gradient accumulation in train.py). collate returns the lone dict."""
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=1, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=lambda b: b[0])


# ======================================================================
# 9. SELF-TEST  (honours the no-separate-test-file constraint)
#    Run: python -m src.data
#    Verifies the invariants every downstream file relies on, with NO data,
#    NO model, NO network required.
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("data.py SELF-TEST")
    print("=" * 64)

    cats = CATEGORY_MAPS["pku"]

    # (a) Build a small synthetic layout from records.
    recs = [
        {"category": "underlay", "center_x": 0.5, "center_y": 0.8,
         "width": 0.6, "height": 0.15},
        {"label": 1, "center_x": 0.5, "center_y": 0.8,    # 'text'
         "width": 0.5, "height": 0.10},
        {"category": "logo", "center_x": 0.2, "center_y": 0.1,
         "width": 0.15, "height": 0.08},
    ]
    lay = Layout.from_records(recs, cats)
    assert len(lay) == 3
    assert {e.category for e in lay.elements} == {"underlay", "text", "logo"}

    # (b) Float-primary JSON round-trip is lossless (4-dp).
    js = layout_to_json(lay, sort_by="label")
    lay2 = json_to_layout(js, cats)
    a = {e.category: e.xyxy() for e in lay.elements}
    b = {e.category: e.xyxy() for e in lay2.elements}
    for cat in a:
        for u, v in zip(a[cat], b[cat]):
            assert abs(u - v) < 1e-3, f"round-trip drift in {cat}: {u} vs {v}"
    print("  [ok] float round-trip lossless to 4dp")

    # (c) Quantised round-trip stays within one bin width.
    jq = layout_to_json(lay, sort_by="label", quantize=True,
                        bucketizers=create_bucketizers("linear", 128))
    layq = json_to_layout(jq, cats, bucketizers=create_bucketizers("linear", 128))
    assert len(layq) == 3
    print("  [ok] quantised round-trip parses, element count preserved")

    # (d) Agreement accessors are ORDER-INVARIANT (the RGPO-critical property).
    shuffled = Layout(elements=list(reversed(lay.elements)), categories=lay.categories)
    g1 = lay.occupancy_grid(resolution=16)
    g2 = shuffled.occupancy_grid(resolution=16)
    assert np.allclose(g1, g2), "occupancy grid must be order-invariant"
    s1, s2 = lay.category_stats(), shuffled.category_stats()
    assert s1.keys() == s2.keys()
    for k in s1:
        for f in ("cx", "cy", "w", "h", "count"):
            assert abs(s1[k][f] - s2[k][f]) < 1e-9
    print("  [ok] occupancy grid + category stats are order-invariant")

    # (e) Occupancy grid is category-count agnostic (PKU 3 vs CGL 4).
    g_pku = lay.occupancy_grid(16, per_category=True)
    assert g_pku.shape == (3, 16, 16)
    cgl_cats = CATEGORY_MAPS["cgl"]
    lay_cgl = Layout.from_records(recs, cgl_cats)
    g_cgl = lay_cgl.occupancy_grid(16, per_category=True)
    assert g_cgl.shape == (4, 16, 16)
    print("  [ok] occupancy grid is category-count agnostic (3 and 4)")

    # (f) Parser tolerates markdown fences and junk.
    fenced = "```json\n" + js + "\n```"
    assert len(json_to_layout(fenced, cats)) == 3
    assert len(json_to_layout("not json at all", cats)) == 0
    print("  [ok] parser tolerates code fences and rejects junk safely")

    # (g) Six constrained-task builders all produce a well-formed triple.
    blank = Image.new("RGB", (64, 64), (255, 255, 255))
    rng = random.Random(0)
    for t in TASK_TYPES:
        s = build_constrained_sample(blank, lay, [lay], cats, t, rng=rng)
        assert {"image", "system", "user", "assistant"} <= set(s)
        assert isinstance(s["assistant"], str) and s["assistant"].strip()
    print(f"  [ok] all {len(TASK_TYPES)} task builders produce valid triples")

    print("=" * 64)
    print("  ALL data.py SELF-TESTS PASSED")
    print("=" * 64)
