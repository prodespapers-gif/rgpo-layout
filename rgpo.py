"""
rgpo.py — Retrieval-Grounded Preference Optimization   THE CONTRIBUTION
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [4] of 10. This is where the paper's novelty lives.

The one idea
------------
Every prior preference-aligned layout method (AesthetiQ, AAPA, Uni-Layout, ...)
needs a *reward model*: a separately trained network, or a hand-weighted sum of
geometric metrics, that says which of two layouts is better. That reward model
is the part everyone hand-builds and argues about.

RGPO removes it. For a given canvas we have already retrieved its K nearest
**real human designs** (retrieval.py). Those neighbours are not just prompt
context - they are a sufficient statistic for "what a good layout on THIS canvas
looks like." So we define preference directly as *agreement with the retrieval
neighbourhood*. The retrieval set that the model already conditions on becomes
the preference signal. No reward model. No learned weights. No human ratings.

Thesis (what a reviewer should take away)
-----------------------------------------
For ANY retrieval-augmented structured generator, the retrieval neighbourhood
can supply the preference signal for free. Posters are the testbed; the
mechanism is general. This is the learning-systems claim the paper makes, and
this file is its operationalisation.

How the claim is tested (the A/B/C control)
-------------------------------------------
The central experiment holds EVERYTHING fixed (same SFT model, same DMPO loss,
same candidate sampler, same pairs) and varies only the *source* of the
preference signal delta:

  Source A - LearnedRewardScorer : delta from the 5-term hand-weighted composite
             reward S (the prior-art reward model). This is the baseline.
  Source B - RGPOScorer          : delta from neighbourhood agreement. No reward
             model at all. This is the contribution.
  Source C - HybridScorer        : delta from a convex blend of A and B. Hedge.

All three implement ONE interface - ``PreferenceSource.delta(winner, loser,
ctx)`` -> delta in (0, 1] - so swapping them is a one-line change and the
comparison is a clean control. losses.py never learns which source produced
delta; preferences.py is the single place the three diverge.

Why delta in (0, 1] and why it is computed HERE
-----------------------------------------------
DMPO's dynamic margin is f(delta) = 2*sinh(delta), and DMPOLoss expects an
ALREADY normalised delta in (0, 1] (see losses.py). Normalisation is a property
of the *scorer*, not the loss - each source knows its own attainable gap and
divides by it (the "fixed-denominator control"). Centralising delta here is
exactly what keeps losses.py scorer-agnostic.

Order-invariance (guards a referee attack)
------------------------------------------
The agreement score is built ONLY from order-invariant Layout accessors
(``occupancy_grid`` and ``category_stats`` from data.py). It never reads element
order. So source B cannot be dismissed as a serialisation/sort-order artifact -
the preference signal depends on geometry alone. (data.py's self-test proves
those accessors are permutation-invariant; this file's self-test re-checks it at
the agreement-score level.)

References
----------
  - DMPO dynamic margin: Lu et al. (Uni-Layout), ACM MM 2025, Eq. 6-7
  - Composite reward terms (Occ/Rea/Und/Ove/Align): RALF (Horita et al.,
    CVPR 2024); PosterLayout (Hsu et al., CVPR 2023); LayoutGAN++ (TVCG 2020)
  - DPO implicit reward / self-rewarding framing (contrast): Rafailov et al.,
    NeurIPS 2023; Yuan et al., 2024 - none derive preference from a retrieval
    neighbourhood, which is what makes RGPO distinct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from data import Layout

logger = logging.getLogger(__name__)


# ======================================================================
# 0. CONTEXT OBJECT
#    Everything a preference source might need to turn a candidate Layout
#    into a scalar quality. Sources use only the subset they care about:
#    source B needs `neighbours`; source A needs `canvas`/`saliency`.
# ======================================================================

@dataclass
class ScoringContext:
    """Per-canvas information shared by all preference sources.

    Attributes:
        neighbours: the retrieved real-design Layouts for THIS canvas
            (retrieval.py's ``neighbour_layouts``). The preference signal of
            source B is defined entirely against these.
        canvas: optional (H, W, 3) uint8 canvas, for the readability term of
            source A. Not needed by source B.
        saliency: optional (H, W) float saliency in [0, 1], for the occlusion
            term of source A. Not needed by source B.
        categories: the category vocabulary (PKU 3 / CGL 4). Sources are
            category-count agnostic; this is used only for metric bookkeeping.
        grid_resolution: G for the occupancy grid used by source B.
    """
    neighbours: list = field(default_factory=list)
    canvas: Optional[np.ndarray] = None
    saliency: Optional[np.ndarray] = None
    categories: tuple = ()
    grid_resolution: int = 16


# ======================================================================
# 1. GEOMETRIC METRICS  (the prior-art reward terms; used by SOURCE A)
#    Order-invariant. Operate on Layout objects (file [1]'s type). These are
#    faithful reimplementations of the five terms the original composite reward
#    used, lifted onto the Layout API so source A and the evaluator share them.
# ======================================================================

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def occlusion(layout: Layout, saliency: Optional[np.ndarray]) -> float:
    """Mean saliency under the union of element boxes (Occ down)."""
    if layout.is_empty() or saliency is None:
        return 0.0
    H, W = saliency.shape
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in layout.boxes_xyxy():
        mask[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = True
    covered = saliency[mask]
    return float(covered.mean()) if covered.size else 0.0


def readability(layout: Layout, canvas: Optional[np.ndarray]) -> float:
    """Mean gradient energy under text, minus underlay-backed regions (Rea down)."""
    if layout.is_empty() or canvas is None:
        return 0.0
    gray = canvas.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gray /= 255.0
    H, W = gray.shape
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    grad = (gx + gy) / 2.0
    mask = np.zeros((H, W), dtype=np.float32)
    for e in layout.elements:
        if e.category == "text":
            x1, y1, x2, y2 = e.xyxy()
            mask[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 1.0
    for e in layout.elements:
        if e.category == "underlay":
            x1, y1, x2, y2 = e.xyxy()
            mask[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 0.0
    vals = grad[mask > 0]
    return float(vals.mean()) if vals.size else 0.0


def underlay_effectiveness(layout: Layout) -> float:
    """Fraction of underlays that fully contain some non-underlay (Und up)."""
    unders = [e for e in layout.elements if e.category == "underlay"]
    others = [e for e in layout.elements if e.category != "underlay"]
    if not unders:
        return 1.0
    valid = 0
    for u in unders:
        ux1, uy1, ux2, uy2 = u.xyxy()
        for ne in others:
            nx1, ny1, nx2, ny2 = ne.xyxy()
            if (nx1 >= ux1 - 0.01 and ny1 >= uy1 - 0.01
                    and nx2 <= ux2 + 0.01 and ny2 <= uy2 + 0.01):
                valid += 1
                break
    return float(valid) / len(unders)


def overlay(layout: Layout) -> float:
    """Mean pairwise IoU among content elements (Ove down).

    Excludes BOTH ``underlay`` and ``embellishment``. The PKU PosterLayout
    metric excludes underlay; the CGL-Dataset-V2 metric (RADM, CIKM 2023, Sec.
    5.2) additionally excludes embellishment, since underlays and embellishments
    are by design attached to / overlapping other elements and would spuriously
    inflate overlap. ``embellishment`` does not exist in PKU's 3-category vocab,
    so this exclusion is a no-op there and the PKU number is unchanged.
    """
    _excluded = ("underlay", "embellishment")
    boxes = [e.xyxy() for e in layout.elements if e.category not in _excluded]
    n = len(boxes)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += _iou_xyxy(np.asarray(boxes[i]), np.asarray(boxes[j]))
            cnt += 1
    return float(tot / cnt) if cnt else 0.0


def alignment(layout: Layout) -> float:
    """Per-axis minimal-residual alignment energy (Align down).

    X-anchors compare only with X-anchors and Y with Y (comparing an x against
    a y is geometrically meaningless), matching the corrected metric in the
    original losses.py.
    """
    elems = layout.elements
    if len(elems) < 2:
        return 0.0
    xa, ya = [], []
    for e in elems:
        x1, y1, x2, y2 = e.xyxy()
        xa.append([x1, e.center_x, x2])
        ya.append([y1, e.center_y, y2])
    xa, ya = np.asarray(xa), np.asarray(ya)
    scores = []
    for i in range(len(elems)):
        best = float("inf")
        for j in range(len(elems)):
            if i == j:
                continue
            for ai in range(3):
                for aj in range(3):
                    best = min(best, abs(xa[i, ai] - xa[j, aj]))
                    best = min(best, abs(ya[i, ai] - ya[j, aj]))
        scores.append(-np.log10(max(1.0 - best, 1e-10)) if best < 1.0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _squash(x: float) -> float:
    """1 - exp(-x): smooth, monotone, bounds open-ended metrics into [0, 1)."""
    return float(1.0 - np.exp(-max(x, 0.0)))


# ======================================================================
# 2. THE NEIGHBOURHOOD AGREEMENT SCORE   the heart of the contribution
#    A candidate is good to the exact extent that it agrees with the retrieved
#    real designs for this canvas. Two order-invariant views of agreement:
#      (i)  spatial: divergence between the candidate's soft occupancy grid and
#           the neighbourhood's MEAN occupancy grid (where mass goes on canvas);
#      (ii) structural: per-category mismatch in how many elements of each type
#           there are and where their centroids/sizes sit.
#    Both use ONLY data.py's permutation-invariant accessors.
# ======================================================================

def _occupancy_divergence(cand: Layout, neighbours: Sequence,
                          resolution: int) -> float:
    """Mean absolute difference between candidate and neighbourhood-mean grids.

    Each grid is soft occupancy in [0, 1] over a GxG lattice. The neighbourhood
    target is the elementwise mean of the neighbour grids - a smooth "where do
    real designs put things on this canvas" heatmap. Lower is better; range
    [0, 1] because both grids are in [0, 1].
    """
    cg = cand.occupancy_grid(resolution=resolution, per_category=False)
    grids = [n.occupancy_grid(resolution=resolution, per_category=False)
             for n in neighbours]
    target = np.mean(grids, axis=0)
    return float(np.mean(np.abs(cg - target)))


def _structural_mismatch(cand: Layout, neighbours: Sequence) -> float:
    """Per-category disagreement in count and centroid/size, vs the neighbourhood.

    For each category we compare the candidate's (count, centroid, mean-size)
    against the neighbourhood average of the same, all order-invariant via
    ``Layout.category_stats``. Differences are in normalised units and squashed
    so the term sits in [0, 1). Categories present in either the candidate or
    the neighbourhood contribute; this is automatically category-count agnostic
    (PKU 3 / CGL 4 / anything), which is what lets the mechanism generalise.
    """
    cand_stats = cand.category_stats()
    # Aggregate neighbour stats per category (mean over neighbours that have it).
    agg: dict = {}
    for n in neighbours:
        for label, st in n.category_stats().items():
            agg.setdefault(label, []).append(st)
    neigh_mean: dict = {}
    for label, sts in agg.items():
        neigh_mean[label] = {
            "cx": float(np.mean([s["cx"] for s in sts])),
            "cy": float(np.mean([s["cy"] for s in sts])),
            "w": float(np.mean([s["w"] for s in sts])),
            "h": float(np.mean([s["h"] for s in sts])),
            "count": float(np.mean([s["count"] for s in sts])),
        }

    labels = set(cand_stats) | set(neigh_mean)
    if not labels:
        return 0.0

    per_cat = []
    for label in labels:
        c = cand_stats.get(label)
        m = neigh_mean.get(label)
        if c is None or m is None:
            # One side expects this category and the other omits it entirely:
            # full positional disagreement for this category.
            per_cat.append(1.0)
            continue
        # Count disagreement, normalised by the expected count (>=1).
        count_term = abs(c["count"] - m["count"]) / max(m["count"], 1.0)
        # Centroid disagreement (L2 in normalised canvas coords; max ~ sqrt(2)).
        centroid_term = np.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"]) / np.sqrt(2.0)
        # Size disagreement (mean abs diff of w and h).
        size_term = 0.5 * (abs(c["w"] - m["w"]) + abs(c["h"] - m["h"]))
        per_cat.append(_squash(count_term + centroid_term + size_term))
    return float(np.mean(per_cat))


def neighbourhood_agreement_score(candidate: Layout,
                                  neighbours: Sequence,
                                  resolution: int = 16,
                                  w_spatial: float = 1.0,
                                  w_structural: float = 1.0) -> float:
    """RGPO quality in [0, 1]: 1 = perfect agreement with the retrieval set.

    quality = 1 - (w_spatial*occ_div + w_structural*struct_mismatch)
                  / (w_spatial + w_structural)

    Both penalty terms are in [0, 1], so the convex combination is in [0, 1] and
    quality is in [0, 1]. This is the *entire* learned-reward-free signal: there
    is no trained network and no hand-tuned metric weight vector here - only two
    interpretable agreement terms with two knobs (vs the five alpha-weights
    source A must set). With no neighbours we return a neutral 0.5 (the caller
    normally guarantees a non-empty neighbour set; see retrieval.py's contract).

    Order-invariant by construction: it reads only ``occupancy_grid`` and
    ``category_stats``, never element order.
    """
    if not neighbours:
        return 0.5
    occ_div = _occupancy_divergence(candidate, neighbours, resolution)
    struct = _structural_mismatch(candidate, neighbours)
    denom = max(w_spatial + w_structural, 1e-8)
    penalty = (w_spatial * occ_div + w_structural * struct) / denom
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


# ======================================================================
# 3. PREFERENCE SOURCES  (A / B / C) - one interface, three signals
#    Each maps a candidate Layout + context to a scalar quality, and turns a
#    (winner, loser) quality gap into a normalised delta in (0, 1] using its OWN
#    attainable-gap denominator (the fixed-denominator control). losses.py only
#    ever sees delta; it never learns which subclass produced it.
# ======================================================================

class PreferenceSource:
    """Abstract preference signal. Subclasses define ``quality`` and ``span``.

    ``quality(cand, ctx)`` - higher is better (any real range).
    ``span``               - the maximum attainable quality gap, used to
                             normalise delta into (0, 1]. Constant per source.
    """

    name: str = "abstract"
    span: float = 1.0

    def quality(self, cand: Layout, ctx: ScoringContext) -> float:
        raise NotImplementedError

    def delta(self, winner: Layout, loser: Layout, ctx: ScoringContext) -> float:
        """Normalised preference margin delta = (q+ - q-)/span, clipped to (0,1].

        The raw gap is divided by this source's own ``span`` so every source
        hands DMPO a delta on the SAME (0, 1] scale - the control that makes the
        A/B/C comparison about the *signal*, not about its numeric range.
        """
        gap = self.quality(winner, ctx) - self.quality(loser, ctx)
        return float(np.clip(gap / max(self.span, 1e-8), 1e-6, 1.0))


class LearnedRewardScorer(PreferenceSource):
    """SOURCE A - the prior-art reward model: a hand-weighted 5-term composite.

    S = -a_occ*Occ - a_rea*Rea + a_und*Und - a_ove*Ove - a_align*Align
    (each term squashed into [0, 1) first). This is exactly the reward every
    prior preference-aligned layout method builds; here it is the BASELINE we
    are trying to beat without a reward model.

    ``span`` = sum(alpha): the attainable range of S is
    [-(a_occ+a_rea+a_ove+a_align), a_und], so the largest possible gap is
    sum(alpha). Dividing by it reproduces the original delta-normalisation
    contract precisely.
    """

    name = "A_learned_reward"

    def __init__(self, weights: dict):
        self.a_occ = weights.get("alpha_occ", 1.0)
        self.a_rea = weights.get("alpha_rea", 1.0)
        self.a_und = weights.get("alpha_und", 1.0)
        self.a_ove = weights.get("alpha_ove", 0.5)
        self.a_align = weights.get("alpha_align", 0.3)
        self.span = (self.a_occ + self.a_rea + self.a_und
                     + self.a_ove + self.a_align)

    def quality(self, cand: Layout, ctx: ScoringContext) -> float:
        occ = _squash(occlusion(cand, ctx.saliency))
        rea = _squash(readability(cand, ctx.canvas))
        und = _squash(underlay_effectiveness(cand))
        ove = _squash(overlay(cand))
        ali = _squash(alignment(cand))
        return (-self.a_occ * occ - self.a_rea * rea + self.a_und * und
                - self.a_ove * ove - self.a_align * ali)


class RGPOScorer(PreferenceSource):
    """SOURCE B - RETRIEVAL-GROUNDED preference.   the contribution

    quality = neighbourhood_agreement_score(cand, ctx.neighbours). No reward
    model, no learned weights, no ratings - the retrieved real designs ARE the
    signal. Two knobs only (w_spatial, w_structural), against source A's five.

    ``span`` = 1.0: agreement quality already lives in [0, 1], so the largest
    possible gap is 1, and delta = gap directly. (No external denominator to
    tune - the bound is intrinsic to the score, which is itself part of the "no
    reward model to calibrate" argument.)
    """

    name = "B_rgpo"
    span = 1.0

    def __init__(self, resolution: int = 16,
                 w_spatial: float = 1.0, w_structural: float = 1.0):
        self.resolution = resolution
        self.w_spatial = w_spatial
        self.w_structural = w_structural

    def quality(self, cand: Layout, ctx: ScoringContext) -> float:
        res = ctx.grid_resolution or self.resolution
        return neighbourhood_agreement_score(
            cand, ctx.neighbours, resolution=res,
            w_spatial=self.w_spatial, w_structural=self.w_structural)


class HybridScorer(PreferenceSource):
    """SOURCE C - convex blend of A and B. The hedge.

    quality = (1-lam)*A_hat + lam*B_hat, where A_hat is source A's quality
    rescaled to [0, 1] by its own span (so the two live on a common scale before
    mixing) and B_hat is the agreement quality (already [0, 1]). lam =
    ``rgpo_weight``.

    ``span`` = 1.0 because both mixed components are in [0, 1]. lam=0 recovers A
    (up to the monotone rescale), lam=1 recovers B; the sweep over lam is a
    clean interpolation between "reward model" and "retrieval-grounded".
    """

    name = "C_hybrid"
    span = 1.0

    def __init__(self, learned: LearnedRewardScorer, rgpo: RGPOScorer,
                 rgpo_weight: float = 0.5):
        self.learned = learned
        self.rgpo = rgpo
        self.lam = float(np.clip(rgpo_weight, 0.0, 1.0))
        # Map source A's signed quality (range [-a_min, a_und]) into [0, 1].
        self._a_min = -(learned.a_occ + learned.a_rea
                        + learned.a_ove + learned.a_align)
        self._a_max = learned.a_und
        self._a_rng = max(self._a_max - self._a_min, 1e-8)

    def quality(self, cand: Layout, ctx: ScoringContext) -> float:
        a_raw = self.learned.quality(cand, ctx)
        a_hat = (a_raw - self._a_min) / self._a_rng       # -> [0, 1]
        b_hat = self.rgpo.quality(cand, ctx)              # already [0, 1]
        return (1.0 - self.lam) * a_hat + self.lam * b_hat


# ======================================================================
# 4. FACTORY  (the single switch the A/B/C experiment flips)
# ======================================================================

def build_preference_source(pref_source: str, config: dict) -> PreferenceSource:
    """Construct preference source A, B, or C from config.

    This is the ONLY place the experiment selects the signal. train.py passes
    ``--pref-source {A,B,C}`` straight through to here; preferences.py then uses
    the returned object's ``delta`` to label pairs. Everything downstream
    (losses.py, the optimiser) is identical across A/B/C.
    """
    key = pref_source.strip().upper()
    weights = config["train_dpo"]["scorer_weights"]
    rgpo_cfg = config.get("rgpo", {})
    res = rgpo_cfg.get("grid_resolution", 16)
    w_sp = rgpo_cfg.get("w_spatial", 1.0)
    w_st = rgpo_cfg.get("w_structural", 1.0)

    if key == "A":
        return LearnedRewardScorer(weights)
    if key == "B":
        return RGPOScorer(resolution=res, w_spatial=w_sp, w_structural=w_st)
    if key == "C":
        learned = LearnedRewardScorer(weights)
        rgpo = RGPOScorer(resolution=res, w_spatial=w_sp, w_structural=w_st)
        return HybridScorer(learned, rgpo,
                            rgpo_weight=rgpo_cfg.get("rgpo_weight", 0.5))
    raise ValueError(f"Unknown preference source '{pref_source}'. Use A, B, or C.")


# ======================================================================
# 5. SELF-TEST  (run: python -m src.rgpo)
#    No torch, no model, no data - pure Layout geometry. Verifies the
#    properties the paper's claims rest on.
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("rgpo.py SELF-TEST  (the contribution)")
    print("=" * 64)

    cats = ("logo", "text", "underlay")

    def L(recs):
        return Layout.from_records(recs, cats)

    # A small "neighbourhood" of three similar real designs: logo top-left,
    # text+underlay lower-middle.
    neigh = [
        L([{"category": "logo", "center_x": 0.18, "center_y": 0.12, "width": 0.16, "height": 0.08},
           {"category": "text", "center_x": 0.5, "center_y": 0.75, "width": 0.42, "height": 0.10},
           {"category": "underlay", "center_x": 0.5, "center_y": 0.75, "width": 0.5, "height": 0.16}]),
        L([{"category": "logo", "center_x": 0.20, "center_y": 0.10, "width": 0.15, "height": 0.09},
           {"category": "text", "center_x": 0.52, "center_y": 0.78, "width": 0.40, "height": 0.11},
           {"category": "underlay", "center_x": 0.52, "center_y": 0.78, "width": 0.48, "height": 0.17}]),
        L([{"category": "logo", "center_x": 0.16, "center_y": 0.14, "width": 0.17, "height": 0.08},
           {"category": "text", "center_x": 0.48, "center_y": 0.72, "width": 0.44, "height": 0.10},
           {"category": "underlay", "center_x": 0.48, "center_y": 0.72, "width": 0.5, "height": 0.15}]),
    ]
    ctx = ScoringContext(neighbours=neigh, categories=cats, grid_resolution=16)

    # A candidate that closely matches the neighbourhood ...
    good = L([{"category": "logo", "center_x": 0.19, "center_y": 0.12, "width": 0.16, "height": 0.08},
              {"category": "text", "center_x": 0.50, "center_y": 0.75, "width": 0.42, "height": 0.10},
              {"category": "underlay", "center_x": 0.50, "center_y": 0.75, "width": 0.49, "height": 0.16}])
    # ... and one that violates it (everything dumped center, no underlay).
    bad = L([{"category": "logo", "center_x": 0.5, "center_y": 0.5, "width": 0.5, "height": 0.3},
             {"category": "text", "center_x": 0.5, "center_y": 0.5, "width": 0.6, "height": 0.3}])

    # (a) Agreement score ranks the good candidate above the bad one, both in [0,1].
    q_good = neighbourhood_agreement_score(good, neigh)
    q_bad = neighbourhood_agreement_score(bad, neigh)
    assert 0.0 <= q_bad < q_good <= 1.0, (q_bad, q_good)
    print(f"  [ok] agreement ranks good>bad and stays in [0,1]  "
          f"(good={q_good:.3f}, bad={q_bad:.3f})")

    # (b) ORDER-INVARIANCE: shuffling candidate elements cannot change the score.
    shuffled = Layout(elements=list(reversed(good.elements)), categories=good.categories)
    assert abs(neighbourhood_agreement_score(shuffled, neigh) - q_good) < 1e-12
    # also invariant to neighbour order
    assert abs(neighbourhood_agreement_score(good, list(reversed(neigh))) - q_good) < 1e-12
    print("  [ok] agreement is invariant to candidate AND neighbour ordering")

    # (c) A near-perfect match scores ~1; a real neighbour vs its own set is high.
    q_identity = neighbourhood_agreement_score(neigh[0], neigh)
    assert q_identity > 0.7, q_identity
    print(f"  [ok] a real neighbour scores high against its own set ({q_identity:.3f})")

    # (d) SOURCE B delta in (0,1], and prefers good over bad (positive delta).
    B = RGPOScorer()
    dB = B.delta(good, bad, ctx)
    assert 0 < dB <= 1.0 and dB > 1e-3, dB
    # reversed pair clamps to the tiny floor (loser preferred -> ~0)
    dB_rev = B.delta(bad, good, ctx)
    assert dB_rev <= 1e-3 + 1e-9
    print(f"  [ok] source B yields valid delta in (0,1] and is sign-correct (delta={dB:.3f})")

    # (e) SOURCE A delta in (0,1] with span = sum(alpha) (original contract).
    weights = {"alpha_occ": 1.0, "alpha_rea": 1.0, "alpha_und": 1.0,
               "alpha_ove": 0.5, "alpha_align": 0.3}
    A = LearnedRewardScorer(weights)
    assert abs(A.span - 3.8) < 1e-9, A.span
    dA = A.delta(good, bad, ctx)
    assert 0 < dA <= 1.0
    print(f"  [ok] source A span=sum(alpha)=3.8 and delta in (0,1]  (delta={dA:.3f})")

    # (f) SOURCE C interpolates: lam=0 ~ A-ranking, lam=1 == B exactly.
    C0 = HybridScorer(A, B, rgpo_weight=0.0)
    C1 = HybridScorer(A, B, rgpo_weight=1.0)
    # lam=1 hybrid quality equals B quality
    assert abs(C1.quality(good, ctx) - B.quality(good, ctx)) < 1e-12
    # lam=0 hybrid preserves A's ordering of good vs bad
    assert (C0.quality(good, ctx) - C0.quality(bad, ctx)) * \
           (A.quality(good, ctx) - A.quality(bad, ctx)) > 0
    print("  [ok] source C interpolates A<->B (lam=1 == B, lam=0 preserves A order)")

    # (g) FACTORY returns the right types and one common interface.
    cfg = {"train_dpo": {"scorer_weights": weights},
           "rgpo": {"grid_resolution": 16, "w_spatial": 1.0,
                    "w_structural": 1.0, "rgpo_weight": 0.5}}
    assert isinstance(build_preference_source("A", cfg), LearnedRewardScorer)
    assert isinstance(build_preference_source("B", cfg), RGPOScorer)
    assert isinstance(build_preference_source("C", cfg), HybridScorer)
    for key in ("A", "B", "C"):
        d = build_preference_source(key, cfg).delta(good, bad, ctx)
        assert 0 < d <= 1.0
    print("  [ok] factory builds A/B/C, all expose delta in (0,1] via one interface")

    # (h) Category-count agnosticism: same logic runs on CGL's 4 categories.
    cats4 = ("logo", "text", "underlay", "embellishment")
    neigh4 = [Layout.from_records(n.to_records(), cats4) for n in neigh]
    good4 = Layout.from_records(good.to_records(), cats4)
    q4 = neighbourhood_agreement_score(good4, neigh4)
    assert 0.0 <= q4 <= 1.0
    print(f"  [ok] agreement runs unchanged on 4-category CGL layouts ({q4:.3f})")

    print("=" * 64)
    print("  ALL rgpo.py SELF-TESTS PASSED")
    print("=" * 64)
