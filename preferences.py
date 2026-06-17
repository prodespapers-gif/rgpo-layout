"""
preferences.py — Candidate Sampling & Preference-Pair Construction
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [6] of 10. The integration heart: model + rgpo + losses converge here.

What this file does
-------------------
Stage 2 (preference alignment) needs (chosen, rejected, delta) triples. This
file builds them:
  1. Sample N candidate layouts per canvas with the SFT model
     (``model.generate_layout`` -> ``Layout`` objects, file [3]'s contract).
  2. Rank those candidates and assign each pair a normalised margin delta, using
     the selected preference SOURCE from rgpo.py (file [4]).
  3. Emit ``(image, system, user, chosen_json, rejected_json, score_margin)``
     records that model.forward_dpo + losses.DMPOLoss consume (files [3], [5]).

THE single A/B/C divergence point
---------------------------------
Everywhere else in the codebase is identical across the three experimental arms.
HERE is the one place they differ: ``build_preference_source(pref_source, cfg)``
returns source A (learned reward), B (RGPO retrieval-agreement), or C (hybrid),
and that object both ranks the candidates and assigns delta. train.py passes
``--pref-source {A,B,C}`` straight through. losses.py never learns which arm it
is in; the loss is byte-identical across arms (losses.py self-test (d)).

Why each arm forms its OWN pairs (a deliberate methodological choice)
---------------------------------------------------------------------
The candidate POOL is the fixed substrate (same SFT model, same seeds, same
canvases) -> the source is the ONLY variable. RGPO's thesis is that the
retrieval neighbourhood defines the preference signal, and that includes WHICH
layout is preferred, not merely by how much. If we forced a single shared
pairing across arms, the RGPO arm would still depend on a reward model to pick
the pairs, contaminating the control. So by default the ranking source and the
delta source are the SAME object.

The rank-role and the delta-role are nevertheless separated at the function
level (``build_pairs_for_canvas`` takes ``rank_source`` and an optional distinct
``delta_source``). This costs nothing in the main experiment (they are the same
object) but hands a reviewer the pure delta-isolation ablation for free: fix the
pairing with one source and vary only the delta source.

Fairness to the baseline (recorded requirement, now enforced)
-------------------------------------------------------------
Source A's occlusion/readability terms need a canvas and a saliency map; without
them they silently read zero and A emits a near-zero, useless signal (observed
repeatedly in the rgpo/model integration tests). So ``ScoringContext`` is ALWAYS
populated with the canvas array and the saliency map here, giving source A its
best shot. Source B ignores both fields by design.

Dependency direction
--------------------
preferences.py imports FROM data.py (Layout, serialisation), rgpo.py (sources,
ScoringContext), and calls duck-typed ``model.generate_layout``. It does not
import train.py. data.py's old ``DPOPairDataset`` was removed precisely so this
logic could live here with scoring delegated to rgpo.py.

References
----------
  - Candidate sampling for preference data: AesthetiQ (Patnaik et al., CVPR 2025)
  - DMPO consumer of delta: Uni-Layout (Lu et al., ACM MM 2025)
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Sequence

import numpy as np

from data import Layout, layout_to_json, load_config  # noqa: F401
from rgpo import PreferenceSource, ScoringContext, build_preference_source

logger = logging.getLogger(__name__)


# ======================================================================
# 0. SEEDING  (guarded so the module/self-test load without torch)
# ======================================================================

def _set_seed(seed: int):
    """Best-effort deterministic sampling; a no-op if torch is absent.

    Seeding is purely for reproducibility, so a missing OR partial torch must
    never abort pair construction — we swallow any error from the torch call
    and still seed the stdlib / numpy RNGs.
    """
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # torch absent, or a stub without manual_seed
        pass
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


# ======================================================================
# 1. CANDIDATE SAMPLING
#    Draw N layouts from the SFT model for one canvas. Returns Layout objects
#    (model.generate_layout's contract), dropping empties so the scorer never
#    sees a degenerate candidate.
# ======================================================================

def sample_candidates(model, sample: dict, num_candidates: int,
                      temperature: float, top_p: float,
                      seed_base: int = 0) -> list[Layout]:
    """Generate up to ``num_candidates`` non-empty candidate ``Layout``s.

    Each draw is independently seeded (``seed_base + i``) for reproducibility.
    The model is asked one candidate at a time so a single malformed generation
    cannot abort the batch. ``model.generate_layout`` may return a Layout or a
    list (if the backend batches); both are handled.
    """
    cands: list[Layout] = []
    for i in range(num_candidates):
        _set_seed(seed_base + i)
        out = model.generate_layout(sample, temperature=temperature, top_p=top_p)
        for lay in (out if isinstance(out, list) else [out]):
            if isinstance(lay, Layout) and not lay.is_empty():
                cands.append(lay)
    return cands


# ======================================================================
# 2. PAIR CONSTRUCTION FOR ONE CANVAS
#    Rank candidates by `rank_source.quality`, pair best-vs-worst,
#    2nd-best-vs-2nd-worst, ..., and weight each pair by `delta_source`.
# ======================================================================

def build_pairs_for_canvas(candidates: Sequence[Layout],
                           rank_source: PreferenceSource,
                           ctx: ScoringContext,
                           max_pairs: Optional[int] = None,
                           min_margin: float = 1e-3,
                           delta_source: Optional[PreferenceSource] = None
                           ) -> list[tuple]:
    """Return a list of ``(chosen, rejected, delta)`` for one canvas.

    Args:
        candidates: non-empty candidate Layouts for this canvas.
        rank_source: source whose ``quality`` orders the candidates (decides
            which layout is chosen vs rejected).
        ctx: ScoringContext (neighbours + canvas + saliency) for this canvas.
        max_pairs: cap on pairs emitted (default floor(n/2): best-vs-worst,
            2nd-vs-2nd-worst, ...).
        min_margin: drop pairs whose delta falls below this (near-ties add noise,
            not signal).
        delta_source: source whose quality gap defines delta. Defaults to
            ``rank_source`` (the faithful, same-source design). Pass a DIFFERENT
            source only for the delta-isolation ablation.

    The delta is the quality gap divided by the delta-source's ``span`` and
    clipped into (0, 1] — computed from the SAME ``quality`` function used for
    ranking (recomputed here, not cached), so when ``delta_source`` is
    ``rank_source`` the sign of delta is always consistent with the chosen/
    rejected ordering. (The span-normalisation is the fixed-denominator control;
    see rgpo.py.)
    """
    delta_source = delta_source or rank_source
    pool = [c for c in candidates if isinstance(c, Layout) and not c.is_empty()]
    if len(pool) < 2:
        return []

    ranked = sorted(pool, key=lambda c: rank_source.quality(c, ctx), reverse=True)
    n_pairs = len(ranked) // 2
    if max_pairs is not None:
        n_pairs = min(n_pairs, max_pairs)

    span = max(delta_source.span, 1e-8)
    out: list[tuple] = []
    for k in range(n_pairs):
        hi = ranked[k]
        lo = ranked[-(k + 1)]
        gap = delta_source.quality(hi, ctx) - delta_source.quality(lo, ctx)
        delta = float(np.clip(gap / span, 1e-6, 1.0))
        if delta >= min_margin:
            out.append((hi, lo, delta))
    return out


# ======================================================================
# 3. SCORING CONTEXT BUILDER
#    Always populates canvas + saliency (baseline fairness) and the retrieved
#    neighbours (source B's signal).
# ======================================================================

def make_context(sample: dict, dataset, grid_resolution: int) -> ScoringContext:
    """Assemble the ScoringContext for a dataset sample.

    Pulls neighbours from the sample (data.py attaches ``neighbour_layouts``),
    the canvas as a numpy array, and the saliency map via
    ``dataset.load_saliency`` — so BOTH preference families get everything they
    need and source A is never silently starved of its image terms.
    """
    sid = sample["metadata"]["sample_id"]
    neighbours = sample.get("neighbour_layouts", []) or []
    canvas_np = None
    if sample.get("image") is not None:
        canvas_np = np.asarray(sample["image"])
    saliency_np = None
    if hasattr(dataset, "load_saliency"):
        saliency_np = dataset.load_saliency(sid)
    return ScoringContext(
        neighbours=neighbours,
        canvas=canvas_np,
        saliency=saliency_np,
        categories=tuple(getattr(dataset, "categories", ())),
        grid_resolution=grid_resolution,
    )


# ======================================================================
# 4. PREFERENCE-PAIR DATASET
#    Holds the constructed pairs and yields the dicts model.forward_dpo wants.
# ======================================================================

class PreferencePairDataset:
    """In-memory dataset of preference pairs for the alignment loop.

    Each item: ``{image, system, user, chosen, rejected, score_margin}`` where
    chosen/rejected are serialised layout JSON strings (what model.forward_dpo
    re-tokenises) and score_margin is the normalised delta in (0, 1].
    """

    def __init__(self, pairs: list[dict], config: dict):
        self.pairs = pairs
        self.config = config
        logger.info("PreferencePairDataset: %d pairs", len(pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        return {
            "image": p["image"],
            "system": p["system"],
            "user": p["user"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
            "score_margin": p["score_margin"],
        }


# ======================================================================
# 5. DRIVER  (build the whole preference dataset)
# ======================================================================

def build_preference_dataset(model, dataset, config: dict,
                             pref_source: Optional[str] = None,
                             num_pairs: Optional[int] = None,
                             delta_source_key: Optional[str] = None
                             ) -> PreferencePairDataset:
    """Sample candidates over the dataset and build the preference pairs.

    Args:
        model: SFT-trained model exposing ``generate_layout(sample, **kw)``.
        dataset: a PosterLayoutDataset (provides samples + ``load_saliency``).
        config: full config; reads ``train_dpo`` and ``rgpo`` blocks.
        pref_source: "A" | "B" | "C". Defaults to ``config['rgpo']['pref_source']``.
            THIS is the only knob that differs across the experiment's arms.
        num_pairs: target pair count (defaults to config).
        delta_source_key: optional distinct source for delta (ablation only);
            defaults to the same as ``pref_source``.

    Returns:
        PreferencePairDataset ready for the DMPO loop.
    """
    rgpo_cfg = config.get("rgpo", {})
    dpo_cfg = config["train_dpo"]
    pref_source = pref_source or rgpo_cfg.get("pref_source", "B")

    rank_source = build_preference_source(pref_source, config)
    delta_source = (build_preference_source(delta_source_key, config)
                    if delta_source_key else rank_source)

    N = dpo_cfg["num_candidates_per_canvas"]
    target = num_pairs or dpo_cfg["num_preference_pairs"]
    temperature = dpo_cfg["temperature_sampling"]
    top_p = dpo_cfg["top_p_sampling"]
    max_pairs = dpo_cfg.get("max_pairs_per_canvas", None)
    min_margin = dpo_cfg.get("min_margin", 1e-3)
    grid_res = rgpo_cfg.get("grid_resolution", 16)
    sort_by = config["dataset"].get("sort_by", "label")
    seed = dpo_cfg.get("seed", 42)

    logger.info("Building preference pairs: source=%s, N=%d/canvas, target=%d",
                pref_source, N, target)

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    pairs: list[dict] = []
    canvases_used = 0
    for idx in indices:
        if len(pairs) >= target:
            break
        sample = dataset[idx]
        ctx = make_context(sample, dataset, grid_res)

        # Skip canvases with no retrieved neighbours for EVERY arm, not just
        # B/C. Source B/C cannot score without neighbours, but the cleaner and
        # more important reason is the CONTROL: A, B, and C must train on the
        # IDENTICAL set of canvases, or the arms differ in their training
        # distribution and the comparison is no longer "everything fixed except
        # the preference source". Annotated training canvases virtually always
        # have neighbours, so this drops only rare degenerate cases — and drops
        # them equally for all three arms.
        if not ctx.neighbours:
            continue

        candidates = sample_candidates(
            model, sample, N, temperature, top_p, seed_base=idx * N)
        canvas_pairs = build_pairs_for_canvas(
            candidates, rank_source, ctx, max_pairs=max_pairs,
            min_margin=min_margin, delta_source=delta_source)
        canvases_used += 1 if canvas_pairs else 0

        for hi, lo, delta in canvas_pairs:
            pairs.append({
                "image": sample["image"],
                "system": sample["system"],
                "user": sample["user"],
                "chosen": layout_to_json(hi, sort_by=sort_by),
                "rejected": layout_to_json(lo, sort_by=sort_by),
                "score_margin": delta,
            })
            if len(pairs) >= target:
                break

    logger.info("Built %d pairs from %d canvases (source=%s)",
                len(pairs), canvases_used, pref_source)
    return PreferencePairDataset(pairs[:target], config)


# ======================================================================
# 6. SELF-TEST  (run: python -m src.preferences)
#    No torch, no model, no data. A fake model returns canned Layouts and a
#    fake dataset supplies samples + saliency, so the FULL pipeline runs:
#    sample -> rank -> weight -> pair dicts. Verifies the A/B/C divergence is
#    isolated here and the baseline gets canvas+saliency.
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("preferences.py SELF-TEST  (the A/B/C divergence point)")
    print("=" * 64)

    from data import CATEGORY_MAPS
    cats = CATEGORY_MAPS["pku"]

    def L(recs):
        return Layout.from_records(recs, cats)

    # Retrieved neighbourhood: logo top-left, text+underlay lower-middle.
    neigh = [
        L([{"category": "logo", "center_x": 0.18, "center_y": 0.12, "width": 0.16, "height": 0.08},
           {"category": "text", "center_x": 0.5, "center_y": 0.75, "width": 0.42, "height": 0.10},
           {"category": "underlay", "center_x": 0.5, "center_y": 0.75, "width": 0.5, "height": 0.16}]),
        L([{"category": "logo", "center_x": 0.2, "center_y": 0.1, "width": 0.15, "height": 0.09},
           {"category": "text", "center_x": 0.52, "center_y": 0.78, "width": 0.40, "height": 0.11},
           {"category": "underlay", "center_x": 0.52, "center_y": 0.78, "width": 0.48, "height": 0.17}]),
    ]
    # Three candidates of decreasing agreement with the neighbourhood.
    cand_good = L([{"category": "logo", "center_x": 0.19, "center_y": 0.12, "width": 0.16, "height": 0.08},
                   {"category": "text", "center_x": 0.5, "center_y": 0.75, "width": 0.42, "height": 0.10},
                   {"category": "underlay", "center_x": 0.5, "center_y": 0.75, "width": 0.49, "height": 0.16}])
    cand_mid = L([{"category": "logo", "center_x": 0.3, "center_y": 0.25, "width": 0.16, "height": 0.08},
                  {"category": "text", "center_x": 0.45, "center_y": 0.6, "width": 0.42, "height": 0.10}])
    cand_bad = L([{"category": "text", "center_x": 0.5, "center_y": 0.5, "width": 0.6, "height": 0.3}])

    class FakeModel:
        """Returns the canned candidates in round-robin (ignores sampling args)."""
        def __init__(self, layouts):
            self._layouts = layouts
            self._i = 0
        def generate_layout(self, sample, **kw):
            lay = self._layouts[self._i % len(self._layouts)]
            self._i += 1
            return lay

    class FakeDataset:
        categories = cats
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            return self.samples[i]
        def load_saliency(self, sid):
            return np.zeros((64, 64), dtype=np.float32)

    sample = {
        "image": np.full((64, 64, 3), 127, dtype=np.uint8),
        "system": "sys", "user": "usr",
        "neighbour_layouts": neigh,
        "metadata": {"sample_id": "q"},
    }
    ds = FakeDataset([sample])

    cfg = {
        "dataset": {"name": "pku", "sort_by": "label"},
        "train_dpo": {
            "beta": 0.1, "margin_type": "dynamic",
            "num_candidates_per_canvas": 3, "num_preference_pairs": 10,
            "temperature_sampling": 0.9, "top_p_sampling": 0.95,
            "min_margin": 1e-4, "seed": 0,
            "scorer_weights": {"alpha_occ": 1.0, "alpha_rea": 1.0,
                               "alpha_und": 1.0, "alpha_ove": 0.5, "alpha_align": 0.3},
        },
        "rgpo": {"pref_source": "B", "grid_resolution": 16,
                 "w_spatial": 1.0, "w_structural": 1.0, "rgpo_weight": 0.5},
    }

    # (a) Candidate sampling returns the non-empty Layouts from the model.
    cands = sample_candidates(FakeModel([cand_good, cand_mid, cand_bad]),
                              sample, 3, 0.9, 0.95)
    assert len(cands) == 3 and all(isinstance(c, Layout) for c in cands)
    print("  [ok] sample_candidates returns N non-empty Layouts")

    # (b) make_context ALWAYS fills canvas + saliency (baseline fairness) and neighbours.
    ctx = make_context(sample, ds, grid_resolution=16)
    assert ctx.canvas is not None and ctx.saliency is not None
    assert len(ctx.neighbours) == 2
    print("  [ok] context carries canvas + saliency + neighbours (A is not starved)")

    # (c) Source B ranks the neighbourhood-matching candidate as chosen, delta in (0,1].
    srcB = build_preference_source("B", cfg)
    pairsB = build_pairs_for_canvas([cand_good, cand_mid, cand_bad], srcB, ctx,
                                    min_margin=1e-4)
    assert pairsB, "expected at least one pair"
    hi, lo, dB = pairsB[0]
    assert hi is cand_good and lo is cand_bad and 0 < dB <= 1.0
    print(f"  [ok] source B: chosen=good, rejected=bad, delta={dB:.3f} in (0,1]")

    # (d) THE divergence: same candidate pool, source A vs B differ -> the
    #     source is the only variable. (A has canvas+saliency, so it's fair.)
    srcA = build_preference_source("A", cfg)
    pairsA = build_pairs_for_canvas([cand_good, cand_mid, cand_bad], srcA, ctx,
                                    min_margin=0.0)
    dA = pairsA[0][2] if pairsA else 0.0
    # The two sources produce different margins on the same pool (different signal).
    assert abs(dA - dB) > 1e-6 or (pairsA and pairsA[0][0] is not pairsB[0][0]), \
        "A and B should not be identical signals"
    print(f"  [ok] A vs B differ on the SAME pool (deltaA={dA:.3f} vs deltaB={dB:.3f})")

    # (e) delta-isolation ablation: rank with B, weight with A -> distinct roles.
    pairs_mix = build_pairs_for_canvas([cand_good, cand_mid, cand_bad], srcB, ctx,
                                       min_margin=0.0, delta_source=srcA)
    assert pairs_mix and pairs_mix[0][0] is cand_good  # ranking still B's
    print("  [ok] rank/delta roles separable (free delta-isolation ablation)")

    # (f) End-to-end driver yields a PreferencePairDataset with the right schema.
    fake_model = FakeModel([cand_good, cand_mid, cand_bad])
    pref_ds = build_preference_dataset(fake_model, ds, cfg, pref_source="B")
    assert isinstance(pref_ds, PreferencePairDataset) and len(pref_ds) >= 1
    item = pref_ds[0]
    assert set(item) == {"image", "system", "user", "chosen", "rejected", "score_margin"}
    assert isinstance(item["chosen"], str) and isinstance(item["rejected"], str)
    assert 0 < item["score_margin"] <= 1.0
    print(f"  [ok] driver builds PreferencePairDataset ({len(pref_ds)} pairs), schema OK")

    # (g) chosen JSON parses back to a Layout (forward_dpo will re-tokenise it).
    from data import json_to_layout
    back = json_to_layout(item["chosen"], cats)
    assert isinstance(back, Layout) and not back.is_empty()
    print("  [ok] serialised chosen round-trips to a Layout (forward_dpo-ready)")

    # (h) Switching ONLY pref_source changes the produced margins -> the arm
    #     selection is genuinely the single divergence.
    ds2 = FakeDataset([sample])
    pref_A = build_preference_dataset(FakeModel([cand_good, cand_mid, cand_bad]),
                                      ds2, cfg, pref_source="A")
    mA = pref_A[0]["score_margin"] if len(pref_A) else 0.0
    mB = item["score_margin"]
    assert (mA != mB) or (len(pref_A) != len(pref_ds))
    print(f"  [ok] only pref_source changed -> margins/pairs differ (A={mA:.3f}, B={mB:.3f})")

    print("=" * 64)
    print("  ALL preferences.py SELF-TESTS PASSED")
    print("=" * 64)
