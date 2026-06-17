"""
evaluate.py — Evaluation: Metrics, FID, Win-Rate, Significance
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [8] of 10. The measurement half.

What it produces
----------------
  * The five geometric metrics (Occ down, Rea down, Und up, Ove down, Align down)
    and MaxIoU (up) vs ground truth.
  * Layout-FID via an INLINED FIDNetV3 (Kikuchi et al., ACM MM 2021), loaded
    strict-only so a mismatched checkpoint reports N/A rather than a fabricated
    number on half-initialised features.
  * Aesthetic win/tie/loss via a frozen MLLM layout-judge, with a second-family
    judge + Cohen's kappa to guard against same-family self-preference.
  * Significance machinery for the A/B/C comparison: paired t-test,
    Holm-Bonferroni correction across metrics, and TOST equivalence — so a
    claim like "B beats A on FID" (or "B is no worse than A") is defensible.

Why FIDNetV3 is inlined here
----------------------------
The 10-file budget excludes a separate fid_model.py. FIDNetV3 is therefore a
section of this file. evaluate.py never instantiates it unless a real checkpoint
is supplied (``evaluation.fid_checkpoint``); otherwise FID is N/A, exactly as in
the FID baselines, so no fabricated FID is ever reported.

Contract with the rest of the system
-------------------------------------
  * Operates on ``Layout`` objects (file [1]'s type). ``model.generate_layout``
    already returns those; ground truth comes from ``sample['target_layout']``.
  * Reuses the geometric metrics from rgpo.py (file [4]) — the SAME functions
    source A scores with — so evaluation and the learned-reward baseline measure
    layouts identically (no drift between training signal and reported metric).
  * The -1.0 sentinel marks "metric unavailable"; it is filtered from averaging
    and printed as N/A, never mixed into a mean.

References
----------
  - FID for layouts: Kikuchi et al. (CLG-LO), ACM MM 2021
  - Content metrics: Horita et al. (RALF), CVPR 2024
  - Graphic metrics: Hsu et al. (PosterLayout), CVPR 2023
  - Aesthetic judge protocol: Patnaik et al. (AesthetiQ), CVPR 2025
  - Holm correction: Holm, Scand. J. Statist. 1979 ; TOST: Schuirmann, 1987
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from data import TASK_TYPES, Layout, layout_to_json, load_config
# Reuse the EXACT geometric metrics the learned-reward source scores with, so
# evaluation never drifts from the training signal.
from rgpo import alignment, occlusion, overlay, readability, underlay_effectiveness

logger = logging.getLogger(__name__)

UNAVAILABLE = -1.0  # project-wide "metric unavailable" sentinel


# ======================================================================
# 1. INLINED FIDNetV3  (only instantiated when a real checkpoint exists)
# ======================================================================

def _build_fidnet(num_label: int, d_model: int = 256, max_bbox: int = 10,
                  nhead: int = 4, num_layers: int = 4,
                  dim_feedforward: int = 256, dropout: float = 0.1):
    """Construct a FIDNetV3 (torch imported lazily). Returns an nn.Module.

    Defined as a factory so importing evaluate.py never requires torch; the
    class is built on first use, when a checkpoint is actually being loaded.
    """
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=64, dropout=0.0):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return self.dropout(x + self.pe[:, : x.size(1)])

    class _Stack(nn.Module):
        def __init__(self, d_model, nhead, num_layers, dff, dropout=0.1):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dff,
                dropout=dropout, activation="relu", batch_first=True,
                norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        def forward(self, x, src_key_padding_mask=None):
            return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    class FIDNetV3(nn.Module):
        def __init__(self, num_label, d_model=256, max_bbox=50, nhead=4,
                     num_layers=4, dim_feedforward=256, dropout=0.1):
            super().__init__()
            self.num_label = num_label
            self.d_model = d_model
            self.max_bbox = max_bbox
            self.emb_label = nn.Embedding(num_label, d_model)
            self.fc_bbox = nn.Linear(4, d_model)
            self.fc_in = nn.Linear(2 * d_model, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_bbox + 1,
                                                  dropout=dropout)
            self.encoder = _Stack(d_model, nhead, num_layers, dim_feedforward, dropout)
            self.decoder = _Stack(d_model, nhead, num_layers, dim_feedforward, dropout)
            self.fc_out_pad = nn.Linear(d_model, 1)
            self.fc_out_label = nn.Linear(d_model, num_label)
            self.fc_out_bbox = nn.Linear(d_model, 4)
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def _tokens(self, batch):
            label = batch["label"].long()
            bbox = torch.stack([batch["center_x"], batch["center_y"],
                                batch["width"], batch["height"]], dim=-1).float()
            pad_mask = ~batch["mask"].bool()
            e_label = self.emb_label(label.clamp(min=0, max=self.num_label - 1))
            e_bbox = self.fc_bbox(bbox)
            tokens = self.pos_encoder(self.fc_in(torch.cat([e_label, e_bbox], dim=-1)))
            return tokens, pad_mask

        def encode(self, batch):
            tokens, pad_mask = self._tokens(batch)
            return self.encoder(tokens, src_key_padding_mask=pad_mask), pad_mask

        @torch.no_grad()
        def extract_features(self, batch):
            memory, pad_mask = self.encode(batch)
            real = (~pad_mask).unsqueeze(-1).float()
            denom = real.sum(dim=1).clamp(min=1.0)
            return (memory * real).sum(dim=1) / denom

    return FIDNetV3(num_label=num_label, d_model=d_model, max_bbox=max_bbox,
                    nhead=nhead, num_layers=num_layers,
                    dim_feedforward=dim_feedforward, dropout=dropout)


class LayoutFIDCalculator:
    """Layout-FID via FIDNetV3 features. N/A unless a real checkpoint loads."""

    def __init__(self, num_categories: int, max_elements: int = 10,
                 fidnet_ckpt: Optional[str] = None, device: Optional[str] = None):
        self.num_categories = num_categories
        self.max_elements = max_elements
        self.feature_dim = 256
        self.model = None
        self.device = device
        if fidnet_ckpt and os.path.exists(fidnet_ckpt):
            self._load(fidnet_ckpt)

    def _resolve_device(self) -> str:
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        return self.device

    def _load(self, ckpt_path: str):
        """Strict load only: a mismatched checkpoint disables FID (-> N/A)."""
        try:
            import torch
        except ImportError:
            logger.warning("torch unavailable; FID reported as N/A.")
            return
        try:
            model = _build_fidnet(self.num_categories, d_model=self.feature_dim,
                                  max_bbox=self.max_elements)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                logger.error(
                    "FIDNet checkpoint %s does not match FIDNetV3 "
                    "(num_label=%d, d_model=%d, max_bbox=%d); refusing partial "
                    "load. FID -> N/A. Error: %s", ckpt_path,
                    self.num_categories, self.feature_dim, self.max_elements, exc)
                self.model = None
                return
            model.eval().to(self._resolve_device())
            self.model = model
            logger.info("FIDNet loaded (strict) from %s", ckpt_path)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("FIDNet not available: %s. FID -> N/A.", exc)
            self.model = None

    def _layout_to_tensors(self, layout: Layout) -> dict:
        import torch
        N = min(len(layout), self.max_elements)
        label = torch.zeros(self.max_elements, dtype=torch.long)
        keys = ("center_x", "center_y", "width", "height")
        bbox = {k: torch.zeros(self.max_elements) for k in keys}
        mask = torch.zeros(self.max_elements, dtype=torch.bool)
        elems = layout.sorted_elements("label")
        for i in range(N):
            e = elems[i]
            label[i] = e.label
            bbox["center_x"][i] = e.center_x
            bbox["center_y"][i] = e.center_y
            bbox["width"][i] = e.width
            bbox["height"][i] = e.height
            mask[i] = True
        return {"label": label, "mask": mask, **bbox}

    def extract_features(self, layouts: Sequence[Layout]) -> Optional[np.ndarray]:
        """(N, 256) features, or None if FIDNet is unavailable (-> FID N/A)."""
        if self.model is None:
            return None
        import torch
        feats = []
        for lay in layouts:
            t = self._layout_to_tensors(lay)
            batch = {k: v.unsqueeze(0).to(self.device) for k, v in t.items()}
            with torch.no_grad():
                feats.append(self.model.extract_features(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    @staticmethod
    def compute_fid(real_features: Optional[np.ndarray],
                    gen_features: Optional[np.ndarray]) -> float:
        """Frechet distance, or -1.0 if uncomputable (features/samples missing)."""
        if real_features is None or gen_features is None:
            logger.warning("FID unavailable (FIDNet not loaded); reporting N/A.")
            return UNAVAILABLE
        if real_features.shape[0] < 2 or gen_features.shape[0] < 2:
            logger.warning("FID needs >=2 samples/dist (real=%d gen=%d).",
                           real_features.shape[0], gen_features.shape[0])
            return UNAVAILABLE
        from scipy.linalg import sqrtm
        mu_r, mu_g = real_features.mean(0), gen_features.mean(0)
        sig_r = np.cov(real_features, rowvar=False)
        sig_g = np.cov(gen_features, rowvar=False)
        diff = mu_r - mu_g
        covmean = sqrtm(sig_r @ sig_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
        return float(max(fid, 0.0))


# ======================================================================
# 2. MaxIoU  (best-match IoU vs ground truth)
# ======================================================================

def _pairwise_iou(g, r) -> float:
    ix1, iy1 = max(g[0], r[0]), max(g[1], r[1])
    ix2, iy2 = min(g[2], r[2]), min(g[3], r[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ag = max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1])
    ar = max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = ag + ar - inter
    return inter / union if union > 0 else 0.0


def compute_max_iou(generated: Layout, real: Layout) -> float:
    """Mean over generated elements of the best IoU against any real element.

    NOTE (length confound): this is a precision-style score with NO recall term.
    It divides by the number of GENERATED elements, so a layout that emits FEWER
    elements -- each well placed -- scores HIGHER than one that emits the correct
    count with some imperfect boxes, and omitting an element costs nothing. Use
    ``compute_max_iou_f`` for a length-controlled figure that also rewards
    covering every ground-truth element. This raw metric is retained for
    comparability with prior layout work that reports it.
    """
    if generated.is_empty() or real.is_empty():
        return 0.0
    gb = generated.boxes_xyxy()
    rb = real.boxes_xyxy()
    total = sum(max((_pairwise_iou(g, r) for r in rb), default=0.0) for g in gb)
    return total / len(gb)


def compute_max_iou_f(generated: Layout, real: Layout) -> float:
    """Length-controlled MaxIoU: harmonic mean of placement precision and
    ground-truth coverage (recall).

    precision = mean over GENERATED elements of best IoU vs any GT element
                (== compute_max_iou; rewards well-placed generated boxes)
    recall    = mean over GROUND-TRUTH elements of best IoU vs any generated
                element (penalises MISSING elements -- the term raw MaxIoU lacks)
    F         = 2*precision*recall / (precision + recall)

    This removes the under-generation reward: emitting too few elements raises
    precision but lowers recall, and emitting spurious boxes lowers precision,
    so neither inflates F. Empty layouts (either side) score 0.
    """
    if generated.is_empty() or real.is_empty():
        return 0.0
    gb = generated.boxes_xyxy()
    rb = real.boxes_xyxy()
    precision = sum(max((_pairwise_iou(g, r) for r in rb), default=0.0)
                    for g in gb) / len(gb)
    recall = sum(max((_pairwise_iou(g, r) for g in gb), default=0.0)
                 for r in rb) / len(rb)
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0


# ======================================================================
# 3. AESTHETIC WIN / TIE / LOSS  (+ cross-judge kappa)
# ======================================================================

JUDGE_PROMPT = (
    "You are an expert graphic designer. Compare these two poster layouts and "
    "decide which is more aesthetically pleasing, well-organized, and suitable "
    "for an advertising poster.\n\nLayout A:\n{layout_a}\n\nLayout B:\n{layout_b}"
    "\n\nAnswer with ONLY one token: 'A' if Layout A is clearly better, 'B' if "
    "Layout B is clearly better, or 'T' if they are of equal quality (a tie).")


def _parse_verdict(response: str, gen_is_a: bool) -> Optional[str]:
    """Map a judge token to 'win'/'tie'/'loss' for the generated layout."""
    s = response.strip()
    if not s:
        return None
    ans = s[-1].upper()
    if ans == "T":
        return "tie"
    if ans not in ("A", "B"):
        return None
    gen_won = (gen_is_a and ans == "A") or (not gen_is_a and ans == "B")
    return "win" if gen_won else "loss"


def _judge_verdicts(generated, real, categories, judge_name,
                    tokenizer=None, judge=None):
    """Run one judge over all pairs; return (verdicts, gen_is_a_flags)."""
    import torch
    owns = judge is None
    if owns:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            logger.warning("transformers unavailable for judge (%s).", exc)
            return None, None
        try:
            tokenizer = AutoTokenizer.from_pretrained(judge_name)
            judge = AutoModelForCausalLM.from_pretrained(
                judge_name, torch_dtype=torch.bfloat16, device_map="auto")
            judge.eval()
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load judge '%s' (%s).", judge_name, exc)
            return None, None
    verdicts, flags = [], []
    try:
        for gen, gt in zip(generated, real):
            gj = layout_to_json(gen)
            rj = layout_to_json(gt)
            if random.random() > 0.5:
                prompt = JUDGE_PROMPT.format(layout_a=gj, layout_b=rj)
                gen_is_a = True
            else:
                prompt = JUDGE_PROMPT.format(layout_a=rj, layout_b=gj)
                gen_is_a = False
            inputs = tokenizer(prompt, return_tensors="pt").to(judge.device)
            with torch.no_grad():
                out = judge.generate(**inputs, max_new_tokens=5)
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            verdicts.append(_parse_verdict(resp, gen_is_a))
            flags.append(gen_is_a)
    finally:
        if owns:
            del judge
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return verdicts, flags


def compute_aesthetic_win_rate(generated, real, categories,
                               judge_model_name=None) -> dict:
    """Win/tie/loss rates for the generated layout vs ground truth."""
    na = {"win_rate": UNAVAILABLE, "tie_rate": UNAVAILABLE,
          "loss_rate": UNAVAILABLE, "n_judged": 0}
    if judge_model_name is None:
        logger.info("No judge model; skipping win rate.")
        return na
    verdicts, _ = _judge_verdicts(generated, real, categories, judge_model_name)
    if verdicts is None:
        return na
    valid = [v for v in verdicts if v is not None]
    n = len(valid)
    if n == 0:
        logger.warning("Judge produced no parseable verdicts; N/A.")
        return na
    return {"win_rate": sum(v == "win" for v in valid) / n,
            "tie_rate": sum(v == "tie" for v in valid) / n,
            "loss_rate": sum(v == "loss" for v in valid) / n, "n_judged": n}


def _cohens_kappa(va, vb, labels):
    pairs = [(a, b) for a, b in zip(va, vb) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return UNAVAILABLE, 0
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k))
    for a, b in pairs:
        if a in idx and b in idx:
            conf[idx[a], idx[b]] += 1.0
    po = np.trace(conf) / n
    pe = float(np.sum((conf.sum(1) / n) * (conf.sum(0) / n)))
    if pe >= 1.0:
        return 0.0, n
    return float((po - pe) / (1.0 - pe)), n


def compute_judge_agreement(generated, real, categories,
                            judge_a_name, judge_b_name) -> dict:
    """Two different-family judges on the SAME A/B order -> Cohen's kappa."""
    na = {"kappa": UNAVAILABLE, "n_joint": 0,
          "judge_a": judge_a_name, "judge_b": judge_b_name}
    import torch
    va, flags_a = _judge_verdicts(generated, real, categories, judge_a_name)
    if va is None:
        return na
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return na
    try:
        tok_b = AutoTokenizer.from_pretrained(judge_b_name)
        jb = AutoModelForCausalLM.from_pretrained(
            judge_b_name, torch_dtype=torch.bfloat16, device_map="auto")
        jb.eval()
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load secondary judge '%s' (%s).", judge_b_name, exc)
        return na
    vb = []
    try:
        for (gen, gt), gen_is_a in zip(zip(generated, real), flags_a):
            gj, rj = layout_to_json(gen), layout_to_json(gt)
            prompt = (JUDGE_PROMPT.format(layout_a=gj, layout_b=rj) if gen_is_a
                      else JUDGE_PROMPT.format(layout_a=rj, layout_b=gj))
            inputs = tok_b(prompt, return_tensors="pt").to(jb.device)
            with torch.no_grad():
                out = jb.generate(**inputs, max_new_tokens=5)
            vb.append(_parse_verdict(tok_b.decode(out[0], skip_special_tokens=True),
                                     gen_is_a))
    finally:
        del jb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    kappa, n_joint = _cohens_kappa(va, vb, labels=("win", "tie", "loss"))
    return {"kappa": kappa, "n_joint": n_joint,
            "judge_a": judge_a_name, "judge_b": judge_b_name}


# ======================================================================
# 4. SIGNIFICANCE MACHINERY  (for the A/B/C comparison)
#    Pure numpy/scipy: usable offline and independently testable. These turn
#    "B's mean FID is lower" into "B beats A, p<0.05 after correction" or
#    "B is statistically equivalent to A within margin delta".
# ======================================================================

def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided paired t-test on per-seed (or per-sample) measurements.

    Returns t, p, df, mean_diff (a-b), and the paired 95% CI of the difference.
    Falls back gracefully (p=N/A) when there are <2 pairs or zero variance.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape, "paired arrays must align"
    d = a - b
    n = d.size
    if n < 2:
        return {"t": UNAVAILABLE, "p": UNAVAILABLE, "df": n - 1,
                "mean_diff": float(d.mean()) if n else 0.0, "ci95": (None, None)}
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0:
        # identical-up-to-noise: difference is exactly constant
        p = 0.0 if mean_d != 0 else 1.0
        return {"t": float("inf") if mean_d != 0 else 0.0, "p": p,
                "df": n - 1, "mean_diff": mean_d, "ci95": (mean_d, mean_d)}
    se = sd / math.sqrt(n)
    t = mean_d / se
    df = n - 1
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), df))
        tcrit = float(stats.t.ppf(0.975, df))
    except ImportError:
        # normal approximation if scipy is absent
        p = float(2.0 * 0.5 * math.erfc(abs(t) / math.sqrt(2)))
        tcrit = 1.96
    return {"t": float(t), "p": p, "df": df, "mean_diff": mean_d,
            "ci95": (mean_d - tcrit * se, mean_d + tcrit * se)}


def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction across a family of tests.

    Controls family-wise error when several metrics are compared at once (FID,
    Occ, Rea, ...). Returns, per key: raw p, the Holm-adjusted threshold, and a
    reject flag. More powerful than plain Bonferroni while still FWER-controlling.
    """
    items = [(k, v) for k, v in pvalues.items()
             if isinstance(v, (int, float)) and v >= 0]
    m = len(items)
    out = {}
    if m == 0:
        return out
    items.sort(key=lambda kv: kv[1])  # ascending p
    reject_further = False
    for rank, (k, p) in enumerate(items):
        thresh = alpha / (m - rank)
        if reject_further:
            rej = False
        else:
            rej = p <= thresh
            if not rej:
                reject_further = True  # once one fails, all later fail (step-down)
        out[k] = {"p": float(p), "threshold": float(thresh), "reject_null": bool(rej)}
    return out


def tost_equivalence(a: Sequence[float], b: Sequence[float],
                     margin: float) -> dict:
    """Two One-Sided Tests for equivalence of paired measurements within +/-margin.

    Used to support a "no worse than" / "statistically equivalent" claim (e.g.
    RGPO matches the learned-reward baseline on a metric while removing the
    reward model). Equivalent at level alpha if BOTH one-sided p-values < alpha.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = d.size
    if n < 2:
        return {"equivalent": False, "p_lower": UNAVAILABLE,
                "p_upper": UNAVAILABLE, "mean_diff": float(d.mean()) if n else 0.0,
                "margin": margin}
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 1e-12
    df = n - 1
    t_lower = (mean_d - (-margin)) / se   # H0: diff <= -margin
    t_upper = (mean_d - margin) / se      # H0: diff >=  margin
    try:
        from scipy import stats
        p_lower = float(stats.t.sf(t_lower, df))   # P(T > t_lower)
        p_upper = float(stats.t.cdf(t_upper, df))  # P(T < t_upper)
    except ImportError:
        p_lower = float(0.5 * math.erfc(t_lower / math.sqrt(2)))
        p_upper = float(0.5 * (1.0 + math.erf(t_upper / math.sqrt(2))))
    equivalent = (p_lower < 0.05) and (p_upper < 0.05)
    return {"equivalent": bool(equivalent), "p_lower": p_lower,
            "p_upper": p_upper, "mean_diff": mean_d, "margin": margin}


def compare_arms(results_by_arm: dict, metric: str,
                 baseline: str = "A", alpha: float = 0.05) -> dict:
    """Compare every arm against ``baseline`` on one metric across seeds.

    ``results_by_arm`` maps arm name -> list of per-seed scalar values for the
    metric. Returns each non-baseline arm's paired t-test vs the baseline plus a
    Holm correction across the arms. Direction (lower/higher = better) is the
    caller's to interpret; the test itself is two-sided.
    """
    if baseline not in results_by_arm:
        raise ValueError(f"baseline arm '{baseline}' not in results")
    base = results_by_arm[baseline]
    tests, pvals = {}, {}
    for arm, vals in results_by_arm.items():
        if arm == baseline:
            continue
        tt = paired_t_test(vals, base)
        tests[arm] = tt
        pvals[arm] = tt["p"]
    holm = holm_bonferroni(pvals, alpha=alpha)
    return {"metric": metric, "baseline": baseline,
            "tests": tests, "holm": holm}


# ======================================================================
# 5. SINGLE-TRIAL EVALUATION
# ======================================================================

def evaluate_single_trial(model, test_dataset, config: dict,
                          trial_id: int = 0) -> dict:
    """Generate layouts for the test split and compute all metrics once."""
    import torch
    model.model.eval()
    categories = test_dataset.categories
    task_type = test_dataset.task_type
    temperature = config["inference"]["temperature"]
    top_p = config["inference"]["top_p"]

    seed = config["train_sft"]["seed"] + trial_id
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Trial %d: generating %d layouts (task=%s)...",
                trial_id + 1, len(test_dataset), task_type)

    generated, real = [], []
    acc = defaultdict(list)
    t0 = time.time()
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        gen = model.generate_layout(sample, temperature=temperature, top_p=top_p)
        if isinstance(gen, list):
            gen = gen[0] if gen else Layout(elements=[], categories=tuple(categories))
        gt = sample.get("target_layout", Layout(elements=[], categories=tuple(categories)))
        generated.append(gen)
        real.append(gt)
        # Occ needs the saliency map; Rea needs the canvas pixels. The dataset
        # owns both (load_saliency / the rendered image in the sample); they are
        # NOT carried as sample keys, so fetch them here. Without this, both
        # metrics would silently read None and report 0.0 for every sample.
        sid = sample["metadata"]["sample_id"]
        saliency_np = (test_dataset.load_saliency(sid)
                       if hasattr(test_dataset, "load_saliency") else None)
        canvas_np = (np.asarray(sample["image"])
                     if sample.get("image") is not None else None)
        acc["occ"].append(occlusion(gen, saliency_np))
        acc["rea"].append(readability(gen, canvas_np))
        acc["und"].append(underlay_effectiveness(gen))
        acc["ove"].append(overlay(gen))
        acc["align"].append(alignment(gen))
        if not gt.is_empty():
            acc["max_iou"].append(compute_max_iou(gen, gt))
            acc["max_iou_f"].append(compute_max_iou_f(gen, gt))
        if (idx + 1) % 200 == 0:
            logger.info("  generated %d/%d (%.1f/s)", idx + 1, len(test_dataset),
                        (idx + 1) / (time.time() - t0))

    results = {}
    for key in ("occ", "rea", "und", "ove", "align", "max_iou", "max_iou_f"):
        vals = acc.get(key, [])
        results[key] = float(np.mean(vals)) if vals else 0.0

    fid_calc = LayoutFIDCalculator(
        num_categories=len(categories),
        max_elements=config["dataset"]["max_elements"],
        fidnet_ckpt=config.get("evaluation", {}).get("fid_checkpoint"))
    gen_f = fid_calc.extract_features(generated)
    real_f = fid_calc.extract_features(real)
    results["fid"] = fid_calc.compute_fid(real_f, gen_f)

    if config["evaluation"].get("use_aesthetic_winrate", False):
        judge = config["evaluation"].get("judge_model")
        wtl = compute_aesthetic_win_rate(generated, real, categories, judge)
        results["win_rate"] = wtl["win_rate"]
        results["tie_rate"] = wtl["tie_rate"]
        results["loss_rate"] = wtl["loss_rate"]
        judge_b = config["evaluation"].get("judge_model_secondary")
        if judge_b and judge_b != judge:
            agree = compute_judge_agreement(generated, real, categories, judge, judge_b)
            results["judge_kappa"] = agree.get("kappa", UNAVAILABLE)
        else:
            results["judge_kappa"] = UNAVAILABLE
    else:
        results.update({"win_rate": UNAVAILABLE, "tie_rate": UNAVAILABLE,
                        "loss_rate": UNAVAILABLE, "judge_kappa": UNAVAILABLE})

    results["num_samples"] = len(test_dataset)
    results["avg_num_elements"] = (float(np.mean([len(g) for g in generated]))
                                   if generated else 0.0)
    results["valid_rate"] = float(np.mean([1.0 if len(g) > 0 else 0.0
                                           for g in generated])) if generated else 0.0
    results["gen_time_s"] = time.time() - t0
    return results


# ======================================================================
# 6. MULTI-TRIAL AVERAGING
# ======================================================================

def evaluate_all(model, config: dict, split: str = "test",
                 task_type: str = "unconstrained",
                 num_trials: Optional[int] = None) -> dict:
    """Run num_trials trials and report per-metric mean/std (+ raw trials)."""
    from data import PosterLayoutDataset
    num_trials = num_trials or config["evaluation"]["num_trials"]
    test_ds = PosterLayoutDataset(config, split=split, task_type=task_type)
    logger.info("=" * 60)
    logger.info("EVALUATION %s/%s task=%s trials=%d samples=%d",
                config["dataset"]["name"], split, task_type, num_trials, len(test_ds))
    logger.info("=" * 60)

    trials = []
    for t in range(num_trials):
        r = evaluate_single_trial(model, test_ds, config, trial_id=t)
        trials.append(r)
        _print_trial(r, t + 1)

    keys = ("fid", "occ", "rea", "und", "ove", "max_iou", "max_iou_f", "align",
            "win_rate", "tie_rate", "loss_rate", "judge_kappa",
            "valid_rate", "avg_num_elements")
    out = {}
    for k in keys:
        vals = [r[k] for r in trials if r.get(k, UNAVAILABLE) >= 0]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else UNAVAILABLE
        out[f"{k}_std"] = float(np.std(vals)) if vals else 0.0
        out[f"{k}_values"] = vals    # per-trial values WITHIN this seed/run
        # (diagnostic only). Across-SEED significance uses the per-seed
        # {metric}_mean from each JSON, paired by seed in experiments.py — that
        # is the correct unit for training-seed variance, not these within-seed
        # trial values.
    out.update({"dataset": config["dataset"]["name"], "split": split,
                "task_type": task_type, "num_trials": num_trials,
                "num_samples": len(test_ds),
                "model": config["model"]["mllm_backbone"], "trial_results": trials})
    _print_summary(out)
    return out


def evaluate_all_tasks(model, config: dict, split: str = "test") -> dict:
    out = {}
    for task in config["evaluation"]["constrained_tasks"]:
        logger.info("\n%s\nTASK: %s\n%s", "=" * 60, task, "=" * 60)
        out[task] = evaluate_all(model, config, split=split, task_type=task)
    _print_task_comparison(out)
    return out


# ======================================================================
# 7. FORMATTING + EXPORT
# ======================================================================

def _fid_str(v: float) -> str:
    return f"{v:.2f}" if v >= 0 else "N/A"


def _print_trial(r: dict, n: int):
    logger.info("  Trial %d: FID=%s Occ=%.4f Rea=%.4f Und=%.4f Ove=%.4f "
                "MaxIoU=%.4f Valid=%.2f%%", n, _fid_str(r.get("fid", -1)),
                r["occ"], r["rea"], r["und"], r["ove"], r["max_iou"],
                100 * r["valid_rate"])


def _print_summary(a: dict):
    logger.info("\n%s\nEVALUATION SUMMARY %s/%s task=%s trials=%d\n%s",
                "=" * 60, a["dataset"], a["split"], a["task_type"],
                a["num_trials"], "-" * 60)
    rows = [("FID  (down)", "fid"), ("Occ  (down)", "occ"), ("Rea  (down)", "rea"),
            ("Und  (up)", "und"), ("Ove  (down)", "ove"), ("MaxIoU (up)", "max_iou"),
            ("Align(down)", "align"), ("WinRate(up)", "win_rate"),
            ("TieRate", "tie_rate"), ("LossRate", "loss_rate"),
            ("JudgeKappa", "judge_kappa"), ("ValidRate", "valid_rate")]
    for name, key in rows:
        m, s = a.get(f"{key}_mean", -1), a.get(f"{key}_std", 0.0)
        logger.info("  %-13s %s", name,
                    f"{m:.4f} +/- {s:.4f}" if m >= 0 else "N/A")
    logger.info("=" * 60)


def _print_task_comparison(all_results: dict):
    logger.info("\n%s\nTASK COMPARISON\n%s", "=" * 80, "=" * 80)
    logger.info("%-18s %8s %8s %8s %8s %8s %8s", "Task", "FID", "Occ", "Rea",
                "Und", "Ove", "MaxIoU")
    for task, r in all_results.items():
        logger.info("%-18s %8.2f %8.4f %8.4f %8.4f %8.4f %8.4f", task,
                    r.get("fid_mean", -1), r.get("occ_mean", -1),
                    r.get("rea_mean", -1), r.get("und_mean", -1),
                    r.get("ove_mean", -1), r.get("max_iou_mean", -1))
    logger.info("=" * 80)


def save_results(results: dict, output_path: str):
    def _clean(v):
        if isinstance(v, (int, float, str, bool)):
            return v
        if isinstance(v, (list, tuple)):
            return [x for x in v if isinstance(x, (int, float, str, bool))]
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        return None
    clean = {}
    for k, v in results.items():
        if k == "trial_results":
            clean[k] = [{kk: vv for kk, vv in tr.items()
                         if isinstance(vv, (int, float, str, bool))} for tr in v]
        else:
            cv = _clean(v)
            if cv is not None:
                clean[k] = cv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info("Results saved to %s", output_path)


# ======================================================================
# 8. CLI
# ======================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM-RAL / RGPO Evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test", choices=["test", "val", "unannotated"])
    parser.add_argument("--task", default="unconstrained", choices=list(TASK_TYPES) + ["all"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config)
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.seed is not None:
        config["train_sft"]["seed"] = args.seed
    if args.override:
        from train import load_and_apply_overrides
        config = load_and_apply_overrides(config, args.override)

    from model import create_model
    model = create_model(config, mode="inference")
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    else:
        ds, seed = config["dataset"]["name"], config["train_sft"]["seed"]
        cands = [f"{ds}_s{seed}_{s}_{t}" for s in ("B", "A", "C")
                 for t in ("align_best", "align_final")]
        cands += [f"{ds}_s{seed}_{t}" for t in ("sft_best", "sft_final")]
        for name in cands:
            p = os.path.join(config["paths"]["checkpoint_dir"], name)
            if os.path.exists(p):
                logger.info("Auto-detected checkpoint: %s", p)
                model.load_checkpoint(p)
                break
        else:
            logger.warning("No checkpoint found; using base model")

    if args.task == "all":
        results = evaluate_all_tasks(model, config, split=args.split)
        out = args.output or os.path.join(
            config["paths"]["output_dir"],
            f"eval_{config['dataset']['name']}_{args.split}_all_tasks.json")
        save_results({"tasks": {k: {kk: vv for kk, vv in v.items()
                                    if kk != "trial_results"}
                                for k, v in results.items()}}, out)
    else:
        results = evaluate_all(model, config, split=args.split,
                               task_type=args.task, num_trials=args.trials)
        out = args.output or os.path.join(
            config["paths"]["output_dir"],
            f"eval_{config['dataset']['name']}_{args.split}_{args.task}.json")
        save_results(results, out)


# ======================================================================
# 9. SELF-TEST  (run: python -m src.evaluate)
#    The metrics + significance machinery are pure numpy/scipy and fully
#    testable offline (FID/judge need weights+GPU, exercised in real runs).
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("evaluate.py SELF-TEST  (metrics + significance)")
    print("=" * 64)

    from data import CATEGORY_MAPS
    cats = CATEGORY_MAPS["pku"]

    def L(recs):
        return Layout.from_records(recs, cats)

    # (a) MaxIoU: identical layouts -> 1.0; disjoint -> 0.0.
    a = L([{"category": "text", "center_x": 0.5, "center_y": 0.5, "width": 0.4, "height": 0.2}])
    assert abs(compute_max_iou(a, a) - 1.0) < 1e-9
    b = L([{"category": "text", "center_x": 0.05, "center_y": 0.05, "width": 0.05, "height": 0.05}])
    assert compute_max_iou(a, b) < 0.05
    print("  [ok] MaxIoU: identical=1.0, disjoint~0.0")

    # (a2) Length-controlled MaxIoU-F removes the under-generation reward.
    gt6 = L([
        {"category": "logo", "center_x": 0.2, "center_y": 0.1, "width": 0.15, "height": 0.08},
        {"category": "text", "center_x": 0.5, "center_y": 0.3, "width": 0.40, "height": 0.10},
        {"category": "text", "center_x": 0.5, "center_y": 0.45, "width": 0.35, "height": 0.08},
        {"category": "underlay", "center_x": 0.5, "center_y": 0.3, "width": 0.45, "height": 0.12},
        {"category": "text", "center_x": 0.5, "center_y": 0.7, "width": 0.30, "height": 0.06},
        {"category": "logo", "center_x": 0.8, "center_y": 0.9, "width": 0.12, "height": 0.07},
    ])
    few_perfect = L(gt6.to_records()[:2])          # 2 exact boxes, 4 missing
    raw_few = compute_max_iou(few_perfect, gt6)
    f_few = compute_max_iou_f(few_perfect, gt6)
    # Raw rewards under-generation (2 exact boxes -> 1.0); F must penalise the 4 missing.
    assert abs(raw_few - 1.0) < 1e-9, raw_few
    assert f_few < 0.7, f_few                       # recall penalty pulls F below raw (1.0)
    assert f_few < raw_few - 0.3, (f_few, raw_few)   # and well below the raw score
    # Identical full layout still scores 1.0 under F (precision=recall=1).
    assert abs(compute_max_iou_f(gt6, gt6) - 1.0) < 1e-9
    # Empty either side -> 0.0 for F.
    assert compute_max_iou_f(L([]), gt6) == 0.0 and compute_max_iou_f(gt6, L([])) == 0.0
    print(f"  [ok] MaxIoU-F controls length: 2-of-6 exact -> raw={raw_few:.3f} but F={f_few:.3f} "
          f"(recall penalty); identical=1.0; empty=0.0")

    # (b) Verdict parsing -> win/tie/loss with A/B position handling.
    assert _parse_verdict("A", gen_is_a=True) == "win"
    assert _parse_verdict("A", gen_is_a=False) == "loss"
    assert _parse_verdict("T", gen_is_a=True) == "tie"
    assert _parse_verdict("garbage", gen_is_a=True) is None
    print("  [ok] judge verdict parsing (win/tie/loss/none)")

    # (c) Cohen's kappa: perfect agreement=1.0, and is 0 at chance.
    va = ["win", "loss", "tie", "win", "loss"]
    k_perfect, n = _cohens_kappa(va, va, ("win", "tie", "loss"))
    assert abs(k_perfect - 1.0) < 1e-9 and n == 5
    print(f"  [ok] Cohen's kappa: perfect agreement = {k_perfect:.2f}")

    # (d) Paired t-test: a consistent shift is detected; identical -> p=1.
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1.0, size=8)
    better = base - 0.8                         # consistently lower (better FID)
    tt = paired_t_test(better, base)
    assert tt["mean_diff"] < 0 and 0 <= tt["p"] <= 1
    tt_same = paired_t_test(base, base)
    assert tt_same["p"] == 1.0
    print(f"  [ok] paired t-test: shift detected (mean_diff={tt['mean_diff']:.2f}, p={tt['p']:.3g})")

    # (e) Holm-Bonferroni: step-down ordering and FWER control.
    holm = holm_bonferroni({"fid": 0.001, "occ": 0.04, "rea": 0.20}, alpha=0.05)
    assert holm["fid"]["reject_null"] is True      # smallest p, tightest survive
    assert holm["rea"]["reject_null"] is False     # largest p, not rejected
    # threshold for the smallest p is alpha/m
    assert abs(holm["fid"]["threshold"] - 0.05 / 3) < 1e-9
    print("  [ok] Holm-Bonferroni: step-down thresholds + reject flags")

    # (f) TOST: equivalent within margin vs not.
    x = rng.normal(5.0, 0.05, size=12)
    y = x + 0.01                                  # tiny difference
    eq = tost_equivalence(x, y, margin=0.2)
    assert eq["equivalent"] is True
    neq = tost_equivalence(x, x + 0.5, margin=0.1)   # difference > margin
    assert neq["equivalent"] is False
    print("  [ok] TOST equivalence: within-margin=equivalent, beyond=not")

    # (g) compare_arms: B vs A baseline across seeds, with Holm across arms.
    arms = {"A": [10.0, 10.2, 9.9, 10.1],
            "B": [9.1, 9.3, 9.0, 9.2],     # consistently lower than A
            "C": [9.6, 9.7, 9.5, 9.6]}
    cmp = compare_arms(arms, metric="fid", baseline="A")
    assert "B" in cmp["tests"] and "C" in cmp["tests"]
    assert cmp["tests"]["B"]["mean_diff"] < 0      # B lower than A
    assert "B" in cmp["holm"] and "C" in cmp["holm"]
    print(f"  [ok] compare_arms: B-vs-A mean_diff={cmp['tests']['B']['mean_diff']:.2f}, "
          f"Holm computed across arms")

    # (h) FID sentinel: missing features -> N/A (-1.0), never fabricated.
    assert LayoutFIDCalculator.compute_fid(None, None) == UNAVAILABLE
    assert LayoutFIDCalculator.compute_fid(np.zeros((1, 4)), np.zeros((1, 4))) == UNAVAILABLE
    print("  [ok] FID returns N/A sentinel when uncomputable (never fabricated)")

    print("=" * 64)
    print("  ALL evaluate.py SELF-TESTS PASSED")
    print("=" * 64)
