"""
losses.py — Training Objectives (SFT / DPO / DMPO)   SCORER-AGNOSTIC
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [5] of 10.

The one rule this file obeys
----------------------------
losses.py computes LOSSES. It does NOT compute the preference signal. The
dynamic-margin objective receives an already-normalised delta in (0, 1] as an
input tensor and turns it into a margin f(delta); it never knows, and must never
know, whether that delta came from source A (learned reward), source B (RGPO
retrieval-agreement), or source C (hybrid).

Why this matters for the paper
------------------------------
The central experiment is a clean control: hold the SFT model, the sampler, the
pairs, the optimiser, AND THE LOSS all fixed, and vary ONLY the source of delta.
That control is only valid if the loss is byte-identical across A/B/C. Keeping
delta computation out of this file is what guarantees it. The single place the
three sources diverge is preferences.py (file [6]); rgpo.py (file [4]) owns the
delta math; losses.py is a pure consumer of delta.

What moved OUT of this file (vs the original)
---------------------------------------------
The original losses.py also held the five geometric metrics (Occ/Rea/Und/Ove/
Align) and the composite ``AestheticLayoutScorer``. Those are a *preference
source* (the prior-art reward model), so they now live in rgpo.py as
``LearnedRewardScorer`` and the shared metric functions. losses.py is leaner and
single-purpose as a result.

The delta normalisation contract (unchanged, enforced upstream)
---------------------------------------------------------------
DMPO's margin is f(delta) = exp(delta) - exp(-delta) = 2*sinh(delta), and it
expects delta in (0, 1]. The normalisation that puts delta in that range is a
property of the *scorer* (each source divides its raw quality gap by its own
attainable ``span``; see rgpo.py) and is applied at pair-construction time in
preferences.py. The ``clamp(1e-6, 1.0)`` in ``compute_dynamic_margin`` is a
NUMERICAL GUARD ONLY — it is not the source of the (0, 1] range. (An earlier
design fed a raw, unnormalised gap here and hard-clamped to 1.0, collapsing
every well-separated pair to the same margin f(1)=e-1/e~=2.35 and destroying the
confidence signal. That mistake is structurally impossible now: this file has no
access to raw scores at all.)

References
----------
  - DPO: Rafailov et al., NeurIPS 2023
  - DMPO dynamic margin: Lu et al. (Uni-Layout), ACM MM 2025, Eq. 6-7
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ======================================================================
# 1. SFT LOSS  (Stage 1)
#    A thin wrapper over the model's own cross-entropy: it does not recompute
#    anything, it just surfaces the scalar and tracks a running average for
#    logging. The prompt tokens are already masked (-100) inside model.py's
#    _compute_labels, so only layout-response tokens contribute.
# ======================================================================

class SFTLoss:
    """Pass-through wrapper around the model's masked cross-entropy loss."""

    def __init__(self):
        self.running_loss = 0.0
        self.num_steps = 0

    def __call__(self, model_output: dict):
        """Return the scalar SFT loss from a forward_sft() output dict."""
        loss = model_output["loss"]
        # ``.item()`` works on a real tensor; the self-test double provides it.
        self.running_loss += float(loss.item())
        self.num_steps += 1
        return loss

    @property
    def average_loss(self) -> float:
        return self.running_loss / self.num_steps if self.num_steps else 0.0

    def reset(self):
        self.running_loss = 0.0
        self.num_steps = 0


# ======================================================================
# 2. DPO LOSS  (standard, fixed-margin baseline)
#    L_DPO = -log sigma( beta * (Delta_w - Delta_l) )
#    Delta_k = log pi_theta(y_k|x) - log pi_ref(y_k|x)
# ======================================================================

class DPOLoss:
    """Standard Direct Preference Optimization loss (Rafailov et al., 2023)."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta
        self.running_loss = 0.0
        self.running_chosen_reward = 0.0
        self.running_rejected_reward = 0.0
        self.num_steps = 0

    def __call__(self, policy_chosen_logps, policy_rejected_logps,
                 ref_chosen_logps, ref_rejected_logps) -> dict:
        import torch
        import torch.nn.functional as F
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self._track(loss, chosen_rewards, rejected_rewards)
        return {
            "loss": loss,
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        }

    def _track(self, loss, chosen_rewards, rejected_rewards):
        self.running_loss += float(loss.item())
        self.running_chosen_reward += float(chosen_rewards.mean().item())
        self.running_rejected_reward += float(rejected_rewards.mean().item())
        self.num_steps += 1

    @property
    def average_loss(self) -> float:
        return self.running_loss / self.num_steps if self.num_steps else 0.0

    def reset(self):
        self.running_loss = 0.0
        self.running_chosen_reward = 0.0
        self.running_rejected_reward = 0.0
        self.num_steps = 0


# ======================================================================
# 3. DMPO LOSS  (Dynamic-Margin Preference Optimization)
#    L_DMPO = -log sigma( beta * (Delta_w - Delta_l) - f(delta) )
#    f(delta) = exp(delta) - exp(-delta) = 2*sinh(delta),  delta in (0, 1]
#
#    delta IS AN INPUT. This class never computes it. preferences.py assigns
#    delta per pair via the selected rgpo.py source; train.py passes it here.
# ======================================================================

class DMPOLoss:
    """DMPO loss: DPO plus a confidence margin f(delta) scaled by the
    (already-normalised) preference strength delta in (0, 1]."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta
        self.running_loss = 0.0
        self.running_chosen_reward = 0.0
        self.running_rejected_reward = 0.0
        self.running_margin = 0.0
        self.num_steps = 0

    @staticmethod
    def compute_dynamic_margin(score_margin):
        """f(delta) = exp(delta) - exp(-delta) = 2*sinh(delta).

        Smooth, monotone increasing on (0, 1]: near-ties (delta -> 0) get a
        vanishing margin; clearly separated pairs (delta -> 1) get the largest.

        delta MUST ALREADY BE IN (0, 1] — that normalisation is the scorer's
        job (rgpo.py ``span``) applied at pair-construction time. The clamp
        below is a numerical guard, NOT the source of the range.
        """
        import torch
        delta = score_margin.clamp(min=1e-6, max=1.0)
        return torch.exp(delta) - torch.exp(-delta)

    def __call__(self, policy_chosen_logps, policy_rejected_logps,
                 ref_chosen_logps, ref_rejected_logps, score_margin) -> dict:
        import torch
        import torch.nn.functional as F
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        f_delta = self.compute_dynamic_margin(score_margin)
        logits = self.beta * (chosen_logratios - rejected_logratios) - f_delta
        loss = -F.logsigmoid(logits).mean()

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self.running_loss += float(loss.item())
        self.running_chosen_reward += float(chosen_rewards.mean().item())
        self.running_rejected_reward += float(rejected_rewards.mean().item())
        self.running_margin += float(f_delta.mean().item())
        self.num_steps += 1
        return {
            "loss": loss,
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
            "dynamic_margin": f_delta.mean(),
        }

    @property
    def average_loss(self) -> float:
        return self.running_loss / self.num_steps if self.num_steps else 0.0

    @property
    def average_margin(self) -> float:
        return self.running_margin / self.num_steps if self.num_steps else 0.0

    def reset(self):
        self.running_loss = 0.0
        self.running_chosen_reward = 0.0
        self.running_rejected_reward = 0.0
        self.running_margin = 0.0
        self.num_steps = 0


# ======================================================================
# 4. FACTORY
#    Selects the alignment objective from config. Note this depends ONLY on
#    margin_type, NOT on the preference source — A/B/C all use the SAME loss
#    (dynamic by default), which is exactly the control the paper needs.
# ======================================================================

def create_dpo_loss(config: dict):
    """Return DMPOLoss (dynamic margin) or DPOLoss (fixed) per config.

    The preference *source* (A/B/C) is irrelevant here and intentionally not
    consulted: the loss must be identical across the three arms.
    """
    beta = config["train_dpo"]["beta"]
    margin_type = config["train_dpo"]["margin_type"]
    if margin_type == "dynamic":
        logger.info("Using DMPO loss (beta=%s, dynamic margin)", beta)
        return DMPOLoss(beta=beta)
    logger.info("Using standard DPO loss (beta=%s, fixed margin)", beta)
    return DPOLoss(beta=beta)


# ======================================================================
# 5. SELF-TEST  (run: python -m src.losses)
#    torch is unavailable offline, so a numpy-backed tensor DOUBLE stands in
#    for torch (same idea as the FAISS stand-in in retrieval.py). The ACTUAL
#    loss code runs against it — we are testing the real arithmetic, not a
#    re-implementation. The double is installed into sys.modules as `torch`
#    and `torch.nn.functional` only for the duration of the test.
# ======================================================================

if __name__ == "__main__":
    import sys
    import types
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("losses.py SELF-TEST  (scorer-agnostic objectives)")
    print("=" * 64)

    # ---- numpy-backed tensor double ----
    class T:
        """Minimal tensor: wraps a numpy array, supports the ops losses.py uses."""
        __array_priority__ = 1000  # ensure our __rmul__/__rsub__ win vs numpy

        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float64)

        # arithmetic
        def __sub__(self, o): return T(self.data - _v(o))
        def __rsub__(self, o): return T(_v(o) - self.data)
        def __add__(self, o): return T(self.data + _v(o))
        def __mul__(self, o): return T(self.data * _v(o))
        __rmul__ = __mul__
        def __neg__(self): return T(-self.data)

        # tensor methods used by the loss code
        def clamp(self, min=None, max=None):
            return T(np.clip(self.data, min, max))
        def mean(self): return T(self.data.mean())
        def detach(self): return T(self.data.copy())
        def item(self): return float(self.data.reshape(-1)[0]) if self.data.size == 1 \
            else float(self.data.mean())

    def _v(o):
        return o.data if isinstance(o, T) else o

    # ---- fake torch + torch.nn.functional modules ----
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = T
    fake_torch.exp = lambda t: T(np.exp(_v(t)))
    fake_torch.tensor = lambda d: T(d)

    fake_nn = types.ModuleType("torch.nn")
    fake_F = types.ModuleType("torch.nn.functional")
    # numerically-stable log-sigmoid: log_sigmoid(x) = -softplus(-x) = -logaddexp(0, -x)
    fake_F.logsigmoid = lambda t: T(-np.logaddexp(0.0, -_v(t)))
    fake_nn.functional = fake_F

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.nn.functional"] = fake_F

    # ---- (a) f(delta) = 2 sinh(delta): value, monotonicity, range ----
    for d in (0.1, 0.3, 0.5, 0.8, 1.0):
        f = DMPOLoss.compute_dynamic_margin(T([d]))
        assert abs(f.item() - 2.0 * np.sinh(d)) < 1e-9, (d, f.item())
    f_lo = DMPOLoss.compute_dynamic_margin(T([0.1])).item()
    f_hi = DMPOLoss.compute_dynamic_margin(T([0.9])).item()
    assert f_lo < f_hi, "margin must increase with delta"
    print("  [ok] f(delta)=2*sinh(delta): exact, monotone increasing")

    # ---- (b) clamp is a guard: out-of-range delta is bounded, not trusted ----
    f_raw = DMPOLoss.compute_dynamic_margin(T([5.0])).item()   # raw>1 -> clamp to 1
    assert abs(f_raw - 2.0 * np.sinh(1.0)) < 1e-9
    f_zero = DMPOLoss.compute_dynamic_margin(T([0.0])).item()  # 0 -> 1e-6 floor
    assert f_zero > 0 and f_zero < 1e-3
    print("  [ok] clamp guards delta to (0,1] (numerical guard only)")

    # ---- (c) DMPO loss runs and is finite; bigger logit gap -> smaller loss ----
    dmpo = DMPOLoss(beta=0.1)
    out_far = dmpo(T([-10.0]), T([-15.0]), T([-11.0]), T([-14.0]), T([0.5]))
    out_near = dmpo(T([-10.0]), T([-11.0]), T([-11.0]), T([-10.5]), T([0.5]))
    assert np.isfinite(out_far["loss"].item()) and np.isfinite(out_near["loss"].item())
    assert out_far["loss"].item() < out_near["loss"].item()
    assert "dynamic_margin" in out_far
    print("  [ok] DMPO loss finite; clearer preference -> lower loss")

    # ---- (d) SCORER-AGNOSTIC: same logps + same delta -> same loss, no matter
    #          which 'source' produced delta. (Loss only sees the number.) ----
    pc, pr, rc, rr = T([-9.0]), T([-13.0]), T([-10.0]), T([-12.0])
    delta_val = 0.42
    L_as_if_A = DMPOLoss(beta=0.1)(pc, pr, rc, rr, T([delta_val]))["loss"].item()
    L_as_if_B = DMPOLoss(beta=0.1)(pc, pr, rc, rr, T([delta_val]))["loss"].item()
    assert abs(L_as_if_A - L_as_if_B) < 1e-12
    print("  [ok] identical delta -> identical loss (A/B/C control holds)")

    # ---- (e) delta changes the loss monotonically (margin actually bites) ----
    base = (pc, pr, rc, rr)
    L_small = DMPOLoss(beta=0.1)(*base, T([0.1]))["loss"].item()
    L_large = DMPOLoss(beta=0.1)(*base, T([0.9]))["loss"].item()
    # larger margin subtracts more from the logit -> larger loss
    assert L_large > L_small
    print("  [ok] larger delta -> larger subtracted margin -> larger loss")

    # ---- (f) standard DPO loss runs and is finite ----
    dpo = DPOLoss(beta=0.1)
    od = dpo(T([-10.0]), T([-15.0]), T([-11.0]), T([-14.0]))
    assert np.isfinite(od["loss"].item()) and "reward_margin" in od
    print("  [ok] standard DPO loss finite, returns reward margin")

    # ---- (g) SFT wrapper passes the scalar through and averages ----
    sft = SFTLoss()
    _ = sft({"loss": T([2.0])}); _ = sft({"loss": T([4.0])})
    assert abs(sft.average_loss - 3.0) < 1e-9
    print("  [ok] SFT wrapper surfaces loss and tracks running average")

    # ---- (h) factory honours margin_type, ignores preference source ----
    cfg_dyn = {"train_dpo": {"beta": 0.1, "margin_type": "dynamic"}}
    cfg_fix = {"train_dpo": {"beta": 0.1, "margin_type": "fixed"}}
    assert isinstance(create_dpo_loss(cfg_dyn), DMPOLoss)
    assert isinstance(create_dpo_loss(cfg_fix), DPOLoss)
    print("  [ok] factory: dynamic->DMPO, fixed->DPO (source never consulted)")

    # restore: drop the fake torch so we don't leak it to other imports
    for m in ("torch", "torch.nn", "torch.nn.functional"):
        sys.modules.pop(m, None)

    print("=" * 64)
    print("  ALL losses.py SELF-TESTS PASSED")
    print("=" * 64)
