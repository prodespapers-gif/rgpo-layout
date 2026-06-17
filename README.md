# RGPO — Retrieval-Grounded Preference Optimization

**LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design**

Reference implementation for the paper. RGPO aligns a multimodal layout
generator to human design preference **without a reward model**: in a
retrieval-augmented generator, the *K* retrieved neighbours are real human
layouts for the input canvas, so the retrieval set itself defines the preference
signal that prior methods train a separate reward/scoring model to emit.

The repository is deliberately small — ten source files — and is structured so
that the central claim can be tested as a clean controlled experiment.

---

## The idea in one paragraph

Every prior preference-aligned layout method (AesthetiQ, Uni-Layout/DMPO, …)
needs a *reward model*: a separately trained network, or a hand-weighted sum of
geometric metrics, that scores which of two layouts is better. RGPO removes it.
For a given canvas we have already retrieved its *K* nearest **real human
designs**; those neighbours are not merely prompt context but a sufficient
statistic for "what a good layout on *this* canvas looks like." We therefore
define preference directly as **agreement with the retrieved neighbourhood**.
The retrieval set the model already conditions on becomes the preference signal.
No reward model, no learned weights, no human ratings.

## The controlled experiment (A/B/C)

The headline result rests on an experiment that holds **everything fixed** — the
SFT model, the dynamic-margin DPO loss, the candidate sampler, the pairing rule,
and the training data — and varies only the **source** of the preference
strength `δ`:

| Source | Scorer | Preference comes from | Role |
|:------:|--------|-----------------------|------|
| **A** | `LearnedRewardScorer` | a 5-term hand-weighted composite reward (the prior-art reward model) | baseline |
| **B** | `RGPOScorer` | neighbourhood agreement — **no reward model** | the contribution |
| **C** | `HybridScorer` | convex blend of A and B | hedge |

Because only the scorer changes, any difference between the resulting models is
attributable to the preference source alone. One CLI flag selects the arm:
`--pref-source {A,B,C}`.

---

## Repository layout

```
.
├── src/
│   ├── data.py          # [1] datasets, layout (de)serialization, constrained-task construction
│   ├── retrieval.py     # [2] DreamSim embeddings + FAISS nearest-neighbour retrieval
│   ├── model.py         # [3] Qwen2.5-VL + LoRA policy; masked-logprob log-ratio for DPO
│   ├── rgpo.py          # [4] THE CONTRIBUTION: agreement score + A/B/C preference sources
│   ├── preferences.py   # [5] candidate sampling → best/worst pairing → δ → pair dataset
│   ├── losses.py        # [6] dynamic-margin DPO loss  f(δ) = 2·sinh(δ)  (DMPO)
│   ├── train.py         # [7] two-stage training: SFT → RGPO alignment  (CLI entry)
│   ├── evaluate.py      # [8] FID/Occ/Rea/Und_s/Ove/Align/MaxIoU/Val + paired significance
│   ├── inference.py     # [9] sample layouts; per-method comparison renders
│   └── experiments.py   # [10] orchestrates the paper's tables; aggregates per-seed runs
├── config.yaml          # single source of run configuration (all modules read this)
├── requirements.txt
├── AUDIT_LOG.md         # record of bugs found & fixed during review (read before trusting numbers)
├── LICENSE              # MIT
└── README.md
```

Heavy dependencies (`torch`, `transformers`, `faiss`, `dreamsim`) are imported
lazily inside the functions that need them, so the lightweight utilities and the
checks documented in `AUDIT_LOG.md` can be exercised CPU-only.

---

## Installation

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/<your-org>/rgpo-layout.git
cd rgpo-layout
python -m venv .venv && source .venv/bin/activate

# Install PyTorch + FAISS for YOUR hardware first (see requirements.txt notes):
#   GPU:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#         pip install faiss-gpu
#   CPU:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#         pip install faiss-cpu
pip install -r requirements.txt
```

> Do **not** install both `faiss-cpu` and `faiss-gpu`. Pick the one matching your
> machine and edit `requirements.txt` accordingly.

---

## Data

The repository ships **no data**. Both benchmarks and the retrieval pipeline are
public and have their own licenses; obtain them from the original sources:

- **PKU PosterLayout** — poster/layout pairs, 3 element types (logo, text, underlay).
- **CGL-Dataset V2** — adds a 4th type (embellishment) and text-content annotations.
- **RALF** preprocessing, splits, and DreamSim/FAISS retrieval indices — we adopt
  these directly so our neighbourhoods are identical to the retrieval-augmented baseline.

Place them where `config.yaml` expects (edit paths to match your machine):

```yaml
paths:
  pku_root:           ./data/PKU-PosterLayout
  cgl_root:           ./data/CGL-Dataset-v2
  ralf_splits_dir:    ./data/ralf_splits
  ralf_retrieval_dir: ./data/ralf_retrieval
```

> **Geometry note (important):** `dataset.original_poster_size` is `[H, W]`
> (height first). PKU canvases are 513 wide × 750 tall, so this **must** be
> `[750, 513]`. The loader unpacks `orig_h, orig_w = size`; transposing it
> distorts every pixel-coordinate box. This is set correctly in the shipped
> config — do not "fix" it back.

---

## Quick start

All commands read `config.yaml`; flags override individual keys.

### Train (two stages)

```bash
# Stage 1 — supervised fine-tuning of the reference policy (shared by all arms)
python -m src.train --stage sft   --config config.yaml --seed 42

# Stage 2 — RGPO alignment. Pick the preference source with --pref-source.
python -m src.train --stage align --config config.yaml --pref-source B --seed 42

# Or run both stages end-to-end:
python -m src.train --stage full  --config config.yaml --pref-source B --seed 42
```

A single SFT checkpoint per seed is **reused** across A/B/C, so the three arms
differ only in the alignment stage — this is what makes the comparison a control.

### Evaluate

```bash
# Main metrics on the test split
python -m src.evaluate --config config.yaml --dataset pku --split test \
    --checkpoint checkpoints/<run>_sB_seed42 --task unconstrained

# All six constrained tasks
python -m src.evaluate --config config.yaml --dataset pku --task all \
    --checkpoint checkpoints/<run>_sB_seed42
```

### Inference / qualitative comparison

```bash
# Sample layouts for one canvas
python -m src.inference --config config.yaml --mode single \
    --canvas path/to/canvas.png --checkpoint checkpoints/<run>_sB_seed42

# Side-by-side method comparison (Canvas | GT | A | B | C)
python -m src.inference --config config.yaml --mode compare \
    --dataset pku --split test
```

### Reproduce the paper's tables

```bash
# Print the command plan for a table (dry run)
python -m src.experiments --table table1

# Execute it (needs GPU + data)
python -m src.experiments --table table1 --execute

# Aggregate per-seed head-to-head JSONs into a significance summary
python -m src.experiments --aggregate "outputs/head2head_*.json" \
    --output outputs/variance_summary.json
```

---

## Configuration

`config.yaml` is the single source of truth read by all ten modules. The one
experimental knob is the preference source; everything else is fixed across arms.

```yaml
rgpo:
  pref_source:    B      # A = learned reward | B = RGPO (ours) | C = hybrid
  grid_resolution: 16    # G for the G×G occupancy grid (agreement, spatial term)
  w_spatial:       1.0
  w_structural:    1.0
  rgpo_weight:     0.5   # λ for source C: (1-λ)·Â + λ·B̂

train_dpo:
  beta:            0.1
  margin_type:     dynamic        # 'dynamic' → DMPO f(δ)=2·sinh(δ); 'fixed' → DPO
  num_candidates_per_canvas: 8
  min_margin:      1.0e-3         # drop pairs whose normalised δ < this
```

Key defaults (full list in `config.yaml`): Qwen2.5-VL-7B + LoRA r64/α128 on
q/v projections, vision encoder frozen; DreamSim + FAISS-IVF retrieval with
K = 16 (4 exemplars shown in the prompt); SFT 15 epochs @ 2e-5, RGPO 5 epochs @
5e-6; judge = Qwen2.5-VL-72B; bf16.

---

## Reproducibility

Seeds are fixed and per-seed checkpoints are scoped by dataset, seed, and
preference source. The significance protocol treats each seed as one observation
(paired *t*-tests with Holm–Bonferroni correction; five seeds `{13,21,42,87,100}`
in the paper).

**Please read `AUDIT_LOG.md` before trusting any number you regenerate.** It
records the bugs found and fixed during review — including several that would
have corrupted results if left in (e.g., occlusion/readability metrics reading
unset keys and returning identically 0; the SFT learning-rate schedule decaying
too early; and the preference-pair builder skipping neighbour-less canvases for
B/C but not A, which would have broken the controlled comparison). Any numbers
must come from runs made **after** those fixes. The file also documents three
honest limitations that are *not* code defects (summed-logprob DPO length bias;
empty-candidate agreement floor; bucketizer boundary convention).

---

## Method, at a glance

The agreement score combines a **spatial** term (mean-absolute divergence of the
candidate's G×G occupancy grid from the neighbourhood mean occupancy) and a
**structural** term (normalised ℓ₁ deviation of per-category element counts):

```
Agr(L, N) = 1 − [ w_sp · D_spatial + w_st · D_structural ] / (w_sp + w_st)   ∈ [0,1]
```

It satisfies three properties the implementation was probed for: **order
invariance** (occupancy and counts are symmetric over elements), **no reward for
memorisation** (the target is the neighbourhood *mean*, so copying one neighbour
scores < 1), and **overlap awareness** (the grid penalises piling even when
centroids match). The per-candidate quality feeds a best-vs-worst pairing, a
span-normalised strength `δ ∈ (0,1]`, and the dynamic margin `f(δ) = 2·sinh(δ)`
subtracted inside the logistic DPO loss.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{zhou2026rgpo,
  title   = {LLM-Guided Retrieval-Augmented Layout Generation for
             Content-Aware Poster Design},
  author  = {Zhou, You and Chen, Yu and Peng, Lang},
  journal = Waiting acceptance,
  year    = {2026},
  note    = {Retrieval-Grounded Preference Optimization (RGPO)}
}
```

Please also cite the resources this work builds on: PKU PosterLayout and
CGL-Dataset V2 (benchmarks), RALF (retrieval-augmented layout, preprocessing and
splits), Qwen2.5-VL (backbone), LoRA, DreamSim, and FAISS.

## License

Code released under the **MIT License** (see `LICENSE`). The datasets, retrieval
indices, and model weights referenced above are **not** covered by this license
and remain subject to their respective terms.
