# Design-Conformity Learning (DCL)

**A label-free, calibrated, and domain-transferable model of graphic-layout quality — and a controlled audit of what the field's standard layout metrics actually measure.**

This repository accompanies the paper *“Is ‘Good Layout’ Domain-Invariant? Label-Free, Calibrated Design-Conformity Learning for Graphic-Layout Quality Assessment.”* It reproduces every table and figure in the paper.

Layout quality is treated here not as a hand-crafted metric, a large human-feedback dataset, or an MLLM judge, but as a **self-supervised, learnable, and partially domain-invariant structural property of professional design.** Real designs are the positive class; a taxonomy of nine severity-graded structural corruptions produces negatives; a calibrated gradient-boosted classifier learns to tell them apart, and its predicted probability is the **conformity score** *q* ∈ [0, 1]. Because each corruption’s severity is known, the same machinery doubles as a rigorous validity study of the standard metrics.

Everything runs on a **single CPU core with no GPU and no deep-learning framework** — only NumPy, SciPy, scikit-learn, and Matplotlib.

---

## Key findings (five seeds: 13, 21, 42, 87, 100)

Significance is a seed-paired two-sided *t*-test with Holm–Bonferroni correction across the five headline hypotheses.

| # | Hypothesis | Result |
|---|---|---|
| **H1** | Conformity is learnable without labels | Real-vs-corrupted **AUC 0.866**, the strongest *blind* discriminator, beating every hand-crafted metric (coverage 0.713, alignment 0.658, overlap 0.592, underlay 0.534); *p*<sub>Holm</sub> = 2 × 10⁻⁵ |
| **H2** | The learned score is uniquely broad | Responds to all nine corruption types (**breadth 0.659**) where each metric responds to only a few (coverage 0.541, alignment 0.317, overlap 0.312, underlay 0.079); *p*<sub>Holm</sub> = 8 × 10⁻⁵ |
| **H3** | The metric panel is non-redundant | Effective dimensionality **≈ 2.9 / 4**, mean VIF **≈ 1.44**; underlay effectiveness saturates and is dropped |
| **H4** | The score is well-calibrated | **ECE 0.034**, AURC 0.051, Brier 0.115, accuracy 0.828; abstaining on low-confidence layouts lowers error monotonically |
| **H5** | “Good layout” is partly domain-invariant | Geometry transfers Crello↔RICO (**≈ 0.63 AUC**, above chance) with a bounded **≈ 0.27** gap; the category-composition block adds nothing across domains |
| **H6** | Formats transfer, domains only partly | Cross-format gap **≈ 0** vs cross-domain gap **0.271**; *p*<sub>Holm</sub> = 9 × 10⁻⁴ |
| **H7** | Aesthetics is a distinct, noisy construct | On AVA, **54%** of images have a rating standard deviation above 1.4 — motivating an objective structural measure |
| **H8** | Best blind ranker | Selects the real design over its corruptions **79.6%** of the time; *p*<sub>Holm</sub> = 0.032 |

**The transferable signal is geometric, not category-structural.** Permutation importance is **0.185** for the geometric block versus **−0.001** for the structural block, and every top feature is geometric (nearest-neighbour spacing, largest-element area, off-canvas bleed, occupancy entropy). The one intentional null — structural features adding nothing in-domain — is what pins the finding down.

Complete tables with confidence intervals are in [`RESULTS.md`](RESULTS.md).

---

## Repository structure

Nine single-purpose modules, each with a CPU-only `__main__` self-test:

```
data.py          Order-invariant Layout representation + dataset loaders (Crello, RICO, AVA)
features.py      Geometric (transferable) + structural (domain-specific) feature blocks
corruptions.py   Nine severity-graded structural corruptions (the self-supervision engine)
model.py         Calibrated conformity scorer (HistGBDT + isotonic), ECE / selective prediction
metrics.py       The audited layout metrics + a verified statistics toolkit
validity.py      H1 discrimination, H2 sensitivity, H3 redundancy, H7 discriminant-vs-aesthetics
transfer.py      H5 cross-domain and H6 cross-format transfer
experiments.py   Orchestration across seeds: aggregation, significance, table emission
figures.py       The paper's figures, rendered headlessly

config.yaml        Canonical settings reference (documents every module default)
requirements.txt   Runtime dependencies (CPU-only)
DATA.md            Dataset sources, sizes, licenses, and placement
RESULTS.md         Full five-seed results with 95% intervals
LICENSE            MIT
```

Modules import only earlier ones — `data → {features, corruptions} → {metrics, model} → {validity, transfer} → experiments → figures` — and all nine import together cleanly.

---

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `matplotlib`, `joblib`. No GPU, CUDA, or deep-learning framework is required.

---

## Data

The three datasets, their sources, sizes, and licenses, and their exact placement under `./data/` are documented in [`DATA.md`](DATA.md):

```
data/
├── test-00000-of-00004.parquet     # Crello graphic-design domain (primary)
├── AVA.txt                         # AVA photo ratings (H7 only)
└── iconnet/
    ├── test.json                   # RICO-Semantics mobile-UI domain (transfer)
    └── val.json
```

The loaders resolve this directory automatically (`$DCL_DATA_ROOT`, then `./data`), so running any script from the repository root needs no configuration:

```bash
export DCL_DATA_ROOT=/path/to/data     # only if your data lives elsewhere
```

Crello is `cyberagent/crello` (Hugging Face); RICO-Semantics is `google-research-datasets/rico_semantics` (`iconnet` split); AVA is the Aesthetic Visual Analysis ratings file. AVA contains photographs, not layouts, and is used **only** as construct / noise-ceiling evidence for H7 — never as a source of layout geometry.

---

## Quick start

Score a real layout and its corruption in a few lines:

```python
from pathlib import Path
from data import load_crello, split_layouts, DEFAULT_FILES
import corruptions as C
from model import ConformityModel

root = Path("data")                                       # or set $DCL_DATA_ROOT
crello = load_crello(root / DEFAULT_FILES["crello"])
train, test = split_layouts(crello, frac=0.8, seed=42)

# self-supervision: real layouts (+) vs graded corruptions (−)
pairs = C.make_training_pairs(train, n_negatives=4, seed=42)

# calibrated conformity scorer over geometric + structural features
model = ConformityModel(groups=("geo", "struct"), grid=16,
                        random_state=42, train_domain="crello").fit(pairs.layouts, pairs.labels)

print(f"real      q = {model.score(test[0]):.3f}")        # conformity in [0, 1]
rng = __import__('numpy').random.default_rng(0)
print(f"corrupted q = {model.score(C.CORRUPTIONS['overlap_inject'](test[0], 0.8, rng)):.3f}")

model.save("dcl.joblib")                                  # checkpoint; ConformityModel.load(...) restores
```

---

## Reproducing the paper

**All tables, all five seeds** (writes `results/results.json` and `results/tables.csv`):

```bash
python experiments.py --table all --execute
```

Preview the run plan without executing anything:

```bash
python experiments.py --plan
```

Run a single hypothesis, a faster configuration, or re-aggregate saved runs:

```bash
python experiments.py --table t2 --execute            # just H2
python experiments.py --table all --execute --quick   # small settings, 2 seeds
python experiments.py --aggregate "results/*.json"    # re-aggregate without recomputing
```

Table ↔ hypothesis map: `t1`→H1 discrimination, `t2`→H2 sensitivity, `t3`→H3 redundancy, `t4`→H4 calibration, `t5`→H5 cross-domain, `t6`→H6 cross-format, `t7`→H7 aesthetics, `t8`→H8 ranking.

**Figures** (rendered headlessly to `figures/`):

```bash
python figures.py     # self-test renders every figure to a temporary directory
python -c "import figures, pathlib; figures.make_all_figures(pathlib.Path('data'), 'figures')"
```

**Self-tests.** Every module verifies itself on the real data and prints `ALL CHECKS PASSED`:

```bash
python data.py        python features.py    python corruptions.py
python metrics.py     python model.py       python validity.py
python transfer.py    python experiments.py python figures.py
```

---

## Hardware and reproducibility notes

- **CPU-only.** No GPU, CUDA, or deep-learning framework is used or needed. The full study runs on one core within a few gigabytes of RAM; the Crello parquet is read one row group at a time and its embedded image bytes are never loaded.
- **Deterministic.** Every stochastic step — corruptions, the booster, calibration folds, permutation importance, and bootstrap intervals — is seeded. Given the same seeds, results are reproducible up to floating-point summation order, which the order-invariant code paths control with `math.fsum` where it matters.
- **Honest scope.** Reference-based scores such as `max_iou` require the ground-truth layout and are reported for context only — they are **not** blind scorers and are excluded from every “best blind metric” comparison. FID and occlusion/readability require a neural feature extractor and saliency maps and are deliberately out of scope; a clearly-labelled feature-space Fréchet distance is provided in their place.

---

## Citation

This repository is released for peer review; identifying details are withheld under the review process and a full citation will be added on acceptance.

```bibtex
@article{dcl_anonymous,
  title  = {Is ``Good Layout'' Domain-Invariant? Label-Free, Calibrated
            Design-Conformity Learning for Graphic-Layout Quality Assessment},
  author = {Anonymous},
  note   = {Under review},
  year   = {2025}
}
```

---

## License

Code is released under the **MIT License** (see [`LICENSE`](LICENSE)). Dataset licenses are listed in [`DATA.md`](DATA.md): CDLA-Permissive-2.0 for Crello, CC BY-SA 4.0 for RICO-Semantics, and academic-use terms for AVA (Murray et al., CVPR 2012).
