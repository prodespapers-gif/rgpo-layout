"""
experiments.py — Experiment Orchestration & Paper Tables (the capstone)
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [10] of 10.

What it does
------------
One entry point that produces every table in the paper and, as the centerpiece,
the multi-seed A/B/C head-to-head that decides the RGPO claim. It subsumes the
two original orchestration artifacts:
  * run_experiments.sh  -> the table-by-table training/eval plan, as a typed
    registry of table specs (Tables 1-4 + the head-to-head).
  * aggregate_seeds.py  -> the pre-registered comparisons and across-seed
    statistics, REFRAMED from "Full vs SFT/fixed" to the arm contrasts
    B-vs-A and C-vs-A.

Single source of truth for statistics
--------------------------------------
evaluate.py now owns paired_t_test, holm_bonferroni, tost_equivalence, and
compare_arms. This file IMPORTS and USES those rather than re-implementing them,
so there is exactly one copy of the significance math and no chance of the
capstone drifting from what the evaluation reports. (That is why the original
standalone aggregate_seeds.py is folded in here instead of kept separate.)

The centerpiece: multi-seed A/B/C
---------------------------------
For each seed in SEEDS, train each arm (A=learned reward, B=RGPO, C=hybrid) via
train.py's --pref-source, evaluate each on PKU, and collect per-seed metric
values. Then, per metric, call evaluate.compare_arms(..., baseline="A") to get
B-vs-A and C-vs-A paired t-tests with a Holm correction across the arms, plus a
TOST equivalence read where the claim is "no worse than". This is the make-or-
break result: if B does not beat (or at least match) A while removing the
reward model, the thesis weakens — and the harness will say so plainly.

Honesty of the harness
----------------------
Nothing here fabricates numbers. Real tables require a GPU, the datasets, and
trained checkpoints; without them each cell is recorded as PENDING and the
command that would fill it is emitted. The aggregation/significance layer is the
SAME code the self-test exercises on synthetic per-seed inputs, so the
statistical verdicts are trustworthy the moment real eval JSONs exist.

Usage
-----
  python -m src.experiments --table all            # full plan (prints commands)
  python -m src.experiments --table head2head      # just the A/B/C centerpiece
  python -m src.experiments --aggregate "outputs/seed*_*.json" \
                            --output outputs/variance_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

from data import load_config
# THE single source of significance math — imported, never re-implemented.
from evaluate import (UNAVAILABLE, compare_arms, holm_bonferroni,
                      paired_t_test, tost_equivalence)

logger = logging.getLogger(__name__)


# ======================================================================
# 0. EXPERIMENT CONSTANTS  (mirrors run_experiments.sh)
# ======================================================================

SEEDS = (13, 21, 42, 87, 100)          # multi-seed variance study seeds
ARMS = ("A", "B", "C")                  # learned-reward / RGPO / hybrid
PRIMARY_DATASET = "pku"                 # ablation + head-to-head dataset
DATASETS = ("pku", "cgl")
METRICS = ("fid", "occ", "rea", "und", "ove", "max_iou", "win_rate", "valid_rate")

# Direction each metric improves (for interpreting a signed mean difference).
LOWER_IS_BETTER = {"fid", "occ", "rea", "ove"}
HIGHER_IS_BETTER = {"und", "max_iou", "win_rate", "valid_rate"}

# Pre-declared, substantive equivalence margins (justified in the paper):
#   FID  +/-0.30  (below run-to-run noise / not perceptually meaningful here)
#   Win  +/-0.02  (a <2pp win-rate gap is not meaningful)
EQUIV_MARGIN = {"fid": 0.30, "win_rate": 0.02}


# ======================================================================
# 1. TABLE REGISTRY  (run_experiments.sh, as typed specs)
#    Each spec lists the eval cells it needs and the train/eval commands that
#    would produce them. Running the plan emits commands (and executes them when
#    --execute is given and a GPU/data are present); it never invents results.
# ======================================================================

def _eval_cmd(dataset: str, split: str, task: str, seed: int,
              output: str, overrides: Optional[list] = None,
              checkpoint: Optional[str] = None) -> list:
    cmd = [sys.executable, "-m", "evaluate", "--dataset", dataset,
           "--split", split, "--task", task, "--seed", str(seed),
           "--config", "config.yaml", "--output", output]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if overrides:
        cmd += ["--override", *overrides]
    return cmd


def _train_cmd(dataset: str, stage: str, seed: int,
               pref_source: Optional[str] = None,
               overrides: Optional[list] = None) -> list:
    cmd = [sys.executable, "-m", "train", "--stage", stage,
           "--dataset", dataset, "--seed", str(seed), "--config", "config.yaml"]
    if pref_source:
        cmd += ["--pref-source", pref_source]
    if overrides:
        cmd += ["--override", *overrides]
    return cmd


def table1_main(seed: int = 42) -> dict:
    """Table 1: main results, PKU+CGL, test + unannotated, unconstrained."""
    cells, cmds = [], []
    for ds in DATASETS:
        for split in ("test", "unannotated"):
            out = f"outputs/eval_{ds}_{split}_unconstrained.json"
            cells.append({"dataset": ds, "split": split, "metric_file": out})
            cmds.append(_eval_cmd(ds, split, "unconstrained", seed, out))
    return {"table": "Table1_main", "cells": cells, "commands": cmds}


def table2_constrained(seed: int = 42) -> dict:
    """Table 2: the 6 constrained tasks (evaluate.py --task all) per dataset."""
    cells, cmds = [], []
    for ds in DATASETS:
        out = f"outputs/eval_{ds}_test_all_tasks.json"
        cells.append({"dataset": ds, "split": "test", "task": "all", "metric_file": out})
        cmds.append(_eval_cmd(ds, "test", "all", seed, out))
    return {"table": "Table2_constrained", "cells": cells, "commands": cmds}


def table3_ablations(seed: int = 42) -> dict:
    """Table 3: ablations on PKU (retrieval, CoT, SFT-only, backbone, K, margin, rank)."""
    ds = PRIMARY_DATASET
    cmds, cells = [], []

    def add(tag, overrides=None, checkpoint=None, train=None):
        out = f"outputs/ablation_{tag}.json"
        cells.append({"tag": tag, "metric_file": out})
        if train is not None:
            cmds.append(train)
        cmds.append(_eval_cmd(ds, "test", "unconstrained", seed, out,
                              overrides=overrides, checkpoint=checkpoint))

    add("no_retrieval", overrides=["retrieval.num_neighbors=0",
                                   "retrieval.max_retrieved_in_prompt=0"],
        train=_train_cmd(ds, "sft", seed,
                         overrides=["retrieval.num_neighbors=0",
                                    "retrieval.max_retrieved_in_prompt=0"]))
    add("no_cot", overrides=["model.use_cot_reasoning=false"],
        train=_train_cmd(ds, "sft", seed,
                         overrides=["model.use_cot_reasoning=false"]))
    add("sft_only", checkpoint=f"checkpoints/{ds}_s{seed}_sft_best")
    for sim in ("dreamsim", "clip", "dinov2"):
        add(f"retrieval_{sim}", overrides=[f"retrieval.similarity_model={sim}"],
            train=_train_cmd(ds, "sft", seed,
                             overrides=[f"retrieval.similarity_model={sim}"]))
    # K co-varies num_neighbors AND max_retrieved_in_prompt (in-prompt exemplars).
    for k in (1, 4, 8, 16, 32):
        add(f"K{k}", overrides=[f"retrieval.num_neighbors={k}",
                                f"retrieval.max_retrieved_in_prompt={k}"])
    add("fixed_margin", overrides=["train_dpo.margin_type=fixed"],
        train=_train_cmd(ds, "align", seed, pref_source="B",
                         overrides=["train_dpo.margin_type=fixed"]))
    for rank in (16, 32, 64, 128):
        add(f"lora_r{rank}", overrides=[f"model.lora.rank={rank}",
                                        f"model.lora.alpha={rank * 2}"],
            train=_train_cmd(ds, "sft", seed,
                             overrides=[f"model.lora.rank={rank}",
                                        f"model.lora.alpha={rank * 2}"]))
    return {"table": "Table3_ablations", "cells": cells, "commands": cmds}


def table4_cross_dataset(seed: int = 42) -> dict:
    """Table 4: cross-dataset (CGL->PKU, PKU->CGL) on unannotated, RGPO arm."""
    cells, cmds = [], []
    pairs = [("cgl", "pku"), ("pku", "cgl")]
    for train_ds, eval_ds in pairs:
        ckpt = f"checkpoints/{train_ds}_s{seed}_B_align_best"
        out = f"outputs/cross_{train_ds}_to_{eval_ds}.json"
        cells.append({"train_on": train_ds, "eval_on": eval_ds, "metric_file": out})
        cmds.append(_eval_cmd(eval_ds, "unannotated", "unconstrained", seed, out,
                              checkpoint=ckpt))
    return {"table": "Table4_cross_dataset", "cells": cells, "commands": cmds}


def head2head_plan(seeds=SEEDS, dataset=PRIMARY_DATASET) -> dict:
    """The centerpiece: per seed, train + eval each arm A/B/C; collect per-seed JSONs.

    Output filenames follow seed{S}_arm{X}.json so the aggregator can pair them
    seed-by-seed. This is the plan whose results feed compare_arms.
    """
    cmds, cells = [], []
    for s in seeds:
        # SFT is shared across arms (identical model); train once per seed.
        cmds.append(_train_cmd(dataset, "sft", s))
        for arm in ARMS:
            cmds.append(_train_cmd(dataset, "align", s, pref_source=arm))
            out = f"outputs/seed{s}_arm{arm}.json"
            cells.append({"seed": s, "arm": arm, "metric_file": out})
            cmds.append(_eval_cmd(dataset, "test", "unconstrained", s, out,
                                  checkpoint=f"checkpoints/{dataset}_s{s}_{arm}_align_best"))
    return {"table": "HeadToHead_ABC", "seeds": list(seeds),
            "dataset": dataset, "cells": cells, "commands": cmds}


TABLES = {
    "table1": table1_main,
    "table2": table2_constrained,
    "table3": table3_ablations,
    "table4": table4_cross_dataset,
    "head2head": head2head_plan,
}


# ======================================================================
# 2. PLAN EXECUTION  (emit commands; optionally run them)
# ======================================================================

def _cell_status(cell: dict) -> str:
    """A cell is DONE if its metric_file exists, else PENDING."""
    mf = cell.get("metric_file")
    return "DONE" if (mf and os.path.exists(mf)) else "PENDING"


def run_plan(plan: dict, execute: bool = False) -> dict:
    """Print a table's plan; run its commands only if execute=True.

    Without --execute (the default, and the only option offline), this lists the
    exact commands that produce each cell and marks which outputs already exist.
    Cells are never fabricated.
    """
    name = plan["table"]
    logger.info("=" * 72)
    logger.info("PLAN: %s  (%d cells, %d commands)", name,
                len(plan.get("cells", [])), len(plan.get("commands", [])))
    logger.info("=" * 72)
    for cell in plan.get("cells", []):
        logger.info("  [%s] %s", _cell_status(cell), cell.get("metric_file"))
    if not execute:
        logger.info("-- commands (not executed; pass --execute to run) --")
        for cmd in plan.get("commands", []):
            logger.info("    %s", " ".join(cmd))
        return {"table": name, "executed": False,
                "pending": [c for c in plan.get("cells", [])
                            if _cell_status(c) == "PENDING"]}
    # Execution path (requires GPU + data; each command is a real subprocess).
    for cmd in plan.get("commands", []):
        logger.info("RUN: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return {"table": name, "executed": True}


# ======================================================================
# 3. ACROSS-SEED AGGREGATION  (aggregate_seeds.py, reframed to arms)
#    Uses evaluate.py's paired_t_test / tost_equivalence / holm_bonferroni /
#    compare_arms — one source of truth for the statistics.
# ======================================================================

# Per-seed filename pattern for the head-to-head: seed{SEED}_arm{ARM}.json
_ARM_RE = re.compile(r"seed(?P<seed>\d+)_arm(?P<arm>[ABC])\.json$")
# Generic per-seed config pattern (for ablation seeds): seed{SEED}_{CONFIG}.json
_CFG_RE = re.compile(r"seed(?P<seed>\d+)_(?P<config>.+)\.json$")


def _metric_value(blob: dict, metric: str):
    """Pull a finite, available metric value from an eval JSON (or None)."""
    val = blob.get(f"{metric}_mean", blob.get(metric))
    if val is None or not np.isfinite(val) or val == UNAVAILABLE:
        return None
    return float(val)


def load_arm_results(paths) -> dict:
    """Load head-to-head JSONs into {metric: {arm: [per-seed values (seed-ordered)]}}.

    Values are ordered by seed so compare_arms pairs arms seed-by-seed. A seed
    missing a metric for any arm is dropped for THAT metric across all arms, so
    the paired tests stay balanced.
    """
    # arm -> seed -> {metric: value}
    table = defaultdict(lambda: defaultdict(dict))
    seeds = set()
    for path in paths:
        m = _ARM_RE.search(os.path.basename(path))
        if not m:
            continue
        seed, arm = int(m.group("seed")), m.group("arm")
        seeds.add(seed)
        with open(path) as f:
            blob = json.load(f)
        for metric in METRICS:
            v = _metric_value(blob, metric)
            if v is not None:
                table[arm][seed][metric] = v

    out = {}
    seeds_sorted = sorted(seeds)
    for metric in METRICS:
        # Seeds where ALL arms have this metric (balanced pairing).
        usable = [s for s in seeds_sorted
                  if all(metric in table[a].get(s, {}) for a in ARMS if table.get(a))]
        present_arms = [a for a in ARMS if table.get(a)]
        if not usable or len(present_arms) < 2:
            continue
        out[metric] = {a: [table[a][s][metric] for s in usable] for a in present_arms}
    return out, seeds_sorted


def aggregate_head2head(paths) -> dict:
    """Across-seed mean/std/CI per arm + B-vs-A and C-vs-A significance per metric.

    For each metric: report each arm's across-seed summary, then call
    evaluate.compare_arms(baseline="A") for paired t-tests + a Holm correction
    across the non-baseline arms, plus a TOST equivalence read using the
    pre-declared margin (so "B is no worse than A" is a real claim, not p>=0.05).
    """
    by_metric, seeds_sorted = load_arm_results(paths)
    if not by_metric:
        return {"status": "NO_DATA", "n_files": len(list(paths))}

    summary = {"seeds": seeds_sorted, "per_metric": {}}
    for metric, arms_vals in by_metric.items():
        # Per-arm across-seed summary.
        per_arm = {}
        for arm, vals in arms_vals.items():
            arr = np.asarray(vals, float)
            n = arr.size
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            # 95% CI via the same paired_t_test machinery (CI of arm-vs-itself=0,
            # so compute a one-sample t-interval directly here).
            if n > 1:
                import math
                try:
                    from scipy import stats
                    tcrit = float(stats.t.ppf(0.975, n - 1))
                except ImportError:
                    tcrit = 1.96
                sem = std / math.sqrt(n)
                ci = (mean - tcrit * sem, mean + tcrit * sem)
            else:
                ci = (mean, mean)
            per_arm[arm] = {"n_seeds": n, "mean": mean, "std": std,
                            "ci95": [ci[0], ci[1]]}

        result = {"per_arm": per_arm, "direction":
                  ("lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better")}

        # Significance vs baseline A (only if A is present).
        if "A" in arms_vals:
            cmp = compare_arms(arms_vals, metric=metric, baseline="A")
            # Interpret each arm's signed difference by the metric's direction.
            interpreted = {}
            for arm, tt in cmp["tests"].items():
                md = tt["mean_diff"]                 # arm - A
                if metric in LOWER_IS_BETTER:
                    better = md < 0                  # arm lower than A => better
                else:
                    better = md > 0
                holm = cmp["holm"].get(arm, {})
                sig = bool(holm.get("reject_null", False))
                # Equivalence ("no worse than A") where a margin is declared.
                eq = None
                if metric in EQUIV_MARGIN:
                    eq = tost_equivalence(arms_vals[arm], arms_vals["A"],
                                          margin=EQUIV_MARGIN[metric])
                interpreted[arm] = {
                    "mean_diff_vs_A": md, "p": tt["p"],
                    "holm_reject": sig,
                    "better_than_A": bool(better),
                    "significant_and_better": bool(sig and better),
                    "equivalence": eq,
                    "verdict": _verdict(metric, md, sig, better, eq),
                }
            result["vs_A"] = interpreted
        summary["per_metric"][metric] = result
    return summary


def _verdict(metric: str, mean_diff: float, sig: bool, better: bool, eq) -> str:
    """Plain-language read of one arm-vs-A contrast on one metric."""
    arrow = "lower" if metric in LOWER_IS_BETTER else "higher"
    want = "lower" if metric in LOWER_IS_BETTER else "higher"
    if sig and better:
        return f"beats A (significantly {want}, Holm-corrected)"
    if sig and not better:
        return f"significantly WORSE than A (wrong direction: {arrow} is better)"
    if eq is not None and eq.get("equivalent"):
        return f"no worse than A (TOST-equivalent within +/-{eq['margin']})"
    return "no significant difference from A (and not shown equivalent)"


# ======================================================================
# 4. CLI
# ======================================================================

def _print_head2head_digest(summary: dict):
    print("=" * 78)
    print("A/B/C HEAD-TO-HEAD  (baseline = A: learned reward)")
    print("=" * 78)
    if summary.get("status") == "NO_DATA":
        print("  No head-to-head eval JSONs found (expected seed{S}_arm{A,B,C}.json).")
        print("  Run:  python -m src.experiments --table head2head --execute")
        return
    print(f"  seeds: {summary['seeds']}")
    for metric, res in summary["per_metric"].items():
        print(f"\n[{metric}]  ({res['direction']})")
        for arm, s in res["per_arm"].items():
            print(f"   arm {arm}: mean={s['mean']:.4f} std={s['std']:.4f} "
                  f"95%CI=[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] (n={s['n_seeds']})")
        for arm, info in res.get("vs_A", {}).items():
            flag = "PASS" if info["significant_and_better"] else \
                   ("EQUIV" if (info["equivalence"] and info["equivalence"].get("equivalent"))
                    else "----")
            print(f"   [{flag}] {arm} vs A: d={info['mean_diff_vs_A']:+.4f} "
                  f"p={info['p']:.4f} Holm={'rej' if info['holm_reject'] else 'no'} "
                  f"-> {info['verdict']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RGPO experiment orchestration + aggregation")
    parser.add_argument("--table", default=None,
                        choices=list(TABLES) + ["all"],
                        help="Which table plan to print/run.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run the plan's commands (needs GPU+data).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single-seed tables.")
    parser.add_argument("--aggregate", default=None,
                        help="Glob of per-seed head-to-head JSONs to aggregate.")
    parser.add_argument("--output", default="outputs/variance_summary.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.aggregate:
        paths = sorted(glob.glob(args.aggregate))
        if not paths:
            logger.error("No files matched: %s", args.aggregate)
            return
        summary = aggregate_head2head(paths)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        _print_head2head_digest(summary)
        print(f"\nWrote {args.output}")
        return

    if args.table:
        plans = (list(TABLES) if args.table == "all" else [args.table])
        for name in plans:
            builder = TABLES[name]
            plan = builder(seed=args.seed) if name != "head2head" else builder()
            run_plan(plan, execute=args.execute)
        return

    parser.print_help()


# ======================================================================
# 5. SELF-TEST  (run: python -m src.experiments)
#    Verifies the table registry, the head-to-head plan shape, and the FULL
#    aggregation+significance path on SYNTHETIC per-seed JSONs — the same
#    evaluate.compare_arms/TOST code the real run uses. No GPU, no data.
# ======================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.WARNING)  # quiet the plan prints in test
    print("=" * 64)
    print("experiments.py SELF-TEST  (registry + plan + aggregation)")
    print("=" * 64)

    # (a) Every table builder returns a well-formed plan with commands.
    for name, builder in TABLES.items():
        plan = builder() if name == "head2head" else builder(seed=42)
        assert "commands" in plan and plan["commands"], name
        assert "cells" in plan, name
    print("  [ok] all 5 table builders produce non-empty plans")

    # (b) Head-to-head plan: SFT shared per seed, align per arm, eval per arm.
    h2h = head2head_plan(seeds=(13, 42), dataset="pku")
    arms_in_cells = {c["arm"] for c in h2h["cells"]}
    assert arms_in_cells == {"A", "B", "C"}
    # checkpoints are source-scoped (no A/B/C collision)
    joined = " ".join(" ".join(c) for c in h2h["commands"])
    assert "pku_s13_B_align_best" in joined and "pku_s42_A_align_best" in joined
    # one SFT per seed (shared), three aligns per seed (one per arm)
    n_sft = joined.count("--stage sft")
    n_align = joined.count("--stage align")
    assert n_sft == 2 and n_align == 6, (n_sft, n_align)
    print(f"  [ok] head2head: {n_sft} shared SFT + {n_align} per-arm aligns, source-scoped ckpts")

    # (c) Table 3 K-sweep co-varies BOTH retrieval fields (the dilution control).
    t3 = table3_ablations(seed=42)
    t3cmds = " ".join(" ".join(c) for c in t3["commands"])
    assert "retrieval.num_neighbors=16" in t3cmds
    assert "retrieval.max_retrieved_in_prompt=16" in t3cmds
    print("  [ok] Table 3 K-sweep co-varies num_neighbors AND max_retrieved_in_prompt")

    # (d) Cell status: PENDING when the metric file is absent.
    fake_cell = {"metric_file": "/nope/does_not_exist.json"}
    assert _cell_status(fake_cell) == "PENDING"
    print("  [ok] cell status PENDING when output missing (never fabricated)")

    # (e) THE aggregation path on synthetic per-seed JSONs: B beats A on FID.
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    seeds = (13, 21, 42, 87, 100)
    for s in seeds:
        a_fid = 10.0 + rng.normal(0, 0.05)
        b_fid = 9.1 + rng.normal(0, 0.05)     # B consistently lower (better)
        c_fid = 9.6 + rng.normal(0, 0.05)
        a_win = 0.50 + rng.normal(0, 0.005)
        b_win = 0.52 + rng.normal(0, 0.005)
        c_win = 0.51 + rng.normal(0, 0.005)
        for arm, fid, win in (("A", a_fid, a_win), ("B", b_fid, b_win), ("C", c_fid, c_win)):
            with open(os.path.join(tmp, f"seed{s}_arm{arm}.json"), "w") as f:
                json.dump({"fid_mean": fid, "win_rate_mean": win,
                           "ove_mean": 0.01, "valid_rate_mean": 1.0}, f)
    paths = sorted(glob.glob(os.path.join(tmp, "seed*_arm*.json")))
    summary = aggregate_head2head(paths)
    assert summary["per_metric"]["fid"]["per_arm"]["B"]["mean"] < \
           summary["per_metric"]["fid"]["per_arm"]["A"]["mean"]
    b_vs_a_fid = summary["per_metric"]["fid"]["vs_A"]["B"]
    assert b_vs_a_fid["mean_diff_vs_A"] < 0           # B lower FID than A
    assert b_vs_a_fid["significant_and_better"] is True
    print(f"  [ok] aggregation: B beats A on FID "
          f"(d={b_vs_a_fid['mean_diff_vs_A']:+.3f}, {b_vs_a_fid['verdict']})")

    # (f) Win-rate direction is handled (higher is better) + TOST equivalence read.
    b_vs_a_win = summary["per_metric"]["win_rate"]["vs_A"]["B"]
    assert b_vs_a_win["better_than_A"] is True         # B higher win-rate
    assert b_vs_a_win["equivalence"] is not None       # margin declared for win_rate
    print(f"  [ok] win_rate: higher-is-better direction + TOST read present "
          f"({b_vs_a_win['verdict']})")

    # (g) Holm correction is applied across arms (B and C both vs A).
    assert "B" in summary["per_metric"]["fid"]["vs_A"]
    assert "C" in summary["per_metric"]["fid"]["vs_A"]
    print("  [ok] B-vs-A and C-vs-A both present with Holm correction across arms")

    # (h) No-data path is graceful (empty glob -> NO_DATA, not a crash).
    empty = aggregate_head2head([])
    assert empty["status"] == "NO_DATA"
    print("  [ok] empty input -> NO_DATA (graceful, no crash)")

    print("=" * 64)
    print("  ALL experiments.py SELF-TESTS PASSED")
    print("=" * 64)
