"""
train.py — Two-Stage Training Driver (SFT -> Retrieval-Grounded Alignment)
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [7] of 10.

Two stages
----------
  Stage 1 - SFT: train the LoRA-adapted MLLM to emit layouts from canvas +
    retrieved references (cross-entropy on layout tokens).
  Stage 2 - Alignment: build preference pairs with the SFT model and the
    SELECTED preference source (A/B/C), then fine-tune with DMPO against the
    frozen SFT reference.

The only experimental knob: --pref-source {A,B,C}
-------------------------------------------------
Stage 2 calls ``preferences.build_preference_dataset(model, ds, cfg,
pref_source)``. That single argument chooses source A (learned reward),
B (RGPO, the contribution), or C (hybrid) — and NOTHING else in this driver
changes between the three arms. The loss (DMPO), the optimiser, the schedule,
the reference model, and the candidate sampler are identical. This is the
control the paper's central comparison rests on; see preferences.py (the one
divergence point) and losses.py (byte-identical loss across arms).

Checkpoints are scoped by dataset x seed x source
-------------------------------------------------
``checkpoint_name`` includes the dataset, the seed, AND (for alignment) the
preference source, so multi-seed A/B/C runs never overwrite each other — the
precondition for reporting mean +/- std and significance across the arms.

Usage
-----
  python -m src.train --stage sft   --config config.yaml
  python -m src.train --stage align --config config.yaml --pref-source B
  python -m src.train --stage full  --config config.yaml --pref-source B
  # 'dpo' is accepted as a hidden alias of 'align' for backward compatibility.

References
----------
  - SFT: instruction tuning (Chung et al., 2022)
  - DPO: Rafailov et al., NeurIPS 2023 ; DMPO: Lu et al. (Uni-Layout), ACM MM 2025
  - LoRA: Hu et al., ICLR 2022
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import time

import numpy as np

# data.py owns config loading and the dataset; preferences.py owns pair
# construction (delegating scoring to rgpo.py). torch is imported lazily inside
# the functions that train, so naming/override logic is testable without it.
from data import PosterLayoutDataset, load_config
from losses import DMPOLoss, SFTLoss, create_dpo_loss
from model import ReferenceModel, create_model
from preferences import build_preference_dataset
from retrieval import Retriever

logger = logging.getLogger(__name__)


# ======================================================================
# 0. CHECKPOINT NAMING  (dataset x seed [x source])
# ======================================================================

def checkpoint_name(config: dict, tag: str,
                    pref_source: str | None = None) -> str:
    """Dataset- and seed-scoped checkpoint name, optionally source-scoped.

    SFT checkpoints are shared across arms (the SFT model is identical), so they
    are NOT source-scoped: ``pku_s42_sft_best``. Alignment checkpoints ARE
    source-scoped so A/B/C never collide: ``pku_s42_B_align_best``.
    """
    name = config["dataset"]["name"]
    seed = config["train_sft"]["seed"]
    if pref_source:
        return f"{name}_s{seed}_{pref_source.upper()}_{tag}"
    return f"{name}_s{seed}_{tag}"


def checkpoint_path(config: dict, tag: str,
                    pref_source: str | None = None) -> str:
    return os.path.join(config["paths"]["checkpoint_dir"],
                        checkpoint_name(config, tag, pref_source))


# ======================================================================
# 1. ENVIRONMENT
# ======================================================================

def set_seed(seed: int):
    """Seed stdlib/numpy always; seed torch/cudnn when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        import torch.backends.cudnn as cudnn
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


def setup_directories(config: dict):
    for key in ("output_dir", "checkpoint_dir", "log_dir", "cache_dir"):
        os.makedirs(config["paths"][key], exist_ok=True)


def setup_logging(log_dir: str, stage: str):
    log_file = os.path.join(
        log_dir, f"train_{stage}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    logger.info("Logging to %s", log_file)


# ======================================================================
# 2. OPTIMISER / SCHEDULER  (torch imported lazily)
# ======================================================================

def create_optimizer(model, config: dict, stage: str = "sft"):
    import torch
    skey = f"train_{stage}"
    lr = config[skey]["learning_rate"]
    wd = config[skey]["weight_decay"]
    params = model.get_trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    logger.info("Optimizer AdamW lr=%s wd=%s trainable=%s",
                lr, wd, f"{sum(p.numel() for p in params):,}")
    return opt


def create_scheduler(optimizer, config: dict, stage: str, num_training_steps: int):
    from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR, SequentialLR)
    skey = f"train_{stage}"
    warmup_steps = int(num_training_steps * config[skey]["warmup_ratio"])
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=max(warmup_steps, 1))
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(num_training_steps - warmup_steps, 1))
    sched = SequentialLR(optimizer, schedulers=[warmup, cosine],
                         milestones=[warmup_steps])
    logger.info("Scheduler cosine, %d warmup / %d total steps",
                warmup_steps, num_training_steps)
    return sched


# ======================================================================
# 3. RETRIEVER WIRING
#    Stage-2 candidate scoring (source B) needs the retrieved neighbours, which
#    the dataset materialises as Layout objects. We inject a Retriever whose
#    canvas_getter is the dataset's own load_layout, so neighbour ids resolve to
#    Layout objects (the rgpo contract) with no extra plumbing.
# ======================================================================

def attach_retriever(dataset: PosterLayoutDataset, config: dict) -> PosterLayoutDataset:
    """Give the dataset a Retriever so neighbour_layouts() is populated."""
    retriever = Retriever(config, canvas_getter=dataset.load_layout,
                          image_getter=dataset._load_canvas)
    dataset.retriever = retriever
    return dataset


# ======================================================================
# 4. STAGE 1 — SFT
# ======================================================================

def train_sft(model, config: dict) -> str:
    import torch
    scfg = config["train_sft"]
    epochs = scfg["epochs"]
    grad_accum = scfg["gradient_accumulation_steps"]
    max_grad_norm = scfg["max_grad_norm"]
    save_every, eval_every = scfg["save_every_n_epochs"], scfg["eval_every_n_epochs"]

    logger.info("=" * 60)
    logger.info("STAGE 1: SUPERVISED FINE-TUNING")
    logger.info("=" * 60)

    train_ds = attach_retriever(PosterLayoutDataset(config, split="train"), config)
    val_ds = attach_retriever(PosterLayoutDataset(config, split="val"), config)
    logger.info("Train %d / Val %d samples", len(train_ds), len(val_ds))

    # The SFT loop below iterates ONE sample at a time and takes an optimizer
    # step every grad_accum samples (gradient accumulation over single samples;
    # there is no minibatch of size bs in the loop). So the true number of
    # optimizer steps is len(train_ds)//grad_accum per epoch. Dividing by
    # (bs * grad_accum) — as an earlier version did — undercounts steps by a
    # factor of bs, which makes the cosine scheduler decay the LR to its minimum
    # bs-times too early and run most of training at ~0 LR. Effective batch size
    # here IS grad_accum; config's batch_size is not used to batch in this loop.
    total_steps = (len(train_ds) * epochs) // grad_accum
    optimizer = create_optimizer(model, config, "sft")
    scheduler = create_scheduler(optimizer, config, "sft", max(total_steps, 1))
    loss_fn = SFTLoss()

    best_val, best_ckpt, global_step = float("inf"), "", 0
    writer = _create_writer(config, "sft")

    for epoch in range(1, epochs + 1):
        model.model.train()
        loss_fn.reset()
        t0 = time.time()
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        optimizer.zero_grad()
        for step, idx in enumerate(indices):
            sample = train_ds[idx]
            out = model.forward_sft(sample)
            loss = loss_fn(out)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 100 == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info("E%d/%d step%d loss=%.4f lr=%.2e",
                                epoch, epochs, global_step, loss.item(), lr)
                    if writer:
                        writer.add_scalar("sft/loss", loss.item(), global_step)
                        writer.add_scalar("sft/lr", lr, global_step)
        logger.info("Epoch %d/%d done avg_loss=%.4f (%.1fs)",
                    epoch, epochs, loss_fn.average_loss, time.time() - t0)
        if epoch % eval_every == 0:
            val = validate_sft(model, val_ds, config)
            logger.info("Validation loss=%.4f", val)
            if writer:
                writer.add_scalar("sft/val_loss", val, epoch)
            if val < best_val:
                best_val = val
                best_ckpt = checkpoint_path(config, "sft_best")
                model.save_checkpoint(best_ckpt, optimizer, epoch,
                                      extra={"val_loss": val})
                logger.info("New best SFT (val=%.4f)", val)
        if epoch % save_every == 0:
            model.save_checkpoint(checkpoint_path(config, f"sft_epoch{epoch}"),
                                  optimizer, epoch)

    final = checkpoint_path(config, "sft_final")
    model.save_checkpoint(final, optimizer, epochs)
    if writer:
        writer.close()
    logger.info("SFT complete. Best: %s", best_ckpt or final)
    return best_ckpt or final


def validate_sft(model, val_ds, config: dict) -> float:
    import torch
    with torch.no_grad():
        model.model.eval()
        total, n = 0.0, 0
        cap = min(len(val_ds), 500)
        indices = list(range(len(val_ds)))
        random.shuffle(indices)
        for idx in indices[:cap]:
            out = model.forward_sft(val_ds[idx])
            total += out["loss"].item()
            n += 1
    return total / max(n, 1)


# ======================================================================
# 5. STAGE 2 — RETRIEVAL-GROUNDED ALIGNMENT  (source-parameterised)
# ======================================================================

def train_align(model, config: dict, sft_checkpoint: str,
                pref_source: str) -> str:
    """Build preference pairs with the chosen SOURCE, then DMPO fine-tune.

    Everything here is identical across arms EXCEPT ``pref_source``, which is
    handed to preferences.build_preference_dataset. That is the single point of
    divergence; the loss/optimiser/reference are common.
    """
    import torch
    scfg = config["train_dpo"]   # alignment hyperparameters live here
    epochs = scfg["epochs"]
    if epochs == 0:
        logger.info("Alignment skipped (epochs=0)")
        return sft_checkpoint

    grad_accum = scfg["gradient_accumulation_steps"]
    max_grad_norm = scfg["max_grad_norm"]

    logger.info("=" * 60)
    logger.info("STAGE 2: RETRIEVAL-GROUNDED ALIGNMENT (source=%s)", pref_source)
    logger.info("=" * 60)

    model.load_checkpoint(sft_checkpoint)
    logger.info("Loaded SFT checkpoint: %s", sft_checkpoint)
    ref_model = ReferenceModel(config, sft_checkpoint)

    # Build preference pairs — THE one source-dependent step.
    train_ds = attach_retriever(PosterLayoutDataset(config, split="train"), config)
    logger.info("Constructing preference pairs (source=%s)...", pref_source)
    pref_ds = build_preference_dataset(model, train_ds, config,
                                       pref_source=pref_source)
    logger.info("Built %d preference pairs", len(pref_ds))

    loss_fn = create_dpo_loss(config)
    total_steps = (len(pref_ds) * epochs) // grad_accum
    optimizer = create_optimizer(model, config, "dpo")
    scheduler = create_scheduler(optimizer, config, "dpo", max(total_steps, 1))

    best_margin, best_ckpt, global_step = -float("inf"), "", 0
    writer = _create_writer(config, f"align_{pref_source}")

    for epoch in range(1, epochs + 1):
        model.model.train()
        loss_fn.reset()
        t0 = time.time()
        indices = list(range(len(pref_ds)))
        random.shuffle(indices)
        optimizer.zero_grad()
        for step, idx in enumerate(indices):
            sample = pref_ds[idx]
            policy = model.forward_dpo(sample)
            ref = ref_model.compute_logps(sample)
            device = policy["chosen_logps"].device
            kw = dict(policy_chosen_logps=policy["chosen_logps"],
                      policy_rejected_logps=policy["rejected_logps"],
                      ref_chosen_logps=ref["ref_chosen_logps"],
                      ref_rejected_logps=ref["ref_rejected_logps"])
            if isinstance(loss_fn, DMPOLoss):
                margin = torch.tensor([sample["score_margin"]],
                                      dtype=torch.float32).to(device)
                out = loss_fn(score_margin=margin, **kw)
            else:
                out = loss_fn(**kw)
            (out["loss"] / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 50 == 0:
                    logger.info("E%d/%d step%d loss=%.4f margin=%.3f",
                                epoch, epochs, global_step, out["loss"].item(),
                                out["reward_margin"].item())
                    if writer:
                        for k in ("loss", "chosen_reward", "rejected_reward",
                                  "reward_margin", "dynamic_margin"):
                            if k in out:
                                writer.add_scalar(f"align/{k}",
                                                  out[k].item(), global_step)
        logger.info("Align E%d/%d done avg_loss=%.4f (%.1fs)",
                    epoch, epochs, loss_fn.average_loss, time.time() - t0)
        model.save_checkpoint(
            checkpoint_path(config, f"align_epoch{epoch}", pref_source),
            optimizer, epoch)
        avg_margin = ((loss_fn.running_chosen_reward
                       - loss_fn.running_rejected_reward)
                      / max(loss_fn.num_steps, 1))
        if avg_margin > best_margin:
            best_margin = avg_margin
            best_ckpt = checkpoint_path(config, "align_best", pref_source)
            model.save_checkpoint(best_ckpt, optimizer, epoch,
                                  extra={"reward_margin": avg_margin,
                                         "pref_source": pref_source})
            logger.info("New best alignment (margin=%.4f)", avg_margin)

    final = checkpoint_path(config, "align_final", pref_source)
    model.save_checkpoint(final, optimizer, epochs)
    if writer:
        writer.close()
    logger.info("Alignment complete (source=%s). Best: %s",
                pref_source, best_ckpt or final)
    return best_ckpt or final


# ======================================================================
# 6. FULL PIPELINE
# ======================================================================

def train_full(config: dict, pref_source: str) -> dict:
    logger.info("=" * 60)
    logger.info("FULL PIPELINE | dataset=%s | source=%s",
                config["dataset"]["name"], pref_source)
    logger.info("=" * 60)
    model = create_model(config, mode="train")
    sft_ckpt = train_sft(model, config)
    align_ckpt = train_align(model, config, sft_ckpt, pref_source)
    logger.info("DONE. sft=%s align=%s", sft_ckpt, align_ckpt)
    return {"sft_checkpoint": sft_ckpt, "align_checkpoint": align_ckpt,
            "pref_source": pref_source}


# ======================================================================
# 7. UTILITIES
# ======================================================================

def _create_writer(config: dict, stage: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        d = os.path.join(config["paths"]["log_dir"], stage)
        os.makedirs(d, exist_ok=True)
        return SummaryWriter(d)
    except ImportError:
        logger.info("TensorBoard unavailable; skipping scalar logging")
        return None


def load_and_apply_overrides(config: dict, overrides: list) -> dict:
    """Apply dot-notation key=value overrides with type inference.

    Supports creating the ``rgpo`` block keys too (e.g. rgpo.pref_source=B),
    which may not pre-exist in older config files.
    """
    config = copy.deepcopy(config)
    for ov in overrides:
        if "=" not in ov:
            logger.warning("Invalid override (no '='): %s", ov)
            continue
        key, value = ov.split("=", 1)
        keys = key.split(".")
        d = config
        ok = True
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}            # allow creating nested blocks (e.g. rgpo)
            d = d[k]
        if ok:
            fk = keys[-1]
            if fk in d and d[fk] is not None:
                t = type(d[fk])
                if t is bool:
                    d[fk] = value.lower() in ("true", "1", "yes")
                elif t is int:
                    d[fk] = int(value)
                elif t is float:
                    d[fk] = float(value)
                else:
                    d[fk] = value
            else:
                d[fk] = _infer_scalar(value)
            logger.info("Config override: %s = %r", key, d[fk])
    return config


def _infer_scalar(value: str):
    """Best-effort scalar typing for newly-created override keys."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def resolve_pref_source(args, config: dict) -> str:
    """CLI --pref-source wins; else config['rgpo']['pref_source']; else 'B'."""
    if getattr(args, "pref_source", None):
        return args.pref_source.upper()
    return str(config.get("rgpo", {}).get("pref_source", "B")).upper()


# ======================================================================
# 8. CLI
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-RAL / RGPO Training")
    p.add_argument("--stage", default="full",
                   choices=["sft", "align", "dpo", "full"],
                   help="'sft', 'align' (Stage 2), or 'full'. 'dpo' aliases 'align'.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", default=None, help="Override dataset (cgl/pku).")
    p.add_argument("--pref-source", dest="pref_source", default=None,
                   choices=["A", "B", "C", "a", "b", "c"],
                   help="Preference source for Stage 2: A learned / B RGPO / C hybrid.")
    p.add_argument("--checkpoint", default=None,
                   help="SFT checkpoint (for --stage align).")
    p.add_argument("--override", nargs="*", default=[],
                   help="Config overrides: key=value (dot-notation).")
    p.add_argument("--seed", type=int, default=None, help="Override seed.")
    return p


def main(argv: list | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    stage = "align" if args.stage == "dpo" else args.stage  # alias

    config = load_config(args.config)
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.seed is not None:
        config["train_sft"]["seed"] = args.seed
        config["train_dpo"]["seed"] = args.seed
    if args.override:
        config = load_and_apply_overrides(config, args.override)

    pref_source = resolve_pref_source(args, config)
    seed = config["train_sft"]["seed"]
    set_seed(seed)
    setup_directories(config)
    setup_logging(config["paths"]["log_dir"], stage)
    logger.info("config=%s seed=%d dataset=%s stage=%s pref_source=%s",
                args.config, seed, config["dataset"]["name"], stage, pref_source)

    if stage == "full":
        train_full(config, pref_source)
    elif stage == "sft":
        model = create_model(config, mode="train")
        train_sft(model, config)
    elif stage == "align":
        ckpt = args.checkpoint or checkpoint_path(config, "sft_best")
        if not os.path.exists(ckpt):
            parser.error("--checkpoint required for --stage align "
                         "(or run --stage full first)")
        model = create_model(config, mode="train")
        train_align(model, config, ckpt, pref_source)


if __name__ == "__main__":
    main()
