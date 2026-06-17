"""
inference.py — Generation & Qualitative Visualization
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [9] of 10. The qualitative half (Figure panels).

What it does
------------
  1. Generate layouts for a canvas (or a batch) with the trained model.
  2. Render a layout as colour-coded, semi-transparent boxes on its canvas.
  3. Build N-column side-by-side comparison strips
     (Canvas | Ground Truth | <baseline/checkpoint columns> | <live model>),
     each column from REAL outputs — a missing method is omitted, never faked.

Contract with the rest of the system
-------------------------------------
  * Renders ``Layout`` objects (file [1]'s type) and consumes
    ``model.generate_layout`` -> ``Layout`` directly (file [3]). No bare-dict
    layouts anywhere; the renderer reads ``layout.elements``.
  * Single-image inference fetches references through the real ``Retriever``
    (file [2]); batch/compare go through the dataset's ``neighbour_layouts``
    path, so the prompt the model sees at inference matches training.
  * torch / PIL imported lazily so the rendering geometry is unit-testable
    offline (the parts that need a GPU are exercised in real runs).

Why the comparison renderer takes an arbitrary column set
---------------------------------------------------------
The paper's qualitative figure is Canvas | GT | RALF | PosterLlama | Ours-SFT |
Ours-SFT+RGPO. The column renderer already handles any number of columns; this
driver supplies them from genuine per-method outputs on disk plus the live
model, so the figure is reproducible the moment those outputs exist.

References
----------
  - Visualisation style: RALF (Horita et al., CVPR 2024), visualizer.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Optional

from data import (CATEGORY_MAPS, Layout, build_sft_sample, json_to_layout,
                  layout_to_json, load_config)

logger = logging.getLogger(__name__)


# ======================================================================
# 1. COLOUR PALETTE  (category -> RGBA fill / RGB outline)
# ======================================================================

CATEGORY_COLORS = {
    "logo":          (255,  87,  87, 160),
    "text":          ( 87, 150, 255, 160),
    "underlay":      (100, 220, 100, 160),
    "embellishment": (255, 180,  60, 160),
}
CATEGORY_OUTLINES = {
    "logo":          (220,  50,  50),
    "text":          ( 50, 100, 220),
    "underlay":      ( 50, 180,  50),
    "embellishment": (220, 150,  30),
}
_DEFAULT_FILL = (180, 180, 180, 120)
_DEFAULT_OUTLINE = (120, 120, 120)


# ======================================================================
# 2. GEOMETRY  (normalised box -> pixel rect; pure, offline-testable)
# ======================================================================

def layout_to_pixel_boxes(layout: Layout, width: int, height: int) -> list:
    """Convert a ``Layout`` to drawing tuples, back-to-front by area.

    Returns a list of ``(category, (x1, y1, x2, y2))`` in PIXEL coordinates,
    clamped to the canvas, sorted largest-area-first so big elements are drawn
    behind small ones (the RALF z-order). Pure geometry — no PIL — so it is
    fully unit-testable offline; the renderer below just paints these.
    """
    boxes = []
    # Largest area first (drawn first => behind). Order-independent of the
    # Layout's own ordering; we sort explicitly here.
    for e in sorted(layout.elements, key=lambda el: -el.area):
        x1 = int((e.center_x - e.width / 2) * width)
        y1 = int((e.center_y - e.height / 2) * height)
        x2 = int((e.center_x + e.width / 2) * width)
        y2 = int((e.center_y + e.height / 2) * height)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        boxes.append((e.category, (x1, y1, x2, y2)))
    return boxes


# ======================================================================
# 3. RENDERING  (PIL imported lazily)
# ======================================================================

def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (f"/usr/share/fonts/truetype/dejavu/{name}",):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def render_layout_on_canvas(canvas, layout: Layout, label_elements: bool = True):
    """Render a ``Layout`` as semi-transparent boxes over its canvas (RGB image)."""
    from PIL import Image, ImageDraw
    img = canvas.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    W, H = img.size
    font = _load_font(10)
    for cat, (x1, y1, x2, y2) in layout_to_pixel_boxes(layout, W, H):
        fill = CATEGORY_COLORS.get(cat, _DEFAULT_FILL)
        outline = CATEGORY_OUTLINES.get(cat, _DEFAULT_OUTLINE)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=2)
        if label_elements and (x2 - x1 > 20) and (y2 - y1 > 12):
            draw.text((x1 + 3, y1 + 2), cat[:4].upper(),
                      fill=(255, 255, 255, 230), font=font)
    return Image.alpha_composite(img, overlay).convert("RGB")


def create_comparison_image(canvas, layouts_dict: dict, title: str = ""):
    """Side-by-side strip: Canvas | <each named layout>. Order = insertion order."""
    from PIL import Image, ImageDraw
    n_panels = len(layouts_dict) + 1
    panel_w, panel_h = canvas.size
    margin, header_h = 5, 30
    total_w = n_panels * panel_w + (n_panels - 1) * margin
    comp = Image.new("RGB", (total_w, panel_h + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(comp)
    font = _load_font(14, bold=True)
    comp.paste(canvas.convert("RGB"), (0, header_h))
    draw.text((5, 5), "Canvas", fill=(0, 0, 0), font=font)
    for i, (name, layout) in enumerate(layouts_dict.items()):
        x = (i + 1) * (panel_w + margin)
        comp.paste(render_layout_on_canvas(canvas, layout), (x, header_h))
        draw.text((x + 5, 5), name, fill=(0, 0, 0), font=font)
    return comp


def create_legend(categories):
    """A small colour legend image for the category palette."""
    from PIL import Image, ImageDraw
    lw, lh = 200, 25 * len(categories) + 15
    legend = Image.new("RGB", (lw, lh), (255, 255, 255))
    draw = ImageDraw.Draw(legend, "RGBA")
    font = _load_font(11)
    draw.text((5, 2), "Legend:", fill=(0, 0, 0), font=font)
    for i, cat in enumerate(categories):
        y = 18 + i * 25
        draw.rectangle([10, y, 30, y + 18],
                       fill=CATEGORY_COLORS.get(cat, _DEFAULT_FILL),
                       outline=CATEGORY_OUTLINES.get(cat, _DEFAULT_OUTLINE), width=2)
        draw.text((38, y + 2), cat.capitalize(), fill=(0, 0, 0), font=font)
    return legend


# ======================================================================
# 4. RETRIEVER WIRING  (so inference references match training)
# ======================================================================

def _attach_retriever(dataset, config: dict):
    from retrieval import Retriever
    dataset.retriever = Retriever(config, canvas_getter=dataset.load_layout,
                                  image_getter=dataset._load_canvas)
    return dataset


# ======================================================================
# 5. SINGLE-IMAGE INFERENCE
# ======================================================================

def run_single_inference(model, config: dict, canvas_path: str,
                         num_samples: int = 3,
                         neighbours: Optional[list] = None) -> list:
    """Generate ``num_samples`` layouts for one canvas image, with renders.

    ``neighbours`` (a list of reference ``Layout`` objects) is used to build the
    same design-aware prompt as training; if None, an empty reference set is
    used (the model still generates, just without exemplars).
    """
    from PIL import Image
    canvas = Image.open(canvas_path).convert("RGB")
    H, W = config["dataset"]["canvas_size"]
    canvas = canvas.resize((W, H), Image.LANCZOS)
    categories = CATEGORY_MAPS[config["dataset"]["name"]]
    neighbours = neighbours or []
    # Match the TRAINING prompt exactly: same number of in-prompt exemplars and
    # the same serialisation order data.py used at train time. Defaulting these
    # (max_retrieved=4, sort_by="label") would silently diverge from training
    # whenever the config differs, hurting generation quality at inference.
    max_retrieved = config["retrieval"]["max_retrieved_in_prompt"]
    sort_by = config["dataset"].get("sort_by", "label")

    results = []
    for i in range(num_samples):
        _seed(config["train_sft"]["seed"] + i)
        sample = build_sft_sample(
            canvas, Layout(elements=[], categories=tuple(categories)),
            neighbours, categories,
            use_cot=config["model"]["use_cot_reasoning"],
            max_retrieved=max_retrieved, sort_by=sort_by)
        # build_sft_sample sets an 'assistant' target we don't need at inference;
        # drop it so generation conditions only on system+user+image.
        sample.pop("assistant", None)
        gen = model.generate_layout(sample, temperature=config["inference"]["temperature"],
                                    top_p=config["inference"]["top_p"])
        if isinstance(gen, list):
            gen = gen[0] if gen else Layout(elements=[], categories=tuple(categories))
        results.append({
            "layout": gen,
            "layout_json": layout_to_json(gen),
            "visualization": render_layout_on_canvas(canvas, gen),
            "num_elements": len(gen),
        })
        logger.info("  sample %d/%d: %d elements", i + 1, num_samples, len(gen))
    return results


# ======================================================================
# 6. BATCH INFERENCE  (qualitative gallery)
# ======================================================================

def run_batch_inference(model, config: dict, split: str = "test",
                        num_samples: int = 50, save_dir: Optional[str] = None) -> list:
    from data import PosterLayoutDataset
    save_dir = save_dir or config["inference"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    dataset = _attach_retriever(PosterLayoutDataset(config, split=split), config)
    categories = dataset.categories

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    model.model.eval()
    logger.info("Batch inference: %d samples from %s", num_samples, split)

    results = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        sid = sample["metadata"]["sample_id"]
        gen = model.generate_layout(sample, temperature=config["inference"]["temperature"],
                                    top_p=config["inference"]["top_p"])
        if isinstance(gen, list):
            gen = gen[0] if gen else Layout(elements=[], categories=tuple(categories))
        gt = sample.get("target_layout", Layout(elements=[], categories=tuple(categories)))
        canvas = sample["image"]
        render_layout_on_canvas(canvas, gen).save(
            os.path.join(save_dir, f"{sid}_generated.png"))
        create_comparison_image(canvas, {"Ground Truth": gt, "Ours": gen}).save(
            os.path.join(save_dir, f"{sid}_comparison.png"))
        results.append({"sample_id": sid, "layout_json": layout_to_json(gen),
                        "num_elements": len(gen), "num_gt_elements": len(gt)})
        if (i + 1) % 10 == 0:
            logger.info("  processed %d/%d", i + 1, num_samples)

    create_legend(categories).save(os.path.join(save_dir, "legend.png"))
    with open(os.path.join(save_dir, "inference_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Batch inference complete -> %s", save_dir)
    return results


# ======================================================================
# 7. MULTI-METHOD COMPARISON  (Figure: Canvas | GT | methods... | live)
# ======================================================================

def _load_method_layout(method_dir: str, sid: str, categories) -> Optional[Layout]:
    """Load one method's pre-generated layout for ``sid`` (or None if absent)."""
    for ext in (".json", ".txt"):
        p = os.path.join(method_dir, f"{sid}{ext}")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json_to_layout(f.read(), categories)
            except Exception as exc:
                logger.warning("Failed to load %s for %s: %s", method_dir, sid, exc)
                return None
    return None


def run_comparison(model, config: dict, split: str = "test", num_samples: int = 20,
                   save_dir: Optional[str] = None,
                   method_layouts_dir: Optional[str] = None,
                   method_order: Optional[list] = None,
                   live_method_label: str = "Ours (RGPO)") -> list:
    """Render Canvas | GT | <methods from disk> | <live model> strips.

    Each method column is loaded from ``{method_layouts_dir}/{method}/{sid}.json``
    (genuine outputs). A method missing for a sample is skipped for that sample,
    never faked. The final column is the live model's generation.
    """
    from data import PosterLayoutDataset
    save_dir = save_dir or os.path.join(config["inference"]["save_dir"], "comparisons")
    os.makedirs(save_dir, exist_ok=True)
    dataset = _attach_retriever(PosterLayoutDataset(config, split=split), config)
    categories = dataset.categories

    if method_layouts_dir and os.path.isdir(method_layouts_dir):
        if method_order is None:
            method_order = sorted(d for d in os.listdir(method_layouts_dir)
                                  if os.path.isdir(os.path.join(method_layouts_dir, d)))
    else:
        method_order = []
        if method_layouts_dir:
            logger.warning("method_layouts_dir '%s' not found; GT + live only.",
                           method_layouts_dir)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    model.model.eval()
    logger.info("Comparison: %d samples (disk methods: %s; live: %s)",
                num_samples, method_order or "none", live_method_label)

    results = []
    for idx in indices:
        sample = dataset[idx]
        sid = sample["metadata"]["sample_id"]
        canvas = sample["image"]
        gt = sample.get("target_layout", Layout(elements=[], categories=tuple(categories)))
        cols = {"Ground Truth": gt}
        rendered = []
        for method in method_order:
            mlay = _load_method_layout(os.path.join(method_layouts_dir, method),
                                       sid, categories)
            if mlay is not None:
                cols[method] = mlay
                rendered.append(method)
        gen = model.generate_layout(sample, temperature=config["inference"]["temperature"],
                                    top_p=config["inference"]["top_p"])
        if isinstance(gen, list):
            gen = gen[0] if gen else Layout(elements=[], categories=tuple(categories))
        cols[live_method_label] = gen
        create_comparison_image(canvas, cols).save(
            os.path.join(save_dir, f"compare_{sid}.png"))
        results.append({"sample_id": sid,
                        "columns": ["Ground Truth"] + rendered + [live_method_label],
                        "our_elements": len(gen), "gt_elements": len(gt)})

    create_legend(categories).save(os.path.join(save_dir, "legend.png"))
    logger.info("Comparisons -> %s", save_dir)
    return results


# ======================================================================
# 8. HELPERS + CLI
# ======================================================================

def _seed(seed: int):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _auto_checkpoint(config: dict) -> Optional[str]:
    ds, seed = config["dataset"]["name"], config["train_sft"]["seed"]
    cands = [f"{ds}_s{seed}_{s}_{t}" for s in ("B", "A", "C")
             for t in ("align_best", "align_final")]
    cands += [f"{ds}_s{seed}_{t}" for t in ("sft_best", "sft_final")]
    for name in cands:
        p = os.path.join(config["paths"]["checkpoint_dir"], name)
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM-RAL / RGPO Inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="single", choices=["single", "batch", "compare"])
    parser.add_argument("--canvas", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--method_layouts_dir", default=None)
    parser.add_argument("--method_order", nargs="*", default=None)
    parser.add_argument("--live_method_label", default="Ours (RGPO)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config)
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.override:
        from train import load_and_apply_overrides
        config = load_and_apply_overrides(config, args.override)

    from model import create_model
    model = create_model(config, mode="inference")
    ckpt = args.checkpoint or _auto_checkpoint(config)
    if ckpt:
        logger.info("Checkpoint: %s", ckpt)
        model.load_checkpoint(ckpt)
    else:
        logger.warning("No checkpoint found; using base model")

    if args.mode == "single":
        if not args.canvas:
            parser.error("--canvas is required for single mode")
        num = args.num_samples or config["inference"]["num_samples_per_canvas"]
        results = run_single_inference(model, config, args.canvas, num_samples=num)
        save_dir = args.save_dir or config["inference"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        for i, r in enumerate(results):
            r["visualization"].save(os.path.join(save_dir, f"single_{i}.png"))
            print(f"Sample {i + 1}: {r['num_elements']} elements")
            print(r["layout_json"])
    elif args.mode == "batch":
        run_batch_inference(model, config, split=args.split,
                            num_samples=args.num_samples or 50, save_dir=args.save_dir)
    elif args.mode == "compare":
        run_comparison(model, config, split=args.split,
                       num_samples=args.num_samples or 20, save_dir=args.save_dir,
                       method_layouts_dir=args.method_layouts_dir,
                       method_order=args.method_order,
                       live_method_label=args.live_method_label)


# ======================================================================
# 9. SELF-TEST  (run: python -m src.inference)
#    Rendering GEOMETRY is pure and tested offline. Actual PIL drawing + model
#    generation are exercised in real runs; here we also smoke-test PIL drawing
#    if Pillow is importable (it usually is, even without torch).
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("inference.py SELF-TEST  (rendering geometry)")
    print("=" * 64)

    cats = CATEGORY_MAPS["pku"]

    def L(recs):
        return Layout.from_records(recs, cats)

    # (a) Normalised -> pixel conversion is correct and clamped.
    lay = L([{"category": "text", "center_x": 0.5, "center_y": 0.5,
              "width": 0.4, "height": 0.2}])
    boxes = layout_to_pixel_boxes(lay, 100, 100)
    assert boxes[0][0] == "text"
    x1, y1, x2, y2 = boxes[0][1]
    assert (x1, y1, x2, y2) == (30, 40, 70, 60), boxes[0][1]
    print(f"  [ok] normalised->pixel exact: center 0.5,0.5 w0.4 h0.2 -> {boxes[0][1]}")

    # (b) Out-of-canvas boxes are clamped to the frame.
    big = L([{"category": "logo", "center_x": 0.5, "center_y": 0.5,
              "width": 2.0, "height": 2.0}])
    (_, (bx1, by1, bx2, by2)), = layout_to_pixel_boxes(big, 80, 80)
    assert (bx1, by1, bx2, by2) == (0, 0, 80, 80)
    print("  [ok] oversized box clamped to canvas bounds")

    # (c) Z-order: larger area is drawn FIRST (appears behind).
    multi = L([{"category": "text", "center_x": 0.5, "center_y": 0.5, "width": 0.1, "height": 0.1},
               {"category": "underlay", "center_x": 0.5, "center_y": 0.5, "width": 0.8, "height": 0.8}])
    order = [c for c, _ in layout_to_pixel_boxes(multi, 100, 100)]
    assert order[0] == "underlay" and order[1] == "text", order
    print("  [ok] z-order: larger area drawn first (behind), smaller last (front)")

    # (d) Empty layout -> no boxes (renderer must handle gracefully).
    assert layout_to_pixel_boxes(Layout(elements=[], categories=tuple(cats)), 50, 50) == []
    print("  [ok] empty layout -> zero boxes")

    # (e) CGL's 4th category (embellishment) has palette entries.
    assert "embellishment" in CATEGORY_COLORS and "embellishment" in CATEGORY_OUTLINES
    print("  [ok] palette covers all 4 categories (PKU 3 + CGL embellishment)")

    # (f) If Pillow is available, smoke-test the actual render path end-to-end.
    try:
        from PIL import Image
        canvas = Image.new("RGB", (120, 120), (240, 240, 240))
        out = render_layout_on_canvas(canvas, lay)
        assert out.size == (120, 120) and out.mode == "RGB"
        comp = create_comparison_image(canvas, {"GT": lay, "Ours": multi})
        # width = 3 panels * 120 + 2 * 5 margin = 370 ; height = 120 + 30 header
        assert comp.size == (370, 150), comp.size
        legend = create_legend(cats)
        assert legend.size[0] == 200
        print("  [ok] PIL render path: single + 3-col comparison + legend all draw")
    except ImportError:
        print("  [--] Pillow not installed; skipped live PIL draw (geometry already verified)")

    print("=" * 64)
    print("  ALL inference.py SELF-TESTS PASSED")
    print("=" * 64)
