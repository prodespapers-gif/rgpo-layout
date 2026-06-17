"""
model.py — MLLM Backbone for Layout Generation (Qwen2.5-VL + LoRA)
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [3] of 10.

Role in the system
------------------
Wraps Qwen2.5-VL-7B-Instruct with LoRA for parameter-efficient fine-tuning, and
exposes the three calls the rest of the pipeline needs:
  * ``forward_sft(sample)``  -> cross-entropy loss on layout tokens  (Stage 1)
  * ``forward_dpo(sample)``  -> per-branch policy log-probabilities  (Stage 2)
  * ``generate_layout(sample)`` -> a parsed ``Layout``               (sampling)

The contract that matters for RGPO
----------------------------------
preferences.py (file [6]) samples N candidates per canvas by calling
``generate_layout`` and feeds them straight into rgpo.py's scorer, which
consumes ``Layout`` objects (file [1]'s type). So generation MUST return
``Layout`` — never a raw string or bare dict. The robust output parser here
produces exactly that, so no glue and no type-coercion sits between the model
and the preference signal.

Dependency direction (no cycles)
--------------------------------
data.py owns the types and serialisation. model.py imports FROM data.py
(``Layout``, ``json_to_layout``, ``CATEGORY_MAPS``); data.py never imports
model.py. losses.py consumes the log-probs this file returns but is otherwise
independent (and scorer-agnostic — it never sees a Layout or a reward).

What is preserved verbatim from the original (these were already correct)
-------------------------------------------------------------------------
  * Label masking by re-tokenising the prompt-only prefix in the SAME
    image-aware context as the full sequence (robust to the chat template's
    trailing <|im_end|> tokens and to vision-token boundary merging).
  * NOT wrapping the *policy* DPO forward in no_grad (the rejected term is half
    the preference gradient); grad-free behaviour is the *reference* model's job.
  * The nested-PeftModel guard in load_checkpoint (load adapter in-place when
    already a PeftModel; wrap only a bare base model).

References
----------
  - Qwen2.5-VL: Bai et al., 2025 (2502.13923)
  - LoRA: Hu et al., ICLR 2022
  - DPO: Rafailov et al., NeurIPS 2023
  - DMPO (consumes these log-probs): Lu et al. (Uni-Layout), ACM MM 2025
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional, Union

# data.py is the type owner.
from data import CATEGORY_MAPS, Layout, json_to_layout, load_config  # noqa: F401

logger = logging.getLogger(__name__)


# ======================================================================
# 1. CONVERSATION BUILDER  (Qwen2.5-VL multimodal message format)
# ======================================================================

def build_conversation(system: str, user: str,
                       image=None, assistant: Optional[str] = None) -> list:
    """Build a Qwen2.5-VL OpenAI-style multimodal message list.

    Shapes the system/user(+image)/assistant turns the processor expects.
    ``assistant`` is included only during training (SFT target / DPO branch).
    """
    messages = []
    if system:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system}]})
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": user})
    messages.append({"role": "user", "content": user_content})
    if assistant is not None:
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": assistant}]})
    return messages


# ======================================================================
# 2. OUTPUT PARSER  (raw MLLM text -> Layout)
#    Keeps the original's robust bracket-matching extraction (better than a
#    plain fence-strip for CoT-laden output), but emits a Layout so the whole
#    pipeline has ONE canonical type. Falls back through data.json_to_layout
#    for the actual record construction (lossless float / quantised decode).
# ======================================================================

class LayoutOutputParser:
    """Extracts the JSON layout array from messy model output and returns a
    ``Layout``. Tolerates markdown fences, CoT prefaces, and trailing prose."""

    def __init__(self, categories, strict: bool = False):
        self.categories = tuple(categories)
        self.strict = strict
        self._fence = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```")

    def _extract_json_array(self, text: str) -> Optional[str]:
        t = text.strip()
        m = self._fence.search(t)
        if m:
            t = m.group(1).strip()
        # Outermost balanced [...] block.
        depth, start = 0, None
        for i, ch in enumerate(t):
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    return t[start:i + 1]
        return t if t.startswith("[") else None

    def parse(self, text: str) -> Layout:
        """Return a validated ``Layout`` (empty if nothing parseable)."""
        arr = self._extract_json_array(text)
        if arr is None:
            logger.debug("No JSON array in model output: %.120s", text)
            return Layout(elements=[], categories=self.categories)
        # data.json_to_layout does the tolerant decode + dual-schema build and
        # already clamps/repairs coordinates; reuse it so behaviour is uniform.
        layout = json_to_layout(arr, self.categories)
        if self.strict:
            layout = Layout(
                elements=[e for e in layout.elements
                          if e.category in self.categories],
                categories=self.categories)
        return layout


# ======================================================================
# 3. MAIN MODEL  (Qwen2.5-VL + LoRA)
#    torch / transformers / peft are imported lazily inside methods so this
#    module (and its self-test) load without a GPU stack.
# ======================================================================

class LLMRALModel:
    """Qwen2.5-VL-7B-Instruct + LoRA layout generator.

    ``nn.Module`` subclassing is deferred to ``load_model`` time via a thin
    base so the file imports without torch. Most users go through
    ``create_model`` which loads the backbone immediately.
    """

    def __init__(self, config: dict, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.model_name = config["model"]["mllm_backbone"]
        self.categories = CATEGORY_MAPS[config["dataset"]["name"]]
        self.output_parser = LayoutOutputParser(self.categories, strict=False)
        self.model = None
        self.processor = None
        self.tokenizer = None

    # ---------- loading ----------
    def load_model(self):
        """Load the backbone, freeze the vision encoder, attach LoRA (train)."""
        import torch
        from transformers import AutoProcessor
        from peft import LoraConfig, TaskType, get_peft_model

        # Version-robust conditional-generation class resolution.
        ModelClass = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            logger.info("Using Qwen2_5_VLForConditionalGeneration")
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration as ModelClass
                logger.warning("Qwen2.5-VL class unavailable; using Qwen2-VL. "
                               "Upgrade transformers (>=4.49) for native 2.5 support.")
            except ImportError as exc:
                raise ImportError(
                    "Neither Qwen2_5_VLForConditionalGeneration nor "
                    "Qwen2VLForConditionalGeneration importable from transformers."
                ) from exc

        mcfg = self.config["model"]
        lcfg = mcfg["lora"]
        dtype = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(mcfg["torch_dtype"], torch.bfloat16)

        logger.info("Loading base model: %s", self.model_name)
        self.model = ModelClass.from_pretrained(
            self.model_name, torch_dtype=dtype,
            device_map=mcfg.get("device_map", "auto"),
            attn_implementation="flash_attention_2")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=mcfg.get("image_min_pixels", 784),
            max_pixels=mcfg.get("image_max_pixels", 401408))
        self.tokenizer = self.processor.tokenizer

        if mcfg.get("freeze_vision_encoder", True):
            for p in self.model.visual.parameters():
                p.requires_grad = False
            logger.info("Vision encoder frozen")

        if self.mode == "train":
            lora = LoraConfig(
                r=lcfg["rank"], lora_alpha=lcfg["alpha"],
                lora_dropout=lcfg["dropout"], target_modules=lcfg["target_modules"],
                bias=lcfg.get("bias", "none"), task_type=TaskType.CAUSAL_LM)
            self.model = get_peft_model(self.model, lora)
            self.model.print_trainable_parameters()
        logger.info("Model loaded")

    # ---------- tokenisation helpers ----------
    def _prepare_inputs(self, sample: dict, include_response: bool = True) -> dict:
        """Tokenise a full conversation (optionally including the response)."""
        messages = build_conversation(
            sample["system"], sample["user"], sample.get("image"),
            assistant=sample["assistant"] if include_response else None)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not include_response)
        if sample.get("image") is not None:
            from qwen_vl_utils import process_vision_info
            imgs, _ = process_vision_info(messages)
            return self.processor(text=[text], images=imgs or None,
                                  padding=True, return_tensors="pt")
        return self.tokenizer(text, return_tensors="pt", padding=True)

    def _prompt_prefix_ids(self, sample: dict):
        """Token ids of the prompt-only prefix, tokenised in image-aware context.

        Re-applies the chat template to the system/user/image turns with
        ``add_generation_prompt=True`` so it ends exactly at the assistant
        marker, counting vision-token expansion identically to the full
        sequence. (Preserved from the original; this is what makes label
        masking robust.)
        """
        messages = build_conversation(sample["system"], sample["user"],
                                      sample.get("image"), assistant=None)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if sample.get("image") is not None:
            from qwen_vl_utils import process_vision_info
            imgs, _ = process_vision_info(messages)
            out = self.processor(text=[text], images=imgs or None,
                                 padding=False, return_tensors="pt")
        else:
            out = self.tokenizer(text, return_tensors="pt", padding=False)
        return out["input_ids"][0]

    def _compute_labels(self, input_ids, sample: dict):
        """Labels with prompt tokens masked to -100; only the response is supervised."""
        labels = input_ids.clone()
        seq_len = labels.shape[-1]
        prompt_len = int(self._prompt_prefix_ids(sample).shape[-1])
        if prompt_len >= seq_len:
            logger.warning("Prompt prefix (%d) >= sequence (%d); masking all.",
                           prompt_len, seq_len)
            labels[:, :] = -100
            return labels
        labels[:, :prompt_len] = -100
        return labels

    # ---------- training forwards ----------
    def forward_sft(self, sample: dict) -> dict:
        """Stage-1 cross-entropy on layout tokens (prompt masked)."""
        import torch
        inputs = self._prepare_inputs(sample, include_response=True)
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}
        labels = self._compute_labels(inputs["input_ids"], sample).to(device)
        outputs = self.model(**inputs, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def forward_dpo(self, sample: dict) -> dict:
        """Stage-2 per-branch policy log-probs for chosen/rejected.

        NOTE: deliberately NOT under no_grad — the rejected term is half the
        preference gradient, so the policy must build the graph for both
        branches. The frozen reference uses ReferenceModel.compute_logps,
        which IS no_grad-wrapped.
        """
        import torch
        import torch.nn.functional as F
        device = next(self.model.parameters()).device
        logps = {}
        for key in ("chosen", "rejected"):
            s = {"image": sample["image"], "system": sample["system"],
                 "user": sample["user"], "assistant": sample[key]}
            inputs = self._prepare_inputs(s, include_response=True)
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}
            labels = self._compute_labels(inputs["input_ids"], s).to(device)
            outputs = self.model(**inputs)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            shift_logps = log_probs[:, :-1, :]
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100
            tok = torch.gather(shift_logps, 2,
                               shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            logps[f"{key}_logps"] = (tok * mask.float()).sum(dim=-1)
        return logps

    # ---------- generation ----------
    def generate(self, sample: dict, temperature: float = 0.7, top_p: float = 0.9,
                 max_new_tokens: Optional[int] = None,
                 num_return_sequences: int = 1) -> Union[str, list]:
        """Autoregressively decode layout JSON text (string, or list if N>1)."""
        import torch
        self.model.eval()
        max_tokens = max_new_tokens or self.config["model"]["max_new_tokens"]
        inputs = self._prepare_inputs(sample, include_response=False)
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-7),
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[:, input_len:]
        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return decoded[0] if num_return_sequences == 1 else decoded

    def generate_layout(self, sample: dict, **kwargs) -> Union[Layout, list]:
        """Generate and PARSE into a ``Layout`` (or list of Layouts if N>1).

        This is the call preferences.py uses to sample candidates; the return
        type is exactly what rgpo.py's scorer consumes — no glue between them.
        """
        text = self.generate(sample, **kwargs)
        if isinstance(text, list):
            return [self.output_parser.parse(t) for t in text]
        return self.output_parser.parse(text)

    # ---------- checkpoint io ----------
    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0,
                        extra: Optional[dict] = None):
        """Save LoRA adapters (+ optional optimiser/training state) only."""
        import torch
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        state = {"epoch": epoch}
        if extra:
            state.update(extra)
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(state, os.path.join(path, "training_state.pt"))
        logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)

    def load_checkpoint(self, path: str) -> dict:
        """Load LoRA adapters, handling the three call contexts correctly.

        Preserved nested-PeftModel guard: when self.model is ALREADY a
        PeftModel (train / pipeline), load the adapter in-place; only wrap a
        bare base model (inference). Wrapping a PeftModel in another PeftModel
        produces nested adapters and breaks trainable-parameter counting.
        """
        import torch
        from peft import PeftModel
        if self.model is None:
            self.load_model()
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(path, adapter_name="default",
                                    is_trainable=(self.mode == "train"))
            try:
                self.model.set_adapter("default")
            except (ValueError, KeyError):
                pass
        else:
            self.model = PeftModel.from_pretrained(
                self.model, path, is_trainable=(self.mode == "train"))
        state_path = os.path.join(path, "training_state.pt")
        state = torch.load(state_path, map_location="cpu") \
            if os.path.exists(state_path) else {}
        logger.info("Checkpoint loaded from %s (epoch %s)",
                    path, state.get("epoch", "unknown"))
        return state

    # ---------- introspection ----------
    def get_trainable_parameters(self) -> list:
        return [p for p in self.model.parameters() if p.requires_grad]

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable,
                "trainable_pct": 100.0 * trainable / total if total else 0.0}


# ======================================================================
# 4. REFERENCE MODEL  (frozen pi_ref for DPO/DMPO)
# ======================================================================

class ReferenceModel:
    """Frozen SFT checkpoint serving as pi_ref. log-prob calls are no_grad."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.model = LLMRALModel(config, mode="inference")
        self.model.load_model()
        self.model.load_checkpoint(checkpoint_path)
        for p in self.model.model.parameters():
            p.requires_grad = False
        self.model.model.eval()
        logger.info("Reference model loaded from %s", checkpoint_path)

    def compute_logps(self, sample: dict) -> dict:
        import torch
        with torch.no_grad():
            res = self.model.forward_dpo(sample)
        return {"ref_chosen_logps": res["chosen_logps"],
                "ref_rejected_logps": res["rejected_logps"]}


# ======================================================================
# 5. FACTORY
# ======================================================================

def create_model(config: dict, mode: str = "train") -> LLMRALModel:
    model = LLMRALModel(config, mode=mode)
    model.load_model()
    p = model.count_parameters()
    logger.info("Model created: %s total, %s trainable (%.2f%%)",
                f"{p['total']:,}", f"{p['trainable']:,}", p["trainable_pct"])
    return model


# ======================================================================
# 6. SELF-TEST  (run: python -m src.model)
#    No torch, no weights, no network. Verifies the parser contract — the one
#    piece preferences.py and rgpo.py depend on — and conversation shaping.
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("model.py SELF-TEST")
    print("=" * 64)

    cats = CATEGORY_MAPS["pku"]
    parser = LayoutOutputParser(cats)

    # (a) Clean JSON -> Layout with the right elements.
    clean = ('[{"category": "text", "center_x": 0.5, "center_y": 0.2, '
             '"width": 0.6, "height": 0.1}]')
    lay = parser.parse(clean)
    assert isinstance(lay, Layout) and len(lay) == 1
    assert lay.elements[0].category == "text"
    print("  [ok] clean JSON parses to a 1-element Layout")

    # (b) Markdown-fenced output.
    fenced = ('```json\n[{"category": "logo", "center_x": 0.1, "center_y": 0.1, '
              '"width": 0.15, "height": 0.08}]\n```')
    assert len(parser.parse(fenced)) == 1
    print("  [ok] fenced JSON parses")

    # (c) CoT preface before the array (the realistic case).
    cot = ('Based on the canvas, I will place text near the top.\n\n'
           '[{"category": "text", "center_x": 0.5, "center_y": 0.15, '
           '"width": 0.6, "height": 0.05}]')
    lay_cot = parser.parse(cot)
    assert len(lay_cot) == 1 and lay_cot.elements[0].category == "text"
    print("  [ok] CoT-prefaced output: array extracted past the prose")

    # (d) Trailing prose AFTER the array is ignored (balanced-bracket scan).
    trailing = ('[{"category": "underlay", "center_x": 0.5, "center_y": 0.7, '
                '"width": 0.5, "height": 0.15}] Hope this helps!')
    assert len(parser.parse(trailing)) == 1
    print("  [ok] trailing prose after array is ignored")

    # (e) Garbage -> empty Layout (never raises, never bare dict).
    junk = parser.parse("I could not produce a layout.")
    assert isinstance(junk, Layout) and junk.is_empty()
    print("  [ok] unparseable output -> empty Layout (no raise, correct type)")

    # (f) The parsed Layout is immediately consumable by the RGPO-critical
    #     order-invariant accessor (the model->preference handoff).
    grid = lay.occupancy_grid(resolution=16)
    assert grid.shape == (16, 16)
    print("  [ok] parsed Layout feeds occupancy_grid (model->rgpo handoff OK)")

    # (g) Conversation builder shapes system/user(+image)/assistant correctly.
    msgs = build_conversation("sys", "usr", image="<img>", assistant="ans")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user" and msgs[1]["content"][0]["type"] == "image"
    assert msgs[2]["role"] == "assistant"
    # Inference-time: no assistant turn, image still present.
    msgs2 = build_conversation("sys", "usr", image="<img>", assistant=None)
    assert len(msgs2) == 2 and msgs2[-1]["role"] == "user"
    print("  [ok] conversation builder shapes train/inference message lists")

    print("=" * 64)
    print("  ALL model.py SELF-TESTS PASSED")
    print("=" * 64)
