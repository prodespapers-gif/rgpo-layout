"""
retrieval.py — Retrieval Augmentation
================================================================================

LLM-Guided Retrieval-Augmented Layout Generation for Content-Aware Poster Design
File [2] of 10. The other half of the data contract.

What lives here
---------------
  1. ``CanvasFeatureExtractor`` — perceptual canvas embeddings
     (DreamSim / CLIP / DINOv2 / coarse-saliency), torch imported lazily so the
     module and its self-test load without a GPU.
  2. ``FAISSIndex`` — exact (flat) or approximate (IVF) nearest-neighbour index,
     with IVF list-size chosen from the actual vector count and an automatic
     fall-back to exact search when N is small.
  3. ``mmr_rerank`` — Maximal Marginal Relevance diversification.
  4. ``PrecomputedRetrievalIndex`` — loads RALF's precomputed DreamSim indices
     (one YAML per split: ``id -> [neighbour ids]``).
  5. ``Retriever`` — the high-level object the rest of the pipeline holds.
     Crucially it returns neighbours as ``Layout`` objects (file [1]'s type),
     which is *exactly* what rgpo.py consumes. No bare dicts cross this boundary.

Why this is the second file
---------------------------
data.py (file [1]) already calls two methods on an injected ``retriever``:
``get_neighbor_ids(sid, k=..., split=...)`` and, indirectly, the neighbour
``Layout`` list. Writing retrieval.py now closes that loop end-to-end and lets
us verify the neighbour-``Layout`` return path before we reach the contribution
file (rgpo.py).

Dependency direction (no cycles)
--------------------------------
data.py owns the types. retrieval.py imports FROM data.py
(``from data import Layout, load_config``); data.py never imports retrieval.py —
it merely accepts an optional ``retriever`` and calls the agreed methods. The
two are wired together by the caller (train.py / evaluate.py).

Two distinct getters (this is the corrected design)
---------------------------------------------------
A neighbour id must resolve to a *Layout* for the rgpo.py contract, but the
FAISS-fallback path needs a *PIL image* to embed. These are different return
types and must not be conflated into one callable:
  * ``canvas_getter``: ``sid -> Layout``     (used by ``neighbour_layouts``)
  * ``image_getter`` : ``sid -> PIL.Image``  (used only by the FAISS fallback)

Unannotated canvases
--------------------
The ``with_no_annotation`` split has canvases but no layouts, so it can never
*serve* references. When the query lives in that split we still retrieve from
the annotated **train** pool (the reference corpus), matching RALF.

References
----------
  - DreamSim: Fu et al., NeurIPS 2023
  - FAISS: Johnson et al., IEEE Trans. Big Data 2021
  - Retrieval for layout: Horita et al. (RALF), CVPR 2024, Sec. 3.3
  - MMR: Carbonell & Goldstein, SIGIR 1998
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Callable, Optional, Sequence

import numpy as np
import yaml
from numpy import linalg as LA

# data.py is the type owner; importing it here (and never the reverse) keeps the
# dependency graph acyclic.
from data import Layout, load_config  # noqa: F401  (load_config re-exported for CLI)

logger = logging.getLogger(__name__)

# The reference pool for queries that cannot serve their own references.
_UNANNOTATED_SPLIT = "with_no_annotation"
_REFERENCE_SPLIT_FOR_UNANNOTATED = "train"


# ======================================================================
# 1. FEATURE EXTRACTORS  (torch imported lazily)
# ======================================================================

class CanvasFeatureExtractor:
    """Extracts a unit-norm perceptual embedding from a canvas image.

    Backbones (selectable for ablation):
      * ``dreamsim`` — 1792-d concatenated CLIP+OpenCLIP+DINO (RALF default)
      * ``clip``     — 768-d ViT-B/16 pre-projection pooled features
      * ``dinov2``   — 768-d self-supervised
      * ``saliency`` — 256-d flattened 16x16 coarse saliency (cheap baseline)

    torch / torchvision / model weights are imported the first time a model is
    actually needed, so constructing a ``saliency`` extractor — or merely
    importing this module — never requires a GPU stack.
    """

    SUPPORTED_BACKBONES = ("dreamsim", "clip", "dinov2", "saliency")

    def __init__(self, backbone: str = "dreamsim", device: Optional[str] = None):
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Choose from {self.SUPPORTED_BACKBONES}")
        self.backbone_name = backbone
        self._device = device
        self.model = None
        self.transform = None
        self._embedding_dim: Optional[int] = None
        if backbone == "saliency":
            self._embedding_dim = 256  # 16x16, no model needed

    # ---- device / model are resolved lazily ----
    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def _ensure_model(self):
        if self.model is not None or self.backbone_name == "saliency":
            return
        if self.backbone_name == "dreamsim":
            self._load_dreamsim()
        elif self.backbone_name == "clip":
            self._load_clip()
        elif self.backbone_name == "dinov2":
            self._load_dinov2()

    def _load_dreamsim(self):
        try:
            import torch
            from dreamsim import dreamsim
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError("DreamSim not installed. Run: pip install dreamsim") from exc
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llm_ral", "dreamsim")
        os.makedirs(cache_dir, exist_ok=True)
        model, _ = dreamsim(pretrained=True, cache_dir=cache_dir)
        for p in model.parameters():
            p.requires_grad = False
        model.eval().to(self.device)
        self.model = model.embed
        self.transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
        ])
        self._embedding_dim = 1792
        logger.info("Loaded DreamSim (dim=%d)", self._embedding_dim)

    def _load_clip(self):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm not installed. Run: pip install timm") from exc
        self.model = timm.create_model("vit_base_patch16_clip_224.openai",
                                       pretrained=True, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().to(self.device)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)
        # num_classes=0 -> Identity head -> 768-d pre-projection pooled output.
        # The dim MUST equal the real output (it is handed to FAISSIndex).
        self._embedding_dim = 768
        logger.info("Loaded CLIP ViT-B/16 pre-projection features (dim=768)")

    def _load_dinov2(self):
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError("torchvision not installed") from exc
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._embedding_dim = 768
        logger.info("Loaded DINOv2 ViT-B/14 (dim=768)")

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._ensure_model()
        return self._embedding_dim

    def extract(self, image) -> np.ndarray:
        """Return a unit-norm (embedding_dim,) feature for one PIL/tensor image."""
        if self.backbone_name == "saliency":
            return self._extract_saliency(image)
        import torch  # local: only needed for the neural backbones
        self._ensure_model()
        from PIL import Image as _Image
        if isinstance(image, _Image.Image):
            from torchvision import transforms as T
            image = T.ToTensor()(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 4:        # drop alpha / saliency channel
            image = image[:, :3]
        with torch.no_grad():
            image = self.transform(image).to(self.device)
            feat = self.model(image).cpu().numpy()[0]
        return _unit(feat)

    def extract_batch(self, images: Sequence, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            out.extend(self.extract(im) for im in images[i:i + batch_size])
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _extract_saliency(image) -> np.ndarray:
        """16x16 flattened coarse saliency (256-d), unit-normalised."""
        import torch
        from PIL import Image as _Image
        if isinstance(image, _Image.Image):
            from torchvision import transforms as T
            image = T.ToTensor()(image.convert("L"))
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 4:
            sal = image[:, 3:4]
        elif image.shape[1] == 1:
            sal = image
        else:
            sal = image[:, :1]
        coarse = torch.nn.functional.interpolate(
            sal.float(), size=(16, 16), mode="bilinear", align_corners=False)
        return _unit(coarse.flatten().numpy())


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    n = LA.norm(vec)
    return vec / n if n > 0 else vec


# ======================================================================
# 2. FAISS INDEX  (IVF sized to N, exact-search fallback)
# ======================================================================

class FAISSIndex:
    """Inner-product (cosine, on unit vectors) NN index over sample ids."""

    _MIN_POINTS_PER_CENTROID = 39  # FAISS rule of thumb

    def __init__(self, embedding_dim: int, index_type: str = "faiss_ivf",
                 nlist: int = 100):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS not installed. Run: pip install faiss-gpu "
                              "(or faiss-cpu)") from exc
        self.faiss = faiss
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.requested_nlist = nlist
        self.sample_ids: list[str] = []
        self._needs_training = False
        if index_type == "faiss_flat":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "faiss_ivf":
            self.index = None           # built lazily in add(), once N is known
            self._needs_training = True
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        logger.info("Created FAISS index: type=%s, dim=%d", index_type, embedding_dim)

    def _build_ivf(self, n_vectors: int):
        faiss = self.faiss
        if n_vectors < 2 * self._MIN_POINTS_PER_CENTROID:
            logger.info("Only %d vectors; using exact flat index (too few for IVF).",
                        n_vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index_type = "faiss_flat"
            self._needs_training = False
            return
        nlist = int(round(math.sqrt(n_vectors)))
        nlist = max(1, min(nlist, self.requested_nlist, n_vectors))
        nlist = min(nlist, max(1, n_vectors // self._MIN_POINTS_PER_CENTROID))
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        logger.info("Built IVF index nlist=%d for %d vectors (requested %d).",
                    nlist, n_vectors, self.requested_nlist)

    def add(self, embeddings: np.ndarray, sample_ids: Sequence[str]):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._needs_training:
            if self.index is None:
                self._build_ivf(len(embeddings))
            if not self.index.is_trained:
                self.index.train(embeddings)
            self._needs_training = False
        self.index.add(embeddings)
        self.sample_ids.extend(str(s) for s in sample_ids)
        logger.info("Added %d vectors (total %d).", len(sample_ids), self.index.ntotal)

    def get_embeddings(self, ids: Sequence[str]) -> Optional[np.ndarray]:
        """Reconstruct stored vectors for the given ids (for MMR).

        Returns None if reconstruction is unsupported by the backing index.
        """
        try:
            self.index.make_direct_map()
            pos = {sid: i for i, sid in enumerate(self.sample_ids)}
            rows = [self.index.reconstruct(pos[s]) for s in ids if s in pos]
            return np.asarray(rows, dtype=np.float32) if rows else None
        except Exception:  # not all index types support reconstruction
            return None

    def search(self, query: np.ndarray, k: int = 16,
               exclude_ids: Optional[set[str]] = None) -> list[str]:
        if self.index is None:
            raise RuntimeError("FAISS index is empty; call add() before search().")
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        search_k = min(self.index.ntotal,
                       k + (len(exclude_ids) if exclude_ids else 0) + 1)
        _, indices = self.index.search(query, search_k)
        results: list[str] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.sample_ids):
                continue
            sid = self.sample_ids[idx]
            if exclude_ids and sid in exclude_ids:
                continue
            results.append(sid)
            if len(results) >= k:
                break
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.faiss.write_index(self.index, path)
        with open(path + ".ids.json", "w") as f:
            json.dump(self.sample_ids, f)
        logger.info("Saved FAISS index to %s", path)

    def load(self, path: str):
        self.index = self.faiss.read_index(path)
        with open(path + ".ids.json", "r") as f:
            self.sample_ids = json.load(f)
        self._needs_training = False
        logger.info("Loaded FAISS index from %s (%d vectors).", path, self.index.ntotal)


# ======================================================================
# 3. MMR RE-RANKING
# ======================================================================

def mmr_rerank(query_embedding: np.ndarray, candidate_embeddings: np.ndarray,
               candidate_ids: Sequence[str], top_k: int,
               lambda_param: float = 0.7) -> list[str]:
    """Re-rank candidates to balance query-relevance and mutual diversity.

    Score(d) = lambda * sim(d, q) - (1 - lambda) * max_{s in S} sim(d, s)
    lambda=1 -> pure relevance; lambda=0 -> pure diversity.
    """
    ids = list(candidate_ids)
    if len(ids) <= top_k:
        return ids
    sim_q = candidate_embeddings @ query_embedding          # (N,)
    sim_cc = candidate_embeddings @ candidate_embeddings.T    # (N, N)
    selected = [int(np.argmax(sim_q))]
    remaining = set(range(len(ids))) - set(selected)
    while len(selected) < min(top_k, len(ids)):
        best_idx, best_score = -1, -float("inf")
        for idx in remaining:
            redundancy = max(sim_cc[idx, s] for s in selected)
            score = lambda_param * sim_q[idx] - (1 - lambda_param) * redundancy
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
    return [ids[i] for i in selected]


# ======================================================================
# 4. PRECOMPUTED INDEX LOADER  (RALF YAML: id -> [neighbour ids])
# ======================================================================

class PrecomputedRetrievalIndex:
    """Loads and queries one split of RALF's precomputed DreamSim index."""

    def __init__(self, retrieval_dir: str, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.index: dict[str, list[str]] = {}
        yaml_path = os.path.join(retrieval_dir, dataset_name, f"{split}.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            self.index = {str(k): [str(v) for v in vals] for k, vals in raw.items()}
            logger.info("Loaded precomputed retrieval index: %s (%d queries).",
                        yaml_path, len(self.index))
        else:
            logger.warning("Precomputed index not found: %s. Retrieval will be "
                           "empty for this split. Prepare it via RALF.", yaml_path)

    def query(self, sample_id: str, k: int = 16) -> list[str]:
        return self.index.get(str(sample_id), [])[:k]

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self.index


# ======================================================================
# 5. RETRIEVER  (the object the pipeline holds; returns Layout objects)
# ======================================================================

class Retriever:
    """High-level retrieval the rest of the pipeline depends on.

    Two responsibilities:
      * ``get_neighbor_ids`` — resolve a query id to neighbour ids
        (precomputed-first, FAISS fallback). Signature matches the call in
        data.py: ``get_neighbor_ids(sample_id, k=None, split=...)``.
      * ``neighbour_layouts`` — turn those ids into ``Layout`` objects, which is
        the exact type rgpo.py consumes. Empty layouts are dropped so the
        agreement score never receives a degenerate reference.

    Two getters (different return types, deliberately separate):
      * ``canvas_getter``: ``sid -> Layout``    (neighbour_layouts uses this;
        typically ``dataset.load_layout`` from file [1])
      * ``image_getter`` : ``sid -> PIL.Image`` (FAISS fallback embeds this)
    """

    def __init__(self, config: dict,
                 canvas_getter: Optional[Callable[[str], Layout]] = None,
                 image_getter: Optional[Callable[[str], object]] = None):
        self.config = config
        self.dataset_name = config["dataset"]["name"]
        self.K = config["retrieval"]["num_neighbors"]
        self.backbone_name = config["retrieval"]["similarity_model"]
        self.mmr_enabled = config["retrieval"].get("use_mmr", False)
        self.mmr_lambda = config["retrieval"].get("mmr_lambda", 0.7)
        # Honor exclude_self: never let a query retrieve ITSELF as a neighbour.
        # RALF's precomputed indices are usually built self-excluded, but if a
        # query id appears in its own neighbour list the RGPO agreement score
        # would be trivially inflated (the candidate compared against this very
        # canvas's ground truth), faking source B's signal. Filtering here is
        # cheap and a no-op when the index already excludes self.
        self.exclude_self = config["retrieval"].get("exclude_self", True)
        self._canvas_getter = canvas_getter   # id -> Layout
        self._image_getter = image_getter     # id -> PIL image (FAISS fallback)

        self.precomputed: dict[str, PrecomputedRetrievalIndex] = {}
        if config["retrieval"].get("use_precomputed_index", True):
            rdir = config["paths"]["ralf_retrieval_dir"]
            for split in ("train", "val", "test", _UNANNOTATED_SPLIT):
                self.precomputed[split] = PrecomputedRetrievalIndex(
                    rdir, self.dataset_name, split)

        self._extractor: Optional[CanvasFeatureExtractor] = None
        self._faiss: Optional[FAISSIndex] = None

    # ---- lazy feature extractor (only if we must fall back to FAISS) ----
    @property
    def feature_extractor(self) -> CanvasFeatureExtractor:
        if self._extractor is None:
            self._extractor = CanvasFeatureExtractor(backbone=self.backbone_name)
        return self._extractor

    def attach_faiss(self, index: FAISSIndex):
        """Provide a prebuilt FAISS index for splits without a precomputed one."""
        self._faiss = index

    @staticmethod
    def _reference_split(split: str) -> str:
        """Unannotated queries draw references from the annotated train pool."""
        return _REFERENCE_SPLIT_FOR_UNANNOTATED if split == _UNANNOTATED_SPLIT else split

    # ---- id resolution (matches data.py's call signature) ----
    def get_neighbor_ids(self, sample_id: str, k: Optional[int] = None,
                         split: str = "train") -> list[str]:
        """Top-k neighbour ids for a query. Precomputed-first, FAISS fallback."""
        k = k or self.K
        sample_id = str(sample_id)
        lookup_split = self._reference_split(split)

        idx = self.precomputed.get(lookup_split)
        if idx is not None and sample_id in idx:
            # Over-fetch by one when self-excluding so we still return k after
            # dropping the query if it appears in its own neighbour list.
            base = max(k * 3, k) if self.mmr_enabled else k
            fetch = base + (1 if self.exclude_self else 0)
            ids = idx.query(sample_id, k=fetch)
            if self.exclude_self:
                ids = [i for i in ids if i != sample_id]
            if self.mmr_enabled:
                ids = self._maybe_mmr(sample_id, ids, k)
            return ids[:k]

        if self._faiss is not None and self._image_getter is not None:
            feat = self.feature_extractor.extract(self._image_getter(sample_id))
            # FAISS path already excludes the query via exclude_ids.
            excl = {sample_id} if self.exclude_self else None
            return self._faiss.search(feat, k=k, exclude_ids=excl)

        logger.warning("No retrieval available for %s in split %s (returning empty).",
                       sample_id, split)
        return []

    def _maybe_mmr(self, query_id: str, cand_ids: list[str], k: int) -> list[str]:
        """Diversify candidate ids with MMR when embeddings are reconstructable."""
        if self._faiss is None:
            return cand_ids[:k]
        q = self._faiss.get_embeddings([query_id])
        c = self._faiss.get_embeddings(cand_ids)
        if q is None or c is None or len(c) != len(cand_ids):
            return cand_ids[:k]
        return mmr_rerank(q[0], c, cand_ids, top_k=k, lambda_param=self.mmr_lambda)

    # ---- the rgpo.py contract: neighbours AS Layout objects ----
    def neighbour_layouts(self, sample_id: str, k: Optional[int] = None,
                          split: str = "train") -> list[Layout]:
        """Return the retrieved neighbours as non-empty ``Layout`` objects."""
        if self._canvas_getter is None:
            raise RuntimeError(
                "Retriever has no canvas_getter; inject dataset.load_layout so "
                "neighbour ids can be resolved to Layout objects.")
        k = k or self.K
        out: list[Layout] = []
        for nid in self.get_neighbor_ids(sample_id, k=k, split=split):
            lay = self._canvas_getter(nid)
            if isinstance(lay, Layout) and not lay.is_empty():
                out.append(lay)
        return out

    # ---- building a fresh index (ablations / new canvases) ----
    def build_index(self, images: Sequence, sample_ids: Sequence[str],
                    save_path: Optional[str] = None) -> FAISSIndex:
        logger.info("Building FAISS index with %s for %d images...",
                    self.backbone_name, len(images))
        embs = self.feature_extractor.extract_batch(images)
        index = FAISSIndex(embedding_dim=self.feature_extractor.embedding_dim,
                           index_type=self.config["retrieval"].get("index_type",
                                                                    "faiss_ivf"))
        index.add(embs, sample_ids)
        if save_path:
            index.save(save_path)
        self._faiss = index
        return index


# ======================================================================
# 6. SELF-TEST  (no torch, no FAISS, no network required)
#    Run: python -m src.retrieval
#    A tiny in-memory NN index doubles for FAISS so we can verify ordering,
#    self-exclusion, MMR, id-resolution, and the neighbour_layouts contract.
# ======================================================================

class _NumpyIndex:
    """Minimal exact inner-product index implementing the FAISSIndex surface
    that Retriever touches. Test double only — never used in production."""

    def __init__(self):
        self._vecs: list[np.ndarray] = []
        self.sample_ids: list[str] = []

    def add(self, embeddings, sample_ids):
        for v, s in zip(np.asarray(embeddings, dtype=np.float32), sample_ids):
            self._vecs.append(v)
            self.sample_ids.append(str(s))

    def get_embeddings(self, ids):
        pos = {s: i for i, s in enumerate(self.sample_ids)}
        rows = [self._vecs[pos[s]] for s in ids if s in pos]
        return np.asarray(rows, dtype=np.float32) if rows else None

    def search(self, query, k=16, exclude_ids=None):
        mat = np.asarray(self._vecs, dtype=np.float32)
        sims = mat @ np.asarray(query, dtype=np.float32).reshape(-1)
        order = np.argsort(-sims)
        out = []
        for idx in order:
            sid = self.sample_ids[idx]
            if exclude_ids and sid in exclude_ids:
                continue
            out.append(sid)
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 64)
    print("retrieval.py SELF-TEST")
    print("=" * 64)

    rng = np.random.default_rng(0)

    # (a) Unit normalisation.
    v = _unit(rng.normal(size=64).astype(np.float32))
    assert abs(LA.norm(v) - 1.0) < 1e-5
    print("  [ok] feature unit-normalisation")

    # (b) NN ordering + self-exclusion via the numpy index double.
    dim = 8
    base = _unit(rng.normal(size=dim).astype(np.float32))
    vecs, ids = [], []
    for i in range(10):
        noise = 0.01 * i
        vecs.append(_unit(base + noise * rng.normal(size=dim)))
        ids.append(f"s{i}")
    idx = _NumpyIndex()
    idx.add(np.asarray(vecs), ids)
    nn = idx.search(vecs[0], k=3, exclude_ids={"s0"})
    assert "s0" not in nn and nn[0] == "s1", f"unexpected NN order: {nn}"
    print("  [ok] nearest-neighbour ordering + self-exclusion")

    # (c) MMR returns k unique ids drawn from the candidate set.
    cand = ids[1:]
    cmat = np.asarray(vecs[1:], dtype=np.float32)
    re = mmr_rerank(vecs[0], cmat, cand, top_k=4, lambda_param=0.5)
    assert len(re) == 4 and len(set(re)) == 4 and set(re) <= set(cand)
    print("  [ok] MMR re-ranking yields k unique in-set ids")

    # (d) Retriever id-resolution against a precomputed index (no torch/FAISS).
    cfg = {
        "dataset": {"name": "pku"},
        "retrieval": {"num_neighbors": 3, "similarity_model": "saliency",
                      "use_precomputed_index": False, "use_mmr": False},
        "paths": {"ralf_retrieval_dir": "/nonexistent"},
    }
    layouts = {
        "n1": Layout.from_records(
            [{"category": "text", "center_x": 0.5, "center_y": 0.5,
              "width": 0.3, "height": 0.1}], ["logo", "text", "underlay"]),
        "n2": Layout.from_records(
            [{"category": "logo", "center_x": 0.2, "center_y": 0.1,
              "width": 0.1, "height": 0.1}], ["logo", "text", "underlay"]),
        "n3": Layout(elements=[], categories=("logo", "text", "underlay")),  # empty
    }
    r = Retriever(cfg, canvas_getter=lambda sid: layouts.get(
        sid, Layout(elements=[], categories=())))
    # Inject a precomputed mapping by hand (bypass missing YAML).
    pre = PrecomputedRetrievalIndex.__new__(PrecomputedRetrievalIndex)
    pre.dataset_name, pre.split = "pku", "train"
    pre.index = {"q": ["n1", "n2", "n3"]}
    r.precomputed["train"] = pre
    got = r.get_neighbor_ids("q", split="train")
    assert got == ["n1", "n2", "n3"], f"id resolution wrong: {got}"
    print("  [ok] Retriever.get_neighbor_ids resolves via precomputed index")

    # (e) neighbour_layouts returns Layout objects and DROPS the empty one.
    neigh = r.neighbour_layouts("q", split="train")
    assert all(isinstance(x, Layout) for x in neigh)
    assert len(neigh) == 2, f"empty layout not dropped: {len(neigh)}"
    print("  [ok] neighbour_layouts returns Layout objects, drops empties")

    # (f) Unannotated split remaps to the train reference pool.
    assert Retriever._reference_split("with_no_annotation") == "train"
    assert Retriever._reference_split("test") == "test"
    print("  [ok] unannotated split remaps to train reference pool")

    print("=" * 64)
    print("  ALL retrieval.py SELF-TESTS PASSED")
    print("=" * 64)
