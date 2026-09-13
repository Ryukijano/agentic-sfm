"""Learned retrieval for Phase 2 scene-level episodes.

Uses DINOv2 (or a fallback to image histograms) to compute image embeddings
and retrieve the most similar image pairs for matching.

The retrieval model is used by the agent's ``retrieve`` tool during rollouts.
For the S-GRPO oracle, we use GT overlap_matrix instead (the oracle knows
the answer). For real rollouts, the agent must use the learned retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded retrieval model
_retrieval_model = None
_retrieval_processor = None


def _get_retrieval_model():
    """Lazy-load DINOv2 for image retrieval."""
    global _retrieval_model, _retrieval_processor
    if _retrieval_model is not None:
        return _retrieval_model, _retrieval_processor

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-small"  # 22M params, fast
        _retrieval_processor = AutoImageProcessor.from_pretrained(model_name)
        _retrieval_model = AutoModel.from_pretrained(model_name)
        _retrieval_model = _retrieval_model.eval()
        if torch.cuda.is_available():
            _retrieval_model = _retrieval_model.cuda()
        logger.info(f"Loaded DINOv2 retrieval model: {model_name}")
    except Exception as e:
        logger.warning(f"Failed to load DINOv2 ({e}), falling back to histogram retrieval")
        _retrieval_model = "histogram"
        _retrieval_processor = None

    return _retrieval_model, _retrieval_processor


def compute_image_embedding(image_path: str | Path) -> np.ndarray:
    """Compute a single image embedding using DINOv2.

    Returns a 1-D feature vector. Falls back to a color histogram if
    DINOv2 is not available.
    """
    model, processor = _get_retrieval_model()

    try:
        from PIL import Image
        img = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return np.zeros(384, dtype=np.float32)

    if model == "histogram":
        return _histogram_embedding(img)

    try:
        import torch
        inputs = processor(images=img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            emb = outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
        return emb
    except Exception as e:
        logger.warning(f"DINOv2 embedding failed for {image_path}: {e}")
        return _histogram_embedding(img)


def _histogram_embedding(img) -> np.ndarray:
    """Fallback: compute a simple color histogram embedding."""
    import numpy as np
    img_small = img.resize((64, 64))
    arr = np.array(img_small, dtype=np.float32) / 255.0
    # 8 bins per channel
    hist = np.zeros(24, dtype=np.float32)
    for c in range(3):
        hist[c * 8:(c + 1) * 8] = np.histogram(arr[:, :, c], bins=8, range=(0, 1))[0]
    return hist / hist.sum() if hist.sum() > 0 else hist


def retrieve_pairs(
    image_paths: list[str],
    top_k: int = 20,
    image_root: str = "",
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Retrieve top-K image pairs by embedding similarity.

    Computes DINOv2 embeddings for all images, then returns pairs sorted by
    cosine similarity (highest first). ``image_root`` is prepended to each path.

    Returns a list of dicts with keys: image_a, image_b, score.
    """
    if not image_paths:
        return []

    # Compute embeddings for all images
    embeddings: dict[str, np.ndarray] = {}
    for i, path in enumerate(image_paths):
        full_path = str(Path(image_root) / path) if image_root else path
        emb = compute_image_embedding(full_path)
        embeddings[f"img_{i:04d}"] = emb

    # Compute pairwise cosine similarity
    ids = sorted(embeddings.keys())
    embs = np.stack([embeddings[i] for i in ids])
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embs_norm = embs / norms
    # Cosine similarity matrix
    sim = embs_norm @ embs_norm.T

    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(sim[i, j])
            if score >= min_score:
                pairs.append({
                    "image_a": ids[i],
                    "image_b": ids[j],
                    "score": score,
                })

    pairs.sort(key=lambda p: p["score"], reverse=True)
    return pairs[:top_k]


def retrieve_pairs_from_paths(
    image_paths: list[str],
    registered_ids: dict[str, str],
    top_k: int = 20,
    image_root: str = "",
) -> list[dict[str, Any]]:
    """Retrieve pairs using registered_ids (server-side image IDs).

    ``registered_ids`` maps local paths to server IDs (e.g., "img_0000").
    Returns pairs with server IDs.
    """
    if not image_paths:
        return []

    # Compute embeddings for all registered images
    embeddings: dict[str, np.ndarray] = {}
    path_to_id = {}
    for path, img_id in registered_ids.items():
        full_path = str(Path(image_root) / path) if image_root else path
        emb = compute_image_embedding(full_path)
        embeddings[img_id] = emb
        path_to_id[path] = img_id

    ids = sorted(embeddings.keys())
    embs = np.stack([embeddings[i] for i in ids])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embs_norm = embs / norms
    sim = embs_norm @ embs_norm.T

    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(sim[i, j])
            pairs.append({
                "image_a": ids[i],
                "image_b": ids[j],
                "score": score,
            })

    pairs.sort(key=lambda p: p["score"], reverse=True)
    return pairs[:top_k]
