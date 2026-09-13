"""Doppelganger detection model for Phase 2.

A *visual doppelganger* pair looks similar globally but depicts different
geometry: local features match well, yet essential-matrix verification
fails.  The detector fuses two evidence streams into a calibrated
confidence ``P(doppelganger)``:

1. Appearance (DINOv2): CLS-token cosine similarity plus a patch-token
   spatial-agreement analysis.  Doppelgangers have high patch-level
   similarity (textures "explain" each other) but *incoherent* patch
   correspondences — matched patches do not preserve spatial layout.
2. Geometry (matcher output): essential-matrix inlier ratio, raw match
   count, and the mean inlier Sampson residual in RANSAC-threshold units.

Features feed a logistic model (``ScoringWeights``) producing a bounded
probability.  Hard vetoes keep the score honest:

- ``inlier_ratio >= geom_veto_inlier`` — geometric verification passed,
  cap confidence (appearance cannot override strong geometry).
- ``similarity < sim_veto`` — the pair is simply unrelated, not a
  doppelganger; cap confidence.
- missing geometry / missing appearance → capped confidence.

``ScoringWeights`` holds prior values chosen to match expected feature
distributions (documented per-field); the dataclass makes them explicit
and replaceable once a labelled doppelganger set exists to fit them.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 22M-param ViT-S/14 — ~90 MB on GPU, fits easily next to LoFTR/MASt3R.
DEFAULT_EMBED_MODEL = "facebook/dinov2-small"

# Patch-token best-match similarity above this counts as "explained".
_PATCH_MATCH_SIM = 0.5
# Displacement cells a patch match may deviate from the modal offset.
_PATCH_DISP_TOL = 1
# Max cached image embeddings on the server (images are re-used across
# match / doppelganger_check calls within an episode).
_EMB_CACHE_SIZE = 512


@dataclass
class ScoringWeights:
    """Logistic weights for P(doppelganger).

    Feature transforms (applied in ``DoppelgangerDetector.score``):
      x_sim   = (cos_sim - sim_mid) / sim_scale
      x_cons  = (patch_mid - patch_consistency) / patch_scale   (incoherent -> +)
      x_cov   = (patch_coverage - cov_mid) / cov_scale
      geom    = clip(inlier_ratio / geom_ratio_full)
                * clip(num_inliers / geom_count_full)           (0..1 verified)
      x_geom  = w_geom * (0.5 - geom) / 0.5                     (verified -> -)
      x_match = clip(num_matches / matches_scale, 0, 1)
      x_resid = clip(residual_units - resid_free, 0, resid_cap)
    logit = bias + w_sim*x_sim + w_patch_cons*x_cons + w_patch_cov*x_cov
                  + x_geom + w_matches*x_match + w_resid*x_resid
    """

    # appearance
    w_sim: float = 1.6
    sim_mid: float = 0.65       # DINOv2 cosine midpoint (same scene ~0.8+)
    sim_scale: float = 0.15
    w_patch_cons: float = 0.4
    patch_mid: float = 0.45
    patch_scale: float = 0.20
    w_patch_cov: float = 0.5
    cov_mid: float = 0.50
    cov_scale: float = 0.25
    # geometry: geom_score in [0,1] =
    #   clip(inlier_ratio / geom_ratio_full) * clip(num_inliers / geom_count_full)
    # (multiplicative: a high ratio on 5 points is weak; many inliers at a
    # low ratio is weak; ~30+ inliers at ~30% ratio is a verified pair —
    # COLMAP's own two-view acceptance is ~15 inliers).
    w_geom: float = 2.0
    geom_ratio_full: float = 0.30
    geom_count_full: float = 30.0
    geom_unknown_support: float = 0.75  # count term when num_inliers missing
    w_matches: float = 0.4
    matches_scale: float = 300.0
    w_resid: float = 0.3
    resid_free: float = 0.7     # residual units below this are "clean"
    resid_cap: float = 2.0
    bias: float = -0.9
    # decision + vetoes
    threshold: float = 0.5
    verified_score: float = 0.6        # geom_score >= -> geometrically verified
    verified_cap: float = 0.30
    strong_verified_score: float = 0.8
    strong_verified_cap: float = 0.15
    sim_veto: float = 0.40
    sim_veto_cap: float = 0.35
    no_sim_cap: float = 0.5     # cap when appearance signal unavailable
    no_geom_cap: float = 0.75   # cap when geometry signal unavailable


class _DinoEmbedder:
    """Lazy DINOv2 wrapper returning CLS + patch tokens for a PIL batch."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._failed = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            _patch_broken_torchaudio()
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            if self.device:
                self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("DINOv2 embedder loaded on %s", self.device or "cpu")
            return True
        except Exception as e:
            logger.warning("DINOv2 unavailable (%s); using histogram fallback", e)
            self._failed = True
            return False

    def embed(self, pil_images: list) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (cls (N,D), patches (N,P,D)) — L2-normalized. None on failure."""
        if not self._load():
            return None
        try:
            import torch

            inputs = self._processor(images=pil_images, return_tensors="pt")
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            hs = out.last_hidden_state  # (N, 1 + P, D)
            cls = hs[:, 0].float().cpu().numpy()
            patches = hs[:, 1:].float().cpu().numpy()
            cls = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-8)
            patches = patches / np.maximum(
                np.linalg.norm(patches, axis=2, keepdims=True), 1e-8
            )
            return cls, patches
        except Exception as e:
            logger.warning("DINOv2 embedding failed: %s", e)
            return None


def _patch_broken_torchaudio() -> None:
    """Work around a broken torchaudio install blocking transformers v5.

    transformers>=5 imports ``torchaudio`` unconditionally inside
    ``audio_utils`` (pulled in by ``processing_utils`` -> ``modeling_layers``),
    so a CUDA-mismatched/broken torchaudio makes *every* model class
    unimportable — including vision-only ones like Dinov2Model.  Injecting a
    bare stub module is safe here: the audio helpers are never invoked on
    this code path.  No-op when torchaudio imports cleanly.
    """
    import importlib.machinery
    import importlib.util
    import sys
    import types

    try:
        import torchaudio  # noqa: F401
        return
    except Exception:
        pass
    if "torchaudio" not in sys.modules:
        stub = types.ModuleType("torchaudio")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
        stub.__version__ = "0.0.0"
        sys.modules["torchaudio"] = stub


def _histogram_embedding(pil_img) -> np.ndarray:
    """Fallback global embedding: 8-bin RGB histogram (24-D)."""
    img_small = pil_img.resize((64, 64))
    arr = np.asarray(img_small, dtype=np.float32) / 255.0
    hist = np.zeros(24, dtype=np.float32)
    for c in range(3):
        hist[c * 8 : (c + 1) * 8] = np.histogram(
            arr[:, :, c], bins=8, range=(0, 1)
        )[0]
    return hist / max(hist.sum(), 1e-8)


def _patch_stats(
    pa: np.ndarray | None, pb: np.ndarray | None
) -> tuple[float | None, float | None]:
    """(spatial_consistency, coverage) between two square patch grids.

    coverage: fraction of A-patches whose best B-match exceeds
    ``_PATCH_MATCH_SIM`` — how much of A is locally "explained" by B.
    consistency: fraction of those matches whose displacement agrees
    (±``_PATCH_DISP_TOL`` cells) with the modal displacement — whether the
    local matches form a coherent layout (true overlap) or are scattered
    (doppelganger).
    """
    if pa is None or pb is None or pa.size == 0 or pb.size == 0:
        return None, None
    na, nb = pa.shape[0], pb.shape[0]
    sims = pa @ pb.T  # (Na, Nb) cosine — rows already L2-normalized
    best_j = sims.argmax(axis=1)
    best_s = sims[np.arange(na), best_j]
    good = best_s >= _PATCH_MATCH_SIM
    coverage = float(good.mean())
    n_good = int(good.sum())
    ga = int(round(math.sqrt(na)))
    gb = int(round(math.sqrt(nb)))
    if ga * ga != na or gb * gb != nb or n_good < 8:
        return None, coverage
    rows = np.arange(na) // ga
    cols = np.arange(na) % ga
    disp = np.stack(
        [rows[good] - best_j[good] // gb, cols[good] - best_j[good] % gb],
        axis=1,
    )
    vals, counts = np.unique(disp, axis=0, return_counts=True)
    mode = vals[int(counts.argmax())]
    consistency = float((np.abs(disp - mode).max(axis=1) <= _PATCH_DISP_TOL).mean())
    return consistency, coverage


class DoppelgangerDetector:
    """Embedding + geometry fusion doppelganger detector.

    Usage on the tool server (lazy, shares the matcher GPU)::

        detector = DoppelgangerDetector(device="cuda:0")
        result = detector.check(path_a, path_b, match_result=match_dict)
    """

    def __init__(
        self,
        device: str | None = None,
        model_name: str = DEFAULT_EMBED_MODEL,
        weights: ScoringWeights | None = None,
    ):
        self.weights = weights or ScoringWeights()
        self._embedder = _DinoEmbedder(model_name=model_name, device=device)
        # path -> (cls (D,), patches (P,D) | None)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def _appearance(
        self, path_a: str, path_b: str
    ) -> tuple[float | None, float | None, float | None, str]:
        """Return (cosine_sim, patch_consistency, patch_coverage, tag)."""
        from PIL import Image

        try:
            pil_a = Image.open(path_a).convert("RGB")
            pil_b = Image.open(path_b).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open images for embedding: %s", e)
            return None, None, None, "no_image"

        emb = self._embedder.embed([pil_a, pil_b])
        if emb is not None:
            cls, patches = emb
            sim = float(cls[0] @ cls[1])
            cons, cov = _patch_stats(patches[0], patches[1])
            return sim, cons, cov, "dinov2"

        # Histogram fallback: global similarity only.
        ha = _histogram_embedding(pil_a)
        hb = _histogram_embedding(pil_b)
        sim = float(ha @ hb / max(np.linalg.norm(ha) * np.linalg.norm(hb), 1e-8))
        return sim, None, None, "histogram"

    def _appearance_cached(
        self, path_a: str, path_b: str
    ) -> tuple[float | None, float | None, float | None, str]:
        key = (os.path.abspath(path_a), os.path.abspath(path_b))
        ck = self._cache.get("||".join(key))
        if ck is not None:
            return ck
        result = self._appearance(path_a, path_b)
        if len(self._cache) >= _EMB_CACHE_SIZE:
            # Evict oldest half (dicts are insertion-ordered).
            for k in list(self._cache)[: _EMB_CACHE_SIZE // 2]:
                del self._cache[k]
        self._cache["||".join(key)] = result
        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        similarity_score: float | None,
        patch_consistency: float | None,
        patch_coverage: float | None,
        inlier_ratio: float | None,
        num_matches: int | None,
        residual_units: float | None,
        num_inliers: int | None = None,
    ) -> tuple[float, dict[str, float], float | None]:
        """Logistic combination of features -> (confidence, terms, geom_score).

        ``geom_score`` in [0, 1] is the geometric-verification strength
        (inlier ratio scaled by inlier-count support); also returned so the
        caller can use it for verdicts.
        """
        w = self.weights
        terms: dict[str, float] = {}
        logit = w.bias

        if similarity_score is not None:
            t = w.w_sim * (similarity_score - w.sim_mid) / w.sim_scale
            terms["sim"] = t
            logit += t
        if patch_consistency is not None:
            t = w.w_patch_cons * (w.patch_mid - patch_consistency) / w.patch_scale
            terms["patch_consistency"] = t
            logit += t
        if patch_coverage is not None:
            t = w.w_patch_cov * (patch_coverage - w.cov_mid) / w.cov_scale
            terms["patch_coverage"] = t
            logit += t

        geom_score: float | None = None
        if inlier_ratio is not None:
            ratio_term = min(inlier_ratio / w.geom_ratio_full, 1.0)
            if num_inliers is None:
                count_term = w.geom_unknown_support
            else:
                count_term = min(num_inliers / w.geom_count_full, 1.0)
            geom_score = ratio_term * count_term
            t = w.w_geom * (0.5 - geom_score) / 0.5
            terms["geometry"] = t
            logit += t
        if num_matches is not None:
            t = w.w_matches * min(float(num_matches) / w.matches_scale, 1.0)
            terms["num_matches"] = t
            logit += t
        if residual_units is not None:
            t = w.w_resid * min(
                max(residual_units - w.resid_free, 0.0), w.resid_cap
            )
            terms["residual"] = t
            logit += t

        confidence = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))

        # Vetoes: hard caps so one strong contrary signal keeps the score
        # honest regardless of the remaining evidence.
        if geom_score is not None:
            if geom_score >= w.strong_verified_score:
                confidence = min(confidence, w.strong_verified_cap)
            elif geom_score >= w.verified_score:
                confidence = min(confidence, w.verified_cap)
        if similarity_score is not None and similarity_score < w.sim_veto:
            confidence = min(confidence, w.sim_veto_cap)
        if similarity_score is None:
            confidence = min(confidence, w.no_sim_cap)
        if inlier_ratio is None:
            confidence = min(confidence, w.no_geom_cap)
        return confidence, terms, geom_score

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def check(
        self,
        path_a: str,
        path_b: str,
        match_result: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Full doppelganger check for an image pair.

        ``match_result`` is the output of the ``/match`` endpoint (LoFTR).
        Returns a dict with is_doppelganger, confidence, similarity_score,
        inlier_ratio and diagnostics.
        """
        match_result = match_result or {}
        geo_failed = bool(match_result.get("error"))
        inlier_ratio = (
            None if geo_failed else float(match_result.get("inlier_ratio") or 0.0)
        )
        num_matches = (
            None if geo_failed else int(match_result.get("num_matches") or 0)
        )
        # Keep None when the key is absent: "unknown support" must not weaken
        # a passed verification the way a genuine 0-inlier count would.
        num_inliers = match_result.get("num_inliers")
        num_inliers = None if (geo_failed or num_inliers is None) else int(num_inliers)
        residual_units = match_result.get("residual_units")
        if residual_units is not None:
            residual_units = float(residual_units)

        appearance = (
            self._appearance_cached(path_a, path_b)
            if use_cache
            else self._appearance(path_a, path_b)
        )
        sim, cons, cov, tag = appearance

        confidence, terms, geom_score = self.score(
            similarity_score=sim,
            patch_consistency=cons,
            patch_coverage=cov,
            inlier_ratio=inlier_ratio,
            num_matches=num_matches,
            residual_units=residual_units,
            num_inliers=num_inliers,
        )
        is_doppel = confidence >= self.weights.threshold

        if is_doppel:
            verdict = "doppelganger"
        elif geom_score is not None and geom_score >= self.weights.verified_score:
            verdict = "match"
        elif sim is not None and sim < self.weights.sim_veto:
            verdict = "dissimilar"
        else:
            verdict = "uncertain"

        method = tag + ("+geometry" if not geo_failed else "")
        return {
            "is_doppelganger": bool(is_doppel),
            "confidence": float(confidence),
            "score": float(confidence),  # backward-compat alias
            "similarity_score": None if sim is None else float(sim),
            "patch_consistency": None if cons is None else float(cons),
            "patch_coverage": None if cov is None else float(cov),
            "inlier_ratio": None if inlier_ratio is None else float(inlier_ratio),
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "geom_score": None if geom_score is None else float(geom_score),
            "residual_units": residual_units,
            "verdict": verdict,
            "method": method,
            "threshold": float(self.weights.threshold),
            "logit_terms": {k: round(v, 4) for k, v in terms.items()},
        }
