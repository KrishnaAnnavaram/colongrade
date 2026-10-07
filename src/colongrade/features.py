"""Colour and texture features for the offline baseline (numpy only).

The features follow the Mayo criteria: redness (erythema), loss of the vessel pattern (less fine
dark structure), bright spots (fibrin, ulcers) and dark red spots (bleeding).
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: list[str] = []


def _hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h / 6.0, s, mx], -1)


def extract(rgb: np.ndarray) -> np.ndarray:
    """One feature vector for one RGB image in [0, 1]."""
    feats: dict[str, float] = {}
    hsv = _hsv(rgb)
    for c, name in enumerate("rgb"):
        ch = rgb[..., c]
        feats[f"{name}_mean"], feats[f"{name}_std"] = float(ch.mean()), float(ch.std())
        for q in (10, 50, 90):
            feats[f"{name}_p{q}"] = float(np.percentile(ch, q))
    for c, name in enumerate(("hue", "sat", "val")):
        feats[f"{name}_mean"], feats[f"{name}_std"] = float(hsv[..., c].mean()), float(hsv[..., c].std())
    total = rgb.sum(-1) + 1e-6
    redness = rgb[..., 0] / total
    feats["redness_mean"], feats["redness_p90"] = float(redness.mean()), float(np.percentile(redness, 90))
    feats["red_minus_green"] = float((rgb[..., 0] - rgb[..., 1]).mean())
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    rel = luma / (luma.mean() + 1e-6)  # brightness-independent
    gy, gx = np.gradient(rel)
    grad = np.hypot(gx, gy)
    feats["grad_mean"], feats["grad_p90"] = float(grad.mean()), float(np.percentile(grad, 90))
    feats["bright_spot_share"] = float(((rel > 1.35) & (hsv[..., 1] < 0.35)).mean())
    feats["dark_red_share"] = float(((rel < 0.6) & (redness > 0.6)).mean())
    hue_hist, _ = np.histogram(hsv[..., 0], bins=8, range=(0, 1))
    for i, v in enumerate(hue_hist / hue_hist.sum()):
        feats[f"hue_hist_{i}"] = float(v)
    if not FEATURE_NAMES:
        FEATURE_NAMES.extend(feats)
    return np.array(list(feats.values()), dtype=np.float32)


def extract_many(images) -> np.ndarray:
    return np.stack([extract(img) for img in images])
