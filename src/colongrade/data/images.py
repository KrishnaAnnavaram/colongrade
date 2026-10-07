"""Image loading in RGB and the contrast transforms (numpy only).

* Colour stays: redness (erythema) is a main Mayo criterion. The prototype converted to grayscale.
* ``equalize`` and ``clahe`` work on the luminance only and keep the hue, so redness is not changed.
* ONE function, ``load_image``, prepares images for training, validation, test and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CONTRAST_MODES = ("none", "equalize", "clahe")


@dataclass(frozen=True)
class ImageConfig:
    size: int = 224
    contrast: str = "none"
    clip_limit: float = 2.0
    tiles: int = 8

    def __post_init__(self):
        if self.contrast not in CONTRAST_MODES:
            raise ValueError(f"contrast must be one of {CONTRAST_MODES}")


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _apply_luma(rgb: np.ndarray, new_luma: np.ndarray) -> np.ndarray:
    old = _luma(rgb)
    scale = np.where(old > 1e-6, new_luma / np.maximum(old, 1e-6), 1.0)
    return np.clip(rgb * scale[..., None], 0.0, 1.0)


def equalize(rgb: np.ndarray, bins: int = 256) -> np.ndarray:
    """Global histogram equalization of the luminance. Input and output are float RGB in [0, 1]."""
    y = _luma(rgb)
    hist, edges = np.histogram(y, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(float)
    cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1.0)
    mapped = np.interp(y, edges[:-1], cdf)
    return _apply_luma(rgb, mapped)


def clahe(rgb: np.ndarray, clip_limit: float = 2.0, tiles: int = 8, bins: int = 256) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization of the luminance, with bilinear blending of tiles."""
    y = _luma(rgb)
    h, w = y.shape
    th, tw = max(1, h // tiles), max(1, w // tiles)
    ny, nx = max(1, h // th), max(1, w // tw)
    maps = np.zeros((ny, nx, bins))
    for i in range(ny):
        for j in range(nx):
            tile = y[i * th : (i + 1) * th if i < ny - 1 else h, j * tw : (j + 1) * tw if j < nx - 1 else w]
            hist, _ = np.histogram(tile, bins=bins, range=(0.0, 1.0))
            limit = max(1.0, clip_limit * tile.size / bins)
            excess = np.clip(hist - limit, 0, None).sum()
            hist = np.minimum(hist, limit) + excess / bins
            cdf = np.cumsum(hist)
            maps[i, j] = cdf / cdf[-1]
    idx = np.clip((y * (bins - 1)).astype(int), 0, bins - 1)
    # tile-centre coordinates and bilinear weights
    cy = (np.arange(h) + 0.5) / th - 0.5
    cx = (np.arange(w) + 0.5) / tw - 0.5
    y0 = np.clip(np.floor(cy).astype(int), 0, ny - 1)
    x0 = np.clip(np.floor(cx).astype(int), 0, nx - 1)
    y1, x1 = np.clip(y0 + 1, 0, ny - 1), np.clip(x0 + 1, 0, nx - 1)
    wy = np.clip(cy - y0, 0, 1)[:, None]
    wx = np.clip(cx - x0, 0, 1)[None, :]
    Y0, X0, Y1, X1 = y0[:, None], x0[None, :], y1[:, None], x1[None, :]
    top = maps[Y0, X0, idx] * (1 - wx) + maps[Y0, X1, idx] * wx
    bottom = maps[Y1, X0, idx] * (1 - wx) + maps[Y1, X1, idx] * wx
    return _apply_luma(rgb, top * (1 - wy) + bottom * wy)


def to_array(img: Image.Image, config: ImageConfig) -> np.ndarray:
    rgb = img.convert("RGB").resize((config.size, config.size), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if config.contrast == "equalize":
        arr = equalize(arr)
    elif config.contrast == "clahe":
        arr = clahe(arr, config.clip_limit, config.tiles)
    return arr.astype(np.float32)


def load_image(path: str | Path, config: ImageConfig) -> np.ndarray:
    """RGB float32 array (H, W, 3) in [0, 1]."""
    with Image.open(path) as img:
        return to_array(img, config)


def tanh_to_uint8(x: np.ndarray) -> np.ndarray:
    """Map a generator output in [-1, 1] (tanh) to uint8 [0, 255].

    The prototype multiplied a tanh output by 255 and cast to uint8, so negative values wrapped around.
    """
    return np.clip(np.round((np.asarray(x, dtype=np.float64) + 1.0) * 127.5), 0, 255).astype(np.uint8)
