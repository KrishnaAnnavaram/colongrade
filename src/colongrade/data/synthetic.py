"""Synthetic endoscopy-like images in a LIMUC-like folder layout (no download, no patient data).

These are drawings, not medical images. They imitate the visual cues of the Mayo grades so that
the pipeline can be tested end to end:

* Mayo 0: pink mucosa with a clear vessel pattern
* Mayo 1: more redness, fewer vessels
* Mayo 2: marked redness, no vessels, small erosions
* Mayo 3: deep redness, larger ulcers, dark spots of bleeding

Each patient has its own base tone and illumination, and grades are imbalanced (45/30/15/10 %),
as in real endoscopy data. 8 % of labels are moved to a neighbour grade (rater disagreement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PRIORS = (0.45, 0.30, 0.15, 0.10)


def _draw(rng: np.random.Generator, grade: int, size: int, tone: np.ndarray, light: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    red_boost = [0.0, 0.07, 0.14, 0.20][grade] + rng.normal(0, 0.05)
    base = tone + np.array([red_boost * 0.25, -red_boost, -red_boost * 0.9])
    img = np.ones((size, size, 3)) * base
    img += rng.normal(0, 0.03, (size, size, 3))
    n_vessels = int(rng.integers(*[(3, 9), (1, 6), (0, 3), (0, 1)][grade]))
    for _ in range(n_vessels):  # dark thin branching lines
        x0, y0 = rng.random(2)
        angle = rng.random() * np.pi
        t = np.linspace(0, 1, 4 * size)
        px = np.clip(((x0 + 0.6 * t * np.cos(angle) + 0.05 * np.sin(12 * t)) * size).astype(int), 0, size - 1)
        py = np.clip(((y0 + 0.6 * t * np.sin(angle)) * size).astype(int), 0, size - 1)
        img[py, px] = img[py, px] * 0.55 + np.array([0.45, 0.05, 0.08]) * 0.45
    n_ulcers = int(rng.integers(*[(0, 1), (0, 2), (0, 4), (1, 6)][grade]))
    radius = [0.02, 0.025, 0.035, 0.06][grade]
    for _ in range(n_ulcers):  # yellow-white fibrin spots
        cx, cy = rng.random(2)
        r = radius * (0.6 + 0.8 * rng.random())
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        img[mask] = np.array([0.93, 0.88, 0.70]) + rng.normal(0, 0.02, 3)
    if grade == 3:
        for _ in range(int(rng.integers(2, 5))):  # dark red bleeding
            cx, cy = rng.random(2)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.05 * (0.5 + rng.random())) ** 2
            img[mask] = np.array([0.45, 0.02, 0.05])
    vignette = 1.0 - 0.5 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    img = img * light * vignette[..., None]
    return np.clip(img, 0, 1)


def write_dataset(root: str | Path, n_patients: int = 40, images_per_patient: int = 12, test_patients: int = 10,
                  size: int = 64, seed: int = 13) -> Path:
    """Write ``root/train_and_validation_sets/<patient>/Mayo k/*.png`` and ``root/test_set/...``."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for p in range(n_patients):
        split = "test_set" if p < test_patients else "train_and_validation_sets"
        tone = np.array([0.80, 0.52, 0.50]) + rng.normal(0, 0.07, 3)
        severity_shift = rng.normal(0, 0.6)  # some patients are sicker than others
        for i in range(images_per_patient):
            grade = int(np.clip(rng.choice(4, p=PRIORS) + round(severity_shift * rng.random()), 0, 3))
            light = rng.uniform(0.55, 1.15)
            img = _draw(rng, grade, size, tone, light)
            label = grade
            if rng.random() < 0.08:
                label = int(np.clip(grade + rng.choice([-1, 1]), 0, 3))
            folder = root / split / f"{p:03d}" / f"Mayo {label}"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray((img * 255).astype(np.uint8)).save(folder / f"p{p:03d}_{i:03d}.png")
    return root
