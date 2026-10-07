"""The manifest: one row per image of the dataset. No sub-sampling.

The prototype kept 5 images per class (20 for training, 20 for testing) out of about 11 000.
``build_manifest`` lists EVERY image under the root and reads:

* the grade from the nearest parent folder that is a Mayo grade (``Mayo 0`` ... ``Mayo 3``),
* the split from the top-level folder: a name with ``test`` is the official test set, else ``trainval``,
* the patient from a parent folder named like ``patient_12`` or a pure number, if there is one.

``assign_folds`` makes stratified folds on ``trainval``, grouped by patient when patients are known.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ..labels import is_grade_folder, parse_grade

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_PATIENT = re.compile(r"^(?:patient[_\s-]*)?(\d+)$", re.IGNORECASE)
COLUMNS = ["image_id", "path", "mayo", "split", "patient", "fold"]


class ManifestError(ValueError):
    """Raised when the manifest is empty, unbalanced in a harmful way, or leaks across splits."""


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def build_manifest(root: str | Path, hash_files: bool = True) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        rel = path.relative_to(root)
        parents = list(rel.parts[:-1])
        grade_dirs = [p for p in parents if is_grade_folder(p)]
        if not grade_dirs:
            continue
        top = parents[0].lower() if parents else ""
        patient = next((m.group(1) for p in parents if (m := _PATIENT.match(p)) and not is_grade_folder(p)), "")
        rows.append({
            "image_id": rel.as_posix(),
            "path": rel.as_posix(),
            "mayo": parse_grade(grade_dirs[-1]),
            "split": "test" if "test" in top else "trainval",
            "patient": patient,
            "content_hash": _file_hash(path) if hash_files else "",
        })
    if not rows:
        raise ManifestError(f"no graded images under {root}")
    df = pd.DataFrame(rows)
    df["fold"] = -1
    return df


def assign_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 13) -> pd.DataFrame:
    """Stratified folds on ``trainval``. Grouped by patient when all trainval rows have one."""
    out = df.copy()
    tv = out[out["split"] == "trainval"]
    y = tv["mayo"].to_numpy()
    if (tv["patient"] != "").all():
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(np.zeros(len(tv)), y, groups=tv["patient"].to_numpy())
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(np.zeros(len(tv)), y)
    out["fold"] = -1
    for k, (_, held) in enumerate(folds):
        out.loc[tv.index[held], "fold"] = k
    check_manifest(out)
    return out


def check_manifest(df: pd.DataFrame) -> dict:
    """Leakage checks. Returns counts. Raises ``ManifestError`` on a leak."""
    if df["image_id"].duplicated().any():
        raise ManifestError("duplicate image ids")
    test, tv = df[df["split"] == "test"], df[df["split"] == "trainval"]
    shared_patients = (set(test["patient"]) & set(tv["patient"])) - {""}
    if shared_patients:
        raise ManifestError(f"{len(shared_patients)} patients occur in trainval and test")
    if "content_hash" in df.columns and (df["content_hash"] != "").all():
        shared = set(test["content_hash"]) & set(tv["content_hash"])
        if shared:
            raise ManifestError(f"{len(shared)} identical image files occur in trainval and test")
    if (tv["fold"] >= 0).all() and (tv["patient"] != "").all():
        if (tv.groupby("patient")["fold"].nunique() > 1).any():
            raise ManifestError("a patient occurs in more than one fold")
    return {"images": len(df), "trainval": len(tv), "test": len(test),
            "per_grade": {s: g["mayo"].value_counts().sort_index().to_dict() for s, g in df.groupby("split")}}


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient": str, "content_hash": str}, keep_default_na=False)
    check_manifest(df)
    return df
