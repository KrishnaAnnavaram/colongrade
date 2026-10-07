"""The Mayo endoscopic subscore: four ORDERED grades, and the remission / active split."""

from __future__ import annotations

import re

GRADES = (0, 1, 2, 3)
GRADE_NAMES = {0: "Mayo 0 (normal or inactive)", 1: "Mayo 1 (mild)", 2: "Mayo 2 (moderate)", 3: "Mayo 3 (severe)"}
_PATTERN = re.compile(r"^(?:mayo[\s_-]*)?([0-3])$", re.IGNORECASE)


class LabelError(ValueError):
    """Raised when a folder name is not a Mayo grade."""


def parse_grade(name: str) -> int:
    """``"Mayo 2"``, ``"mayo_2"``, ``"MAYO-2"`` and ``"2"`` all give 2."""
    m = _PATTERN.match(str(name).strip())
    if not m:
        raise LabelError(f"not a Mayo grade: {name!r}")
    return int(m.group(1))


def is_grade_folder(name: str) -> bool:
    return bool(_PATTERN.match(str(name).strip()))


def active(grade: int) -> int:
    """1 for active disease (Mayo 2-3), 0 for remission (Mayo 0-1), the split that many trials use."""
    return int(grade >= 2)
