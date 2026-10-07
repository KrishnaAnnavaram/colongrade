"""Settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    out_dir: Path = Path("runs")
    seed: int = 13
    image_size: int = 224
    folds: int = 5
    device: str = "auto"


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)

    def _int(name: str, default: int, low: int) -> int:
        raw = env.get(name) or str(default)
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if value < low:
            raise ValueError(f"{name} must be at least {low}")
        return value

    device = (env.get("COLONGRADE_DEVICE") or "auto").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("COLONGRADE_DEVICE must be auto, cpu or cuda")
    return Settings(
        data_dir=Path(env.get("COLONGRADE_DATA_DIR") or "data"),
        out_dir=Path(env.get("COLONGRADE_OUT_DIR") or "runs"),
        seed=_int("COLONGRADE_SEED", 13, 0),
        image_size=_int("COLONGRADE_IMAGE_SIZE", 224, 16),
        folds=_int("COLONGRADE_FOLDS", 5, 2),
        device=device,
    )
