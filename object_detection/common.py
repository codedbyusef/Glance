"""Config loading, torch device choice and JPEG coding shared by this folder."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
ENV_PATH = HERE / ".env"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f) or {}


def load_env_file(path: Path = ENV_PATH) -> None:
    """Read KEY=value lines from object_detection/.env into os.environ (shell values win).
    Keeps the Vultr IP and token out of git; .env is gitignored by the root .gitignore."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pick_device(name: str = "auto") -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> BGR uint8 image. Raises ValueError on a bad frame."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode JPEG frame")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode JPEG frame")
    return buf.tobytes()
