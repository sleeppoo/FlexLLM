from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _array_body(text: str, variable: str) -> str:
    pattern = rf"{re.escape(variable)}\s*(?:\[[^\]]+\])+\s*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"could not find C array {variable}")
    return match.group(1).replace("{", " ").replace("}", " ")


@lru_cache(maxsize=64)
def load_float_array(header_path: str, variable: str, shape: tuple[int, ...]) -> np.ndarray:
    text = Path(header_path).read_text(encoding="utf-8")
    data = np.array(_FLOAT_RE.findall(_array_body(text, variable)), dtype=np.float32)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{variable} has {data.size} values, expected {expected}")
    return data.reshape(shape)


def load_scale_sum(
    parameters_dir: Path,
    header_name: str,
    scale_var: str,
    sum_var: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    header_path = str(parameters_dir / header_name)
    return (
        load_float_array(header_path, scale_var, shape),
        load_float_array(header_path, sum_var, shape),
    )
