"""Normalisation utilities (notebook Cell 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalise a pandas Series to [0, 1]; constant series → zeros."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min = s.min()
    s_max = s.max()

    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    return (s - s_min) / (s_max - s_min)
