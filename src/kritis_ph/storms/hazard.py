"""Radial wind hazard on line samples (notebook Cell 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kritis_ph.network.prep_helpers import haversine_km


def radial_wind_factor(
    distance_km: np.ndarray,
    rmax_km: float = 35.0,
    inner_core_floor: float = 0.85,
    outer_decay_power: float = 1.35,
) -> np.ndarray:
    """Stylised radial decay: inner core + power-law outer tail."""
    d = np.asarray(distance_km, dtype=float)
    d = np.maximum(d, 0.0)
    rmax_km = max(float(rmax_km), 1.0)

    out = np.zeros_like(d, dtype=float)

    inner = d <= rmax_km
    outer = ~inner

    if np.any(inner):
        x = d[inner] / rmax_km
        out[inner] = inner_core_floor + (1.0 - inner_core_floor) * x

    if np.any(outer):
        x = d[outer] / rmax_km
        out[outer] = x ** (-outer_decay_power)

    return np.clip(out, 0.0, 1.0)


def estimate_rmax_km_from_point(
    storm_point: pd.Series,
    default_rmax_km: float = 35.0,
    scaling: bool = False,
) -> float:
    """Optional dynamic Rmax from wind speed."""
    if not scaling:
        return float(default_rmax_km)

    wind_kph = float(storm_point.get("wind_kph", np.nan))
    if not np.isfinite(wind_kph):
        return float(default_rmax_km)

    rmax = default_rmax_km + 0.08 * max(wind_kph - 120.0, 0.0)
    return float(np.clip(rmax, 20.0, 80.0))


def compute_line_hazard_up_to_t_sampled(
    storm_points_up_to_t: pd.DataFrame,
    line_samples: pd.DataFrame,
    line_df: pd.DataFrame,
    search_radius_km: float,
    global_max_wind_kph: float,
    rmax_km_default: float = 35.0,
    rmax_scaling: bool = False,
    inner_core_floor: float = 0.85,
    outer_decay_power: float = 1.35,
    min_effective_wind_kph: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-line max hazard up to timestep t over sampled line points."""
    sample_hmax = np.zeros(len(line_samples), dtype=float)
    sample_lat = line_samples["lat"].to_numpy()
    sample_lon = line_samples["lon"].to_numpy()

    for _, sp in storm_points_up_to_t.iterrows():
        wind_kph = float(sp["wind_kph"])

        if not np.isfinite(wind_kph) or wind_kph < min_effective_wind_kph:
            continue

        d = haversine_km(sample_lat, sample_lon, sp["lat"], sp["lon"])
        near = d <= search_radius_km

        if not np.any(near):
            continue

        rmax_km = estimate_rmax_km_from_point(
            storm_point=sp,
            default_rmax_km=rmax_km_default,
            scaling=rmax_scaling,
        )

        wind_norm = wind_kph / global_max_wind_kph

        radial_factor = radial_wind_factor(
            distance_km=d[near],
            rmax_km=rmax_km,
            inner_core_floor=inner_core_floor,
            outer_decay_power=outer_decay_power,
        )

        hz = wind_norm * radial_factor
        sample_hmax[near] = np.maximum(sample_hmax[near], hz)

    sample_result = line_samples.copy()
    sample_result["hazard_sample_max"] = sample_hmax

    line_hazard = (
        sample_result.groupby("line_id", as_index=False)["hazard_sample_max"]
        .max()
        .rename(columns={"hazard_sample_max": "hazard_max"})
    )

    lf = line_df.merge(line_hazard, on="line_id", how="left")
    lf["hazard_max"] = lf["hazard_max"].fillna(0.0)

    return lf, sample_result


def logistic_fragility(hazard: np.ndarray, h0: float = 0.55, k: float = 18.0) -> np.ndarray:
    """Map normalised hazard to line failure probability."""
    h = np.asarray(hazard, dtype=float)
    p = 1.0 / (1.0 + np.exp(-k * (h - h0)))
    return np.clip(p, 0.0, 1.0)
