"""Helpers lifted from notebook Cells 3–4 (WorldPop I/O, planar coords, population assignment, line sampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def latlon_to_km_xy(
    lat: np.ndarray | pd.Series,
    lon: np.ndarray | pd.Series,
    lat_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rough lat/lon → planar km for local nearest-neighbour logic."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat_ref is None:
        lat_ref = float(np.nanmean(lat))

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

    x = lon * km_per_deg_lon
    y = lat * km_per_deg_lat
    return x, y


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in km (supports NumPy broadcasting)."""
    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return earth_radius_km * c


def load_worldpop_xyz(path: str | Path) -> pd.DataFrame:
    """Read a WorldPop-style XYZ table (CSV or whitespace-separated)."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 3:
            raise ValueError("too few columns")
    except Exception:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["lon", "lat", "pop_value"],
            engine="python",
            comment="#",
        )

    if list(df.columns)[:3] != ["lon", "lat", "pop_value"]:
        df = df.iloc[:, :3].copy()
        df.columns = ["lon", "lat", "pop_value"]

    for c in ["lon", "lat", "pop_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["lon", "lat", "pop_value"]).copy()

    if (df["lon"] < 0).any():
        df.loc[df["lon"] < 0, "lon"] += 360

    return df


def assign_population_to_nearest_bus(
    pop_ph: pd.DataFrame, buses: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each population cell to the nearest bus in approximate km space."""
    if pop_ph.empty:
        raise ValueError("pop_ph is empty.")
    if buses.empty:
        raise ValueError("buses is empty.")

    buses_out = buses.copy()

    lat_ref = np.nanmean(
        np.concatenate(
            [
                pop_ph["lat"].to_numpy(dtype=float),
                buses["lat"].to_numpy(dtype=float),
            ]
        )
    )

    bus_x, bus_y = latlon_to_km_xy(
        buses_out["lat"].to_numpy(),
        buses_out["lon"].to_numpy(),
        lat_ref=lat_ref,
    )
    pop_x, pop_y = latlon_to_km_xy(
        pop_ph["lat"].to_numpy(),
        pop_ph["lon"].to_numpy(),
        lat_ref=lat_ref,
    )

    tree = cKDTree(np.column_stack([bus_x, bus_y]))
    dist_km, idx = tree.query(np.column_stack([pop_x, pop_y]), k=1)

    pop_assignment = pop_ph.copy()
    pop_assignment["nearest_bus_idx"] = idx.astype(int)
    pop_assignment["nearest_bus_dist_km"] = dist_km.astype(float)
    pop_assignment["nearest_bus_id"] = buses_out.iloc[idx]["bus_id"].to_numpy()

    bus_pop_nb = (
        pop_assignment.groupby("nearest_bus_id", as_index=False)["pop_value"]
        .sum()
        .rename(columns={"nearest_bus_id": "bus_id", "pop_value": "bus_pop_local_nb"})
    )

    buses_out = buses_out.merge(bus_pop_nb, on="bus_id", how="left")
    buses_out["bus_pop_local_nb"] = buses_out["bus_pop_local_nb"].fillna(0.0)

    return buses_out, pop_assignment


def build_line_samples(line_df: pd.DataFrame, n_samples: int = 7) -> pd.DataFrame:
    """Sample points along each transmission line for hazard evaluation."""
    sample_rows: list[dict[str, object]] = []

    for _, row in line_df.iterrows():
        lats = np.linspace(row["lat0"], row["lat1"], n_samples)
        lons = np.linspace(row["lon0"], row["lon1"], n_samples)

        for sample_idx, (lat, lon) in enumerate(zip(lats, lons)):
            sample_rows.append(
                {
                    "line_id": str(row["line_id"]),
                    "bus0": row["bus0"],
                    "bus1": row["bus1"],
                    "sample_idx": int(sample_idx),
                    "lat": float(lat),
                    "lon": float(lon),
                }
            )

    return pd.DataFrame(sample_rows)
