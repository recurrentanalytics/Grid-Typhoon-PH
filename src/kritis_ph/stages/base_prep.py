"""Benchmark network + population prep (notebook Cell 11)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from kritis_ph import paths as P
from kritis_ph.config import RunConfig
from kritis_ph.io import (
    cache_csv_path,
    cache_pkl_path,
    load_dataframe,
    load_pickle,
    save_dataframe,
    save_pickle,
    write_manifest,
)
from kritis_ph.network.prep_helpers import (
    assign_population_to_nearest_bus,
    build_line_samples,
    load_worldpop_xyz,
)

_BASE_PREP_CSV = (
    "base_buses",
    "base_lines",
    "base_generators",
    "line_df",
    "line_samples",
    "pop_ph",
    "buses_nb",
    "pop_assignment",
)


def artifacts_ready(cfg: RunConfig) -> bool:
    """True if all Cell 11 cache files exist for this MODEL_TAG."""
    for n in _BASE_PREP_CSV:
        if not cache_csv_path(cfg, n).is_file():
            return False
    return cache_pkl_path(cfg, "G_base").is_file()


def _normalize_loaded(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    generators: pd.DataFrame,
    line_df: pd.DataFrame,
    line_samples: pd.DataFrame,
    buses_nb: pd.DataFrame,
) -> None:
    for c in ("bus0", "bus1"):
        if c in lines.columns:
            lines[c] = lines[c].astype(str)
    if "bus" in generators.columns:
        generators["bus"] = generators["bus"].astype(str)
    buses["bus_id"] = buses["bus_id"].astype(str)
    if "line_id" in line_df.columns:
        line_df["line_id"] = line_df["line_id"].astype(str)
    if "bus_id" in buses_nb.columns:
        buses_nb["bus_id"] = buses_nb["bus_id"].astype(str)
    if "line_id" in line_samples.columns:
        line_samples["line_id"] = line_samples["line_id"].astype(str)


def load_bundle(cfg: RunConfig) -> dict[str, Any]:
    """Load base-prep artefacts from cache (same dtypes as notebook else-branch)."""
    buses = load_dataframe(cfg, "base_buses")
    lines = load_dataframe(cfg, "base_lines")
    generators = load_dataframe(cfg, "base_generators")
    line_df = load_dataframe(cfg, "line_df")
    G_base = load_pickle(cfg, "G_base")
    line_samples = load_dataframe(cfg, "line_samples")
    pop_ph = load_dataframe(cfg, "pop_ph")
    buses_nb = load_dataframe(cfg, "buses_nb")
    pop_assignment = load_dataframe(cfg, "pop_assignment")
    _normalize_loaded(buses, lines, generators, line_df, line_samples, buses_nb)
    return {
        "buses": buses,
        "lines": lines,
        "generators": generators,
        "line_df": line_df,
        "G_base": G_base,
        "line_samples": line_samples,
        "pop_ph": pop_ph,
        "buses_nb": buses_nb,
        "pop_assignment": pop_assignment,
    }


def run(cfg: RunConfig) -> dict[str, Any]:
    """Read PyPSA-PH + WorldPop, build graph and assignments, write cache + manifest.

    Returns the same bundle as :func:`load_bundle` (in-memory, dtypes normalized).
    """
    wp_path = P.worldpop_csv(cfg.repo_root)
    if not wp_path.is_file():
        raise FileNotFoundError(
            f"WorldPop CSV missing: {wp_path}\n"
            "Place phl_pd_2020_1km_ASCII_XYZ.csv under data/worldpop_ph_2020_1km/"
        )

    cfg.cache_dir()
    cfg.run_artifact_dir()

    buses = (
        pd.read_csv(P.buses_csv(cfg.repo_root))
        .rename(columns={"name": "bus_id", "x": "lon", "y": "lat"})
        .copy()
    )
    for c in ("lon", "lat", "v_nom"):
        if c in buses.columns:
            buses[c] = pd.to_numeric(buses[c], errors="coerce")
    buses = buses.dropna(subset=["bus_id", "lon", "lat"]).copy()
    buses["bus_id"] = buses["bus_id"].astype(str)

    lines = pd.read_csv(P.lines_csv(cfg.repo_root)).copy()
    lines["bus0"] = lines["bus0"].astype(str)
    lines["bus1"] = lines["bus1"].astype(str)

    generators = pd.read_csv(P.generators_csv(cfg.repo_root)).copy()
    if "bus" in generators.columns:
        generators["bus"] = generators["bus"].astype(str)

    print("Buses:", len(buses))
    print("Lines:", len(lines))
    print("Generators:", len(generators))

    line_df = lines.merge(
        buses[["bus_id", "lat", "lon"]],
        left_on="bus0",
        right_on="bus_id",
        how="left",
    ).rename(columns={"lat": "lat0", "lon": "lon0"}).drop(columns=["bus_id"])

    line_df = line_df.merge(
        buses[["bus_id", "lat", "lon"]],
        left_on="bus1",
        right_on="bus_id",
        how="left",
    ).rename(columns={"lat": "lat1", "lon": "lon1"}).drop(columns=["bus_id"])

    line_df = line_df.dropna(subset=["lat0", "lon0", "lat1", "lon1"]).copy()
    line_df["lat_mid"] = (line_df["lat0"] + line_df["lat1"]) / 2.0
    line_df["lon_mid"] = (line_df["lon0"] + line_df["lon1"]) / 2.0
    line_df["line_id"] = np.arange(len(line_df)).astype(str)

    G_base = nx.Graph()
    for _, b in buses.iterrows():
        G_base.add_node(b["bus_id"])
    for _, l in line_df.iterrows():
        G_base.add_edge(l["bus0"], l["bus1"], line_id=l["line_id"])

    line_samples = build_line_samples(line_df, n_samples=cfg.line_sample_points)

    worldpop = load_worldpop_xyz(wp_path)
    b = cfg.ph_bounds
    pop_ph = worldpop[
        worldpop["lat"].between(b.lat_min, b.lat_max)
        & worldpop["lon"].between(b.lon_min, b.lon_max)
        & (worldpop["pop_value"] > 0)
    ].copy()

    buses_nb, pop_assignment = assign_population_to_nearest_bus(
        pop_ph=pop_ph, buses=buses
    )

    print("Graph nodes:", G_base.number_of_nodes())
    print("Graph edges:", G_base.number_of_edges())
    print("Population rows in PH box:", len(pop_ph))
    print("Population rows assigned:", len(pop_assignment))
    print("Unique buses receiving population:", int((buses_nb["bus_pop_local_nb"] > 0).sum()))
    print("Assigned total population:", f"{buses_nb['bus_pop_local_nb'].sum():,.0f}")
    print("True total population in pop_ph:", f"{pop_ph['pop_value'].sum():,.0f}")

    save_dataframe(cfg, buses, "base_buses")
    save_dataframe(cfg, lines, "base_lines")
    save_dataframe(cfg, generators, "base_generators")
    save_dataframe(cfg, line_df, "line_df")
    save_pickle(cfg, G_base, "G_base")
    save_dataframe(cfg, line_samples, "line_samples")
    save_dataframe(cfg, pop_ph, "pop_ph")
    save_dataframe(cfg, buses_nb, "buses_nb")
    save_dataframe(cfg, pop_assignment, "pop_assignment")
    print("[cache] Base / network prep written.")

    artefact_paths: dict[str, str] = {
        n: str(cache_csv_path(cfg, n)) for n in _BASE_PREP_CSV
    }
    artefact_paths["G_base"] = str(cache_pkl_path(cfg, "G_base"))

    write_manifest(
        cfg,
        "base_prep",
        {
            "stage": "base_prep",
            "run_id": cfg.resolved_run_id(),
            "model_tag": cfg.model_tag,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artefact_paths,
        },
    )

    bundle: dict[str, Any] = {
        "buses": buses,
        "lines": lines,
        "generators": generators,
        "line_df": line_df,
        "G_base": G_base,
        "line_samples": line_samples,
        "pop_ph": pop_ph,
        "buses_nb": buses_nb,
        "pop_assignment": pop_assignment,
    }
    _normalize_loaded(buses, lines, generators, line_df, line_samples, buses_nb)
    return bundle
