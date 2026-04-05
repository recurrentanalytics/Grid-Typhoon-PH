"""Grid impact time series: v4b, v5r deterministic, v5r stochastic (notebook Cells 7–8)."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from kritis_ph.storms.generation import (
    build_component_adequacy_table,
    classify_served_components_relative_v3,
)
from kritis_ph.storms.hazard import (
    compute_line_hazard_up_to_t_sampled,
    logistic_fragility,
)


def simulate_grid_impact_timeseries_v4b(
    storm_id: str,
    ph_points: pd.DataFrame,
    line_df: pd.DataFrame,
    line_samples: pd.DataFrame,
    buses: pd.DataFrame,
    G_base: nx.Graph,
    gen_buses: set[str],
    search_radius_km: float,
    global_max_wind_kph: float,
    fail_threshold: float,
    rmax_km_default: float = 35.0,
    rmax_scaling: bool = False,
    inner_core_floor: float = 0.85,
    outer_decay_power: float = 1.35,
    min_effective_wind_kph: float = 30.0,
    bus_pop_col: str = "bus_pop_local_nb",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Deterministic v4b: any generator in component → served."""
    storm_track = (
        ph_points.loc[ph_points["storm_id"] == storm_id]
        .sort_values("t")
        .reset_index(drop=True)
        .copy()
    )
    if storm_track.empty:
        raise ValueError(f"No points for storm_id={storm_id}")

    failed_line_ids_cumulative: set[str] = set()
    records: list[dict[str, object]] = []
    step_outputs: list[dict[str, Any]] = []

    for t_now in storm_track["t"]:
        storm_points_up_to_t = storm_track.loc[storm_track["t"] <= t_now].copy()

        lf, sample_result = compute_line_hazard_up_to_t_sampled(
            storm_points_up_to_t=storm_points_up_to_t,
            line_samples=line_samples,
            line_df=line_df,
            search_radius_km=search_radius_km,
            global_max_wind_kph=global_max_wind_kph,
            rmax_km_default=rmax_km_default,
            rmax_scaling=rmax_scaling,
            inner_core_floor=inner_core_floor,
            outer_decay_power=outer_decay_power,
            min_effective_wind_kph=min_effective_wind_kph,
        )

        lf["failed_now"] = lf["hazard_max"] > fail_threshold
        newly_failed_ids = set(lf.loc[lf["failed_now"], "line_id"].astype(str))
        failed_line_ids_cumulative |= newly_failed_ids
        lf["failed_cumulative"] = lf["line_id"].astype(str).isin(failed_line_ids_cumulative)

        G_t = G_base.copy()
        for _, row in lf.loc[lf["failed_cumulative"]].iterrows():
            if G_t.has_edge(row["bus0"], row["bus1"]):
                G_t.remove_edge(row["bus0"], row["bus1"])

        served_buses: set[str] = set()
        for comp in nx.connected_components(G_t):
            if len(set(comp) & gen_buses) > 0:
                served_buses |= set(comp)

        bus_state = buses[["bus_id", "lat", "lon", bus_pop_col]].copy()
        bus_state["served"] = bus_state["bus_id"].isin(served_buses)
        bus_state["unserved"] = ~bus_state["served"]

        affected_population = bus_state.loc[bus_state["unserved"], bus_pop_col].sum()
        total_population_proxy = bus_state[bus_pop_col].sum()

        rec: dict[str, object] = {
            "storm_id": storm_id,
            "storm_name": storm_track["storm_name"].iloc[0],
            "year": int(storm_track["year"].iloc[0]),
            "t": int(t_now),
            "n_track_points_used": int(len(storm_points_up_to_t)),
            "n_failed_lines_cumulative": int(lf["failed_cumulative"].sum()),
            "failed_line_share_cumulative": float(lf["failed_cumulative"].mean()),
            "n_unserved_buses": int(bus_state["unserved"].sum()),
            "unserved_bus_share": float(bus_state["unserved"].mean()),
            "affected_population_proxy": float(affected_population),
            "affected_population_share_proxy": (
                float(affected_population) / float(total_population_proxy)
                if total_population_proxy > 0
                else np.nan
            ),
            "n_components": int(nx.number_connected_components(G_t)),
            "max_line_hazard_this_step": float(lf["hazard_max"].max()),
        }
        records.append(rec)

        step_outputs.append(
            {
                "t": int(t_now),
                "storm_points_up_to_t": storm_points_up_to_t.copy(),
                "lf": lf.copy(),
                "sample_result": sample_result.copy(),
                "bus_state": bus_state.copy(),
                "graph": G_t.copy(),
                "record": rec.copy(),
            }
        )

    return pd.DataFrame(records), step_outputs


def simulate_grid_impact_timeseries_v5r(
    storm_id: str,
    ph_points: pd.DataFrame,
    line_df: pd.DataFrame,
    line_samples: pd.DataFrame,
    buses: pd.DataFrame,
    G_base: nx.Graph,
    gen_bus_cap: pd.DataFrame,
    baseline_mw_per_person_proxy: float,
    search_radius_km: float,
    global_max_wind_kph: float,
    fail_threshold: float,
    rmax_km_default: float = 35.0,
    rmax_scaling: bool = False,
    inner_core_floor: float = 0.85,
    outer_decay_power: float = 1.35,
    min_effective_wind_kph: float = 30.0,
    bus_pop_col: str = "bus_pop_local_nb",
    min_relative_ratio_multi: float = 0.35,
    min_relative_ratio_single: float = 0.75,
    min_total_gen_mw: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Deterministic v5r: relative adequacy after fragmentation."""
    storm_track = (
        ph_points.loc[ph_points["storm_id"] == storm_id]
        .sort_values("t")
        .reset_index(drop=True)
        .copy()
    )
    if storm_track.empty:
        raise ValueError(f"No points for storm_id={storm_id}")

    failed_line_ids_cumulative: set[str] = set()
    records: list[dict[str, object]] = []
    step_outputs: list[dict[str, Any]] = []

    for t_now in storm_track["t"]:
        storm_points_up_to_t = storm_track.loc[storm_track["t"] <= t_now].copy()

        lf, sample_result = compute_line_hazard_up_to_t_sampled(
            storm_points_up_to_t=storm_points_up_to_t,
            line_samples=line_samples,
            line_df=line_df,
            search_radius_km=search_radius_km,
            global_max_wind_kph=global_max_wind_kph,
            rmax_km_default=rmax_km_default,
            rmax_scaling=rmax_scaling,
            inner_core_floor=inner_core_floor,
            outer_decay_power=outer_decay_power,
            min_effective_wind_kph=min_effective_wind_kph,
        )

        lf["failed_now"] = lf["hazard_max"] > fail_threshold
        newly_failed_ids = set(lf.loc[lf["failed_now"], "line_id"].astype(str))
        failed_line_ids_cumulative |= newly_failed_ids
        lf["failed_cumulative"] = lf["line_id"].astype(str).isin(failed_line_ids_cumulative)

        G_t = G_base.copy()
        for _, row in lf.loc[lf["failed_cumulative"]].iterrows():
            if G_t.has_edge(row["bus0"], row["bus1"]):
                G_t.remove_edge(row["bus0"], row["bus1"])

        comp_df = build_component_adequacy_table(
            G_t=G_t,
            buses=buses,
            gen_bus_cap=gen_bus_cap,
            bus_pop_col=bus_pop_col,
        )

        comp_df = classify_served_components_relative_v3(
            comp_df=comp_df,
            baseline_mw_per_person_proxy=baseline_mw_per_person_proxy,
            min_relative_ratio_multi=min_relative_ratio_multi,
            min_relative_ratio_single=min_relative_ratio_single,
            min_total_gen_mw=min_total_gen_mw,
        )

        served_buses: set[str] = set()
        for _, comp_row in comp_df.loc[comp_df["served_component"]].iterrows():
            served_buses |= set(comp_row["bus_ids"])

        bus_state = buses[["bus_id", "lat", "lon", bus_pop_col]].copy()
        bus_state = bus_state.merge(gen_bus_cap, on="bus_id", how="left")
        bus_state["gen_capacity_mw"] = bus_state["gen_capacity_mw"].fillna(0.0)
        bus_state["served"] = bus_state["bus_id"].isin(served_buses)
        bus_state["unserved"] = ~bus_state["served"]

        affected_population = float(bus_state.loc[bus_state["unserved"], bus_pop_col].sum())
        total_population_proxy = float(bus_state[bus_pop_col].sum())

        rec = {
            "storm_id": storm_id,
            "storm_name": storm_track["storm_name"].iloc[0],
            "year": int(storm_track["year"].iloc[0]),
            "t": int(t_now),
            "n_track_points_used": int(len(storm_points_up_to_t)),
            "n_failed_lines_cumulative": int(lf["failed_cumulative"].sum()),
            "failed_line_share_cumulative": float(lf["failed_cumulative"].mean()),
            "n_unserved_buses": int(bus_state["unserved"].sum()),
            "unserved_bus_share": float(bus_state["unserved"].mean()),
            "affected_population_proxy": affected_population,
            "affected_population_share_proxy": (
                affected_population / total_population_proxy
                if total_population_proxy > 0
                else np.nan
            ),
            "n_components": int(nx.number_connected_components(G_t)),
            "n_served_components": int(comp_df["served_component"].sum()),
            "n_unserved_components": int((~comp_df["served_component"]).sum()),
            "max_line_hazard_this_step": float(lf["hazard_max"].max()),
        }
        records.append(rec)

        step_outputs.append(
            {
                "t": int(t_now),
                "storm_points_up_to_t": storm_points_up_to_t.copy(),
                "lf": lf.copy(),
                "sample_result": sample_result.copy(),
                "bus_state": bus_state.copy(),
                "graph": G_t.copy(),
                "component_table": comp_df.copy(),
                "record": rec.copy(),
            }
        )

    return pd.DataFrame(records), step_outputs


def simulate_grid_impact_timeseries_v5r_stochastic(
    storm_id: str,
    ph_points: pd.DataFrame,
    line_df: pd.DataFrame,
    line_samples: pd.DataFrame,
    buses: pd.DataFrame,
    G_base: nx.Graph,
    gen_bus_cap: pd.DataFrame,
    baseline_mw_per_person_proxy: float,
    search_radius_km: float,
    global_max_wind_kph: float,
    fragility_h0: float = 0.55,
    fragility_k: float = 18.0,
    rmax_km_default: float = 35.0,
    rmax_scaling: bool = False,
    inner_core_floor: float = 0.85,
    outer_decay_power: float = 1.35,
    min_effective_wind_kph: float = 30.0,
    bus_pop_col: str = "bus_pop_local_nb",
    min_relative_ratio_multi: float = 0.35,
    min_relative_ratio_single: float = 0.75,
    min_total_gen_mw: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Stochastic v5r: logistic fragility + random failures."""
    if rng is None:
        rng = np.random.default_rng()

    storm_track = (
        ph_points.loc[ph_points["storm_id"] == storm_id]
        .sort_values("t")
        .reset_index(drop=True)
        .copy()
    )
    if storm_track.empty:
        raise ValueError(f"No points for storm_id={storm_id}")

    failed_line_ids_cumulative: set[str] = set()
    records: list[dict[str, object]] = []
    step_outputs: list[dict[str, Any]] = []

    for t_now in storm_track["t"]:
        storm_points_up_to_t = storm_track.loc[storm_track["t"] <= t_now].copy()

        lf, sample_result = compute_line_hazard_up_to_t_sampled(
            storm_points_up_to_t=storm_points_up_to_t,
            line_samples=line_samples,
            line_df=line_df,
            search_radius_km=search_radius_km,
            global_max_wind_kph=global_max_wind_kph,
            rmax_km_default=rmax_km_default,
            rmax_scaling=rmax_scaling,
            inner_core_floor=inner_core_floor,
            outer_decay_power=outer_decay_power,
            min_effective_wind_kph=min_effective_wind_kph,
        )

        lf = lf.copy()
        lf["p_fail"] = logistic_fragility(
            lf["hazard_max"].to_numpy(),
            h0=fragility_h0,
            k=fragility_k,
        )

        lf["already_failed"] = lf["line_id"].astype(str).isin(failed_line_ids_cumulative)

        not_failed_mask = ~lf["already_failed"]
        lf["random_u"] = rng.random(len(lf))
        lf["failed_now"] = False
        lf.loc[not_failed_mask, "failed_now"] = (
            lf.loc[not_failed_mask, "random_u"] < lf.loc[not_failed_mask, "p_fail"]
        )

        newly_failed_ids = set(lf.loc[lf["failed_now"], "line_id"].astype(str))
        failed_line_ids_cumulative |= newly_failed_ids
        lf["failed_cumulative"] = lf["line_id"].astype(str).isin(failed_line_ids_cumulative)

        G_t = G_base.copy()
        for _, row in lf.loc[lf["failed_cumulative"]].iterrows():
            if G_t.has_edge(row["bus0"], row["bus1"]):
                G_t.remove_edge(row["bus0"], row["bus1"])

        comp_df = build_component_adequacy_table(
            G_t=G_t,
            buses=buses,
            gen_bus_cap=gen_bus_cap,
            bus_pop_col=bus_pop_col,
        )

        comp_df = classify_served_components_relative_v3(
            comp_df=comp_df,
            baseline_mw_per_person_proxy=baseline_mw_per_person_proxy,
            min_relative_ratio_multi=min_relative_ratio_multi,
            min_relative_ratio_single=min_relative_ratio_single,
            min_total_gen_mw=min_total_gen_mw,
        )

        served_buses: set[str] = set()
        for _, comp_row in comp_df.loc[comp_df["served_component"]].iterrows():
            served_buses |= set(comp_row["bus_ids"])

        bus_state = buses[["bus_id", "lat", "lon", bus_pop_col]].copy()
        bus_state = bus_state.merge(gen_bus_cap, on="bus_id", how="left")
        bus_state["gen_capacity_mw"] = bus_state["gen_capacity_mw"].fillna(0.0)
        bus_state["served"] = bus_state["bus_id"].isin(served_buses)
        bus_state["unserved"] = ~bus_state["served"]

        affected_population = float(bus_state.loc[bus_state["unserved"], bus_pop_col].sum())
        total_population_proxy = float(bus_state[bus_pop_col].sum())

        rec = {
            "storm_id": storm_id,
            "storm_name": storm_track["storm_name"].iloc[0],
            "year": int(storm_track["year"].iloc[0]),
            "t": int(t_now),
            "n_track_points_used": int(len(storm_points_up_to_t)),
            "n_failed_lines_cumulative": int(lf["failed_cumulative"].sum()),
            "failed_line_share_cumulative": float(lf["failed_cumulative"].mean()),
            "n_failed_lines_new": int(lf["failed_now"].sum()),
            "n_unserved_buses": int(bus_state["unserved"].sum()),
            "unserved_bus_share": float(bus_state["unserved"].mean()),
            "affected_population_proxy": affected_population,
            "affected_population_share_proxy": (
                affected_population / total_population_proxy
                if total_population_proxy > 0
                else np.nan
            ),
            "n_components": int(nx.number_connected_components(G_t)),
            "n_served_components": int(comp_df["served_component"].sum()),
            "n_unserved_components": int((~comp_df["served_component"]).sum()),
            "max_line_hazard_this_step": float(lf["hazard_max"].max()),
            "mean_line_p_fail_this_step": float(lf["p_fail"].mean()),
            "max_line_p_fail_this_step": float(lf["p_fail"].max()),
        }
        records.append(rec)

        step_outputs.append(
            {
                "t": int(t_now),
                "storm_points_up_to_t": storm_points_up_to_t.copy(),
                "lf": lf.copy(),
                "sample_result": sample_result.copy(),
                "bus_state": bus_state.copy(),
                "graph": G_t.copy(),
                "component_table": comp_df.copy(),
                "record": rec.copy(),
            }
        )

    return pd.DataFrame(records), step_outputs
