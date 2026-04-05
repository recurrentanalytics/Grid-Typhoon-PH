"""Generator buses and component adequacy (notebook Cell 5)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def get_generator_buses_for_storm_year(
    generators: pd.DataFrame,
    graph_nodes,
    storm_year: int,
    use_commission_year: bool = False,
    year_col: str = "commissioning_year",
) -> set[str]:
    """Return generator buses present in the graph; optional year filter."""
    g = generators.copy()

    if use_commission_year and year_col in g.columns:
        g[year_col] = pd.to_numeric(g[year_col], errors="coerce")
        g = g[g[year_col].isna() | (g[year_col] <= storm_year)].copy()

    gen_buses = set(g["bus"].dropna().astype(str)) & set(graph_nodes)
    return gen_buses


def build_generator_bus_capacity(
    generators: pd.DataFrame, capacity_col: str = "p_nom"
) -> pd.DataFrame:
    """Aggregate installed generation capacity per bus."""
    g = generators.copy()

    if "bus" not in g.columns:
        raise ValueError("generators must contain column 'bus'")
    if capacity_col not in g.columns:
        raise ValueError(f"generators must contain column '{capacity_col}'")

    g["bus"] = g["bus"].astype(str)
    g[capacity_col] = pd.to_numeric(g[capacity_col], errors="coerce").fillna(0.0)

    return (
        g.groupby("bus", as_index=False)[capacity_col]
        .sum()
        .rename(columns={"bus": "bus_id", capacity_col: "gen_capacity_mw"})
    )


def build_component_adequacy_table(
    G_t: nx.Graph,
    buses: pd.DataFrame,
    gen_bus_cap: pd.DataFrame,
    bus_pop_col: str = "bus_pop_local_nb",
) -> pd.DataFrame:
    """Connected components and adequacy-relevant properties."""
    bus_state = buses[["bus_id", "lat", "lon", bus_pop_col]].copy()
    bus_state = bus_state.merge(gen_bus_cap, on="bus_id", how="left")
    bus_state["gen_capacity_mw"] = bus_state["gen_capacity_mw"].fillna(0.0)
    bus_state[bus_pop_col] = bus_state[bus_pop_col].fillna(0.0)

    rows: list[dict[str, object]] = []
    for comp_id, comp in enumerate(nx.connected_components(G_t), start=1):
        comp_set = set(comp)

        comp_buses = pd.DataFrame({"bus_id": list(comp_set)}).merge(
            bus_state, on="bus_id", how="left"
        )

        pop_proxy = float(comp_buses[bus_pop_col].sum())
        gen_cap = float(comp_buses["gen_capacity_mw"].sum())
        n_gen_buses = int((comp_buses["gen_capacity_mw"] > 0).sum())

        rows.append(
            {
                "component_id": comp_id,
                "bus_ids": list(comp_set),
                "n_buses": int(len(comp_set)),
                "n_generator_buses": n_gen_buses,
                "population_proxy": pop_proxy,
                "gen_capacity_mw": gen_cap,
                "mw_per_person_proxy": gen_cap / pop_proxy if pop_proxy > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_system_baseline_mw_per_person_proxy(
    G_base: nx.Graph,
    buses: pd.DataFrame,
    gen_bus_cap: pd.DataFrame,
    bus_pop_col: str = "bus_pop_local_nb",
) -> dict[str, float]:
    """Intact-system benchmark adequacy ratio for v5r."""
    comp_df = build_component_adequacy_table(
        G_t=G_base,
        buses=buses,
        gen_bus_cap=gen_bus_cap,
        bus_pop_col=bus_pop_col,
    )

    if len(comp_df) != 1:
        print(
            "Warning: base graph has more than one component. "
            "Using largest by population proxy."
        )

    base_row = comp_df.sort_values("population_proxy", ascending=False).iloc[0]

    return {
        "population_proxy": float(base_row["population_proxy"]),
        "gen_capacity_mw": float(base_row["gen_capacity_mw"]),
        "mw_per_person_proxy": float(base_row["mw_per_person_proxy"]),
    }


def classify_served_components_relative_v3(
    comp_df: pd.DataFrame,
    baseline_mw_per_person_proxy: float,
    min_relative_ratio_multi: float = 0.35,
    min_relative_ratio_single: float = 0.75,
    min_total_gen_mw: float = 1.0,
) -> pd.DataFrame:
    """Mark components as served using relative adequacy vs intact benchmark."""
    out = comp_df.copy()

    out["relative_adequacy"] = (
        out["mw_per_person_proxy"] / baseline_mw_per_person_proxy
        if baseline_mw_per_person_proxy > 0
        else np.nan
    )

    out["required_relative_ratio"] = np.where(
        out["n_buses"] == 1,
        min_relative_ratio_single,
        min_relative_ratio_multi,
    )

    out["served_component"] = (
        (out["gen_capacity_mw"] >= min_total_gen_mw)
        & (out["relative_adequacy"] >= out["required_relative_ratio"])
    )

    return out
