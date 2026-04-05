"""Full storm catalogue simulations and ranking (notebook Cell 15)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kritis_ph.analysis.normalize import minmax_normalize
from kritis_ph.config import RunConfig
from kritis_ph.io import (
    cache_csv_path,
    cache_pkl_path,
    load_dataframe,
    load_pickle,
    load_pickle_optional,
    save_dataframe,
    save_pickle,
    write_manifest,
)
from kritis_ph.stages import base_prep
from kritis_ph.storms.generation import get_generator_buses_for_storm_year
from kritis_ph.storms.simulate import (
    simulate_grid_impact_timeseries_v4b,
    simulate_grid_impact_timeseries_v5r,
    simulate_grid_impact_timeseries_v5r_stochastic,
)
from kritis_ph.storms.summary import summarize_storm_timeseries

_STORM_SUMMARY_NAMES = (
    "storm_summary_v4b",
    "storm_summary_v5r",
    "storm_summary_v5r_stoch",
)
_STORM_TS_NAMES = (
    "storm_ts_store_v4b",
    "storm_ts_store_v5r",
    "storm_ts_store_v5r_stoch",
)
_CATALOGUE_INPUTS = ("ph_points", "storm_meta_run")


def catalogue_inputs_ready(cfg: RunConfig) -> bool:
    """Track points + run list saved from the IBTrACS notebook cells."""
    return all(cache_csv_path(cfg, n).is_file() for n in _CATALOGUE_INPUTS)


def artifacts_ready(cfg: RunConfig) -> bool:
    """Storm summary CSVs present (matches notebook cache_ready_storm_summaries)."""
    return all(cache_csv_path(cfg, n).is_file() for n in _STORM_SUMMARY_NAMES)


def _apply_screening_scores(df: pd.DataFrame, metric_name: str) -> None:
    if df.empty:
        return
    df["peak_affected_population_proxy_norm"] = minmax_normalize(
        df["peak_affected_population_proxy"]
    )
    df["peak_n_failed_lines_norm"] = minmax_normalize(
        df["peak_n_failed_lines_cumulative"]
    )
    df["peak_n_unserved_buses_norm"] = minmax_normalize(df["peak_n_unserved_buses"])
    df[metric_name] = (
        0.60 * df["peak_affected_population_proxy_norm"]
        + 0.25 * df["peak_n_failed_lines_norm"]
        + 0.15 * df["peak_n_unserved_buses_norm"]
    )
    df.sort_values(metric_name, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["rank_index"] = df.index + 1


def load_results(cfg: RunConfig) -> dict[str, Any]:
    """Load cached storm summaries and optional per-storm time-series dicts."""
    rank_df_v4b = load_dataframe(cfg, "storm_summary_v4b")
    rank_df_v5r = load_dataframe(cfg, "storm_summary_v5r")
    rank_df_v5r_stoch = load_dataframe(cfg, "storm_summary_v5r_stoch")
    storm_ts_store_v4b = load_pickle_optional(cfg, "storm_ts_store_v4b") or {}
    storm_ts_store_v5r = load_pickle_optional(cfg, "storm_ts_store_v5r") or {}
    storm_ts_store_v5r_stoch = load_pickle_optional(cfg, "storm_ts_store_v5r_stoch") or {}
    return {
        "rank_df_v4b": rank_df_v4b,
        "rank_df_v5r": rank_df_v5r,
        "rank_df_v5r_stoch": rank_df_v5r_stoch,
        "storm_ts_store_v4b": storm_ts_store_v4b,
        "storm_ts_store_v5r": storm_ts_store_v5r,
        "storm_ts_store_v5r_stoch": storm_ts_store_v5r_stoch,
    }


def run(cfg: RunConfig) -> dict[str, Any]:
    """Run v4b, v5r deterministic, and one stochastic draw per storm; write cache."""
    if not catalogue_inputs_ready(cfg):
        raise FileNotFoundError(
            "Catalogue inputs missing (expected ph_points + storm_meta_run CSV in CACHE_DIR). "
            "Run the IBTrACS cells in the notebook through storm_meta_run, "
            "then re-run the cell that saves catalogue inputs."
        )
    if not base_prep.artifacts_ready(cfg):
        raise FileNotFoundError(
            "Base prep cache missing. Run: python -m kritis_ph build --stages base_prep"
        )

    cfg.cache_dir()
    cfg.run_artifact_dir()

    bundle = base_prep.load_bundle(cfg)
    generators = bundle["generators"]
    line_df = bundle["line_df"]
    G_base = bundle["G_base"]
    line_samples = bundle["line_samples"]
    buses_nb = bundle["buses_nb"]

    gen_bus_cap = load_dataframe(cfg, "gen_bus_cap")
    baseline_stats = load_pickle(cfg, "baseline_stats")

    ph_points = load_dataframe(cfg, "ph_points")
    storm_meta_run = load_dataframe(cfg, "storm_meta_run")

    global_max_wind = float(ph_points["wind_kph"].max())
    if pd.isna(global_max_wind) or global_max_wind <= 0:
        global_max_wind = 1.0

    storm_summaries_v4b: list[dict[str, object]] = []
    storm_summaries_v5r: list[dict[str, object]] = []
    storm_summaries_v5r_stoch: list[dict[str, object]] = []

    storm_ts_store_v4b: dict[str, pd.DataFrame] = {}
    storm_ts_store_v5r: dict[str, pd.DataFrame] = {}
    storm_ts_store_v5r_stoch: dict[str, pd.DataFrame] = {}

    pbar = tqdm(total=len(storm_meta_run), desc="Running storms")

    for storm_counter, (_, meta) in enumerate(storm_meta_run.iterrows()):
        storm_id = str(meta["storm_id"])
        storm_year = int(meta["year"])
        pbar.set_postfix_str(f"{storm_id} ({storm_year})")

        try:
            gen_buses_y = get_generator_buses_for_storm_year(
                generators=generators,
                graph_nodes=G_base.nodes(),
                storm_year=storm_year,
                use_commission_year=cfg.use_generator_commission_year,
                year_col=cfg.generator_year_column,
            )

            ts_df_v4b, _ = simulate_grid_impact_timeseries_v4b(
                storm_id=storm_id,
                ph_points=ph_points,
                line_df=line_df,
                line_samples=line_samples,
                buses=buses_nb,
                G_base=G_base,
                gen_buses=gen_buses_y,
                search_radius_km=cfg.search_radius_km,
                global_max_wind_kph=global_max_wind,
                fail_threshold=cfg.fail_threshold,
                rmax_km_default=cfg.rmax_km_default,
                rmax_scaling=cfg.rmax_scaling,
                inner_core_floor=cfg.inner_core_floor,
                outer_decay_power=cfg.outer_decay_power,
                min_effective_wind_kph=cfg.min_effective_wind_kph,
                bus_pop_col="bus_pop_local_nb",
            )
            storm_ts_store_v4b[storm_id] = ts_df_v4b.copy()
            storm_summaries_v4b.append(summarize_storm_timeseries(ts_df_v4b))

            ts_df_v5r, _ = simulate_grid_impact_timeseries_v5r(
                storm_id=storm_id,
                ph_points=ph_points,
                line_df=line_df,
                line_samples=line_samples,
                buses=buses_nb,
                G_base=G_base,
                gen_bus_cap=gen_bus_cap,
                baseline_mw_per_person_proxy=baseline_stats["mw_per_person_proxy"],
                search_radius_km=cfg.search_radius_km,
                global_max_wind_kph=global_max_wind,
                fail_threshold=cfg.fail_threshold,
                rmax_km_default=cfg.rmax_km_default,
                rmax_scaling=cfg.rmax_scaling,
                inner_core_floor=cfg.inner_core_floor,
                outer_decay_power=cfg.outer_decay_power,
                min_effective_wind_kph=cfg.min_effective_wind_kph,
                bus_pop_col="bus_pop_local_nb",
                min_relative_ratio_multi=cfg.min_relative_ratio_multi,
                min_relative_ratio_single=cfg.min_relative_ratio_single,
                min_total_gen_mw=cfg.min_total_gen_mw,
            )
            storm_ts_store_v5r[storm_id] = ts_df_v5r.copy()
            storm_summaries_v5r.append(summarize_storm_timeseries(ts_df_v5r))

            rng_storm = np.random.default_rng(10000 + storm_counter)

            ts_df_v5r_stoch, _ = simulate_grid_impact_timeseries_v5r_stochastic(
                storm_id=storm_id,
                ph_points=ph_points,
                line_df=line_df,
                line_samples=line_samples,
                buses=buses_nb,
                G_base=G_base,
                gen_bus_cap=gen_bus_cap,
                baseline_mw_per_person_proxy=baseline_stats["mw_per_person_proxy"],
                search_radius_km=cfg.search_radius_km,
                global_max_wind_kph=global_max_wind,
                fragility_h0=cfg.fail_threshold,
                fragility_k=cfg.stochastic_fragility_k,
                rmax_km_default=cfg.rmax_km_default,
                rmax_scaling=cfg.rmax_scaling,
                inner_core_floor=cfg.inner_core_floor,
                outer_decay_power=cfg.outer_decay_power,
                min_effective_wind_kph=cfg.min_effective_wind_kph,
                bus_pop_col="bus_pop_local_nb",
                min_relative_ratio_multi=cfg.min_relative_ratio_multi,
                min_relative_ratio_single=cfg.min_relative_ratio_single,
                min_total_gen_mw=cfg.min_total_gen_mw,
                rng=rng_storm,
            )
            storm_ts_store_v5r_stoch[storm_id] = ts_df_v5r_stoch.copy()
            storm_summaries_v5r_stoch.append(
                summarize_storm_timeseries(ts_df_v5r_stoch)
            )

        except Exception as exc:
            print(f"Skipping {storm_id}: {exc}")

        pbar.update(1)

    pbar.close()

    rank_df_v4b = pd.DataFrame(storm_summaries_v4b)
    rank_df_v5r = pd.DataFrame(storm_summaries_v5r)
    rank_df_v5r_stoch = pd.DataFrame(storm_summaries_v5r_stoch)

    _apply_screening_scores(rank_df_v4b, "grid_impact_index_v4b")
    _apply_screening_scores(rank_df_v5r, "grid_impact_index_v5r")
    _apply_screening_scores(rank_df_v5r_stoch, "grid_impact_index_v5r_stoch")

    print("Completed storms:")
    print("v4b:", len(rank_df_v4b))
    print("v5r deterministic:", len(rank_df_v5r))
    print("v5r stochastic:", len(rank_df_v5r_stoch))

    save_dataframe(cfg, rank_df_v4b, "storm_summary_v4b")
    save_dataframe(cfg, rank_df_v5r, "storm_summary_v5r")
    save_dataframe(cfg, rank_df_v5r_stoch, "storm_summary_v5r_stoch")
    save_pickle(cfg, storm_ts_store_v4b, "storm_ts_store_v4b")
    save_pickle(cfg, storm_ts_store_v5r, "storm_ts_store_v5r")
    save_pickle(cfg, storm_ts_store_v5r_stoch, "storm_ts_store_v5r_stoch")
    print("[cache] Storm catalogue summaries and time-series stores written.")

    artefact_paths: dict[str, str] = {
        n: str(cache_csv_path(cfg, n)) for n in _STORM_SUMMARY_NAMES
    }
    for n in _STORM_TS_NAMES:
        artefact_paths[n] = str(cache_pkl_path(cfg, n))

    write_manifest(
        cfg,
        "storm_sims",
        {
            "stage": "storm_sims",
            "run_id": cfg.resolved_run_id(),
            "model_tag": cfg.model_tag,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "n_storms_catalogue": int(len(storm_meta_run)),
            "artifacts": artefact_paths,
        },
    )

    return {
        "rank_df_v4b": rank_df_v4b,
        "rank_df_v5r": rank_df_v5r,
        "rank_df_v5r_stoch": rank_df_v5r_stoch,
        "storm_ts_store_v4b": storm_ts_store_v4b,
        "storm_ts_store_v5r": storm_ts_store_v5r,
        "storm_ts_store_v5r_stoch": storm_ts_store_v5r_stoch,
    }
