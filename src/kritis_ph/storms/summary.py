"""Per-storm summary rows for ranking (notebook Cell 9 excerpt)."""

from __future__ import annotations

import pandas as pd


def summarize_storm_timeseries(ts_df: pd.DataFrame) -> dict[str, object]:
    """Collapse a full storm time series to one summary row."""
    if ts_df.empty:
        raise ValueError("ts_df is empty")

    peak_idx = ts_df["affected_population_proxy"].idxmax()
    peak_row = ts_df.loc[peak_idx].copy()

    return {
        "storm_id": peak_row["storm_id"],
        "storm_name": peak_row["storm_name"],
        "year": int(peak_row["year"]),
        "peak_t": int(peak_row["t"]),
        "peak_affected_population_proxy": float(ts_df["affected_population_proxy"].max()),
        "peak_affected_population_share_proxy": float(
            ts_df["affected_population_share_proxy"].max()
        ),
        "peak_n_unserved_buses": int(ts_df["n_unserved_buses"].max()),
        "peak_unserved_bus_share": float(ts_df["unserved_bus_share"].max()),
        "peak_n_failed_lines_cumulative": int(ts_df["n_failed_lines_cumulative"].max()),
        "peak_failed_line_share_cumulative": float(
            ts_df["failed_line_share_cumulative"].max()
        ),
        "final_affected_population_proxy": float(ts_df.iloc[-1]["affected_population_proxy"]),
        "final_n_unserved_buses": int(ts_df.iloc[-1]["n_unserved_buses"]),
        "final_n_failed_lines_cumulative": int(ts_df.iloc[-1]["n_failed_lines_cumulative"]),
        "n_timesteps": int(len(ts_df)),
    }
