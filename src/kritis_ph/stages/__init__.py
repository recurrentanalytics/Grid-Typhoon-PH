"""Pipeline stages; heavy logic is lifted here incrementally from the canonical notebook."""

from __future__ import annotations

from collections.abc import Callable

from kritis_ph.config import RunConfig
from kritis_ph.stages import (
    base_prep,
    hotspot_stats,
    init_layout,
    storm_sims,
    top_event_exports,
)

StageFn = Callable[[RunConfig], None]

STAGE_RUNNERS: dict[str, StageFn] = {
    "init": init_layout.run,
    "base_prep": base_prep.run,
    "storm_sims": storm_sims.run,
    "hotspot_stats": hotspot_stats.run,
    "top_event_exports": top_event_exports.run,
}

ALL_STAGES = (
    "init",
    "base_prep",
    "storm_sims",
    "hotspot_stats",
    "top_event_exports",
)
