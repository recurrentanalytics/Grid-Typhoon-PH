"""Single place for run identity and model defaults (mirror notebook Cell 1 + flags cell)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kritis_ph import paths as P


@dataclass(frozen=True)
class PhilippineBounds:
    """Geographic crop for WorldPop rows (notebook Cell 1 PH_BOUNDS)."""

    lat_min: float = 4.0
    lat_max: float = 22.0
    lon_min: float = 116.0
    lon_max: float = 128.0


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one pipeline run / notebook session."""

    repo_root: Path = field(default_factory=P.repo_root)
    model_tag: str = "v5r_det_threshold055"
    run_id: str | None = None

    start_year: int | None = None
    search_radius_km: float = 200.0
    fail_threshold: float = 0.55
    line_sample_points: int = 7
    random_seed: int = 42
    ph_bounds: PhilippineBounds = field(default_factory=PhilippineBounds)

    use_generator_commission_year: bool = False
    generator_year_column: str = "commissioning_year"
    gen_capacity_col: str = "p_nom"

    rmax_km_default: float = 35.0
    rmax_scaling: bool = False
    inner_core_floor: float = 0.85
    outer_decay_power: float = 1.35
    min_effective_wind_kph: float = 30.0

    min_relative_ratio_multi: float = 0.35
    min_relative_ratio_single: float = 0.75
    min_total_gen_mw: float = 1.0
    stochastic_fragility_k: float = 18.0

    def resolved_run_id(self) -> str:
        return self.run_id or self.model_tag

    def cache_dir(self) -> Path:
        d = P.cache_dir(self.repo_root)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run_artifact_dir(self) -> Path:
        d = P.run_dir(self.resolved_run_id(), self.repo_root)
        d.mkdir(parents=True, exist_ok=True)
        return d
