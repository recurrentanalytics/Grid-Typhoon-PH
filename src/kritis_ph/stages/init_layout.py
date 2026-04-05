"""Create output directories and a bootstrap manifest for the run."""

from __future__ import annotations

from datetime import datetime, timezone

from kritis_ph.config import RunConfig
from kritis_ph.io import write_manifest


def run(cfg: RunConfig) -> None:
    cfg.cache_dir()
    cfg.run_artifact_dir()
    write_manifest(
        cfg,
        "init",
        {
            "stage": "init",
            "run_id": cfg.resolved_run_id(),
            "model_tag": cfg.model_tag,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
