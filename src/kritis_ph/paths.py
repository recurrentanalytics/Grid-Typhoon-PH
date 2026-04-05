"""Repository-relative paths. Prefer this over Path.cwd() so CLI and notebooks agree."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Project root (contains pyproject.toml, PyPSA-PH/, notebooks/)."""
    return Path(__file__).resolve().parents[2]


def pypsa_ph_dir(root: Path | None = None) -> Path:
    r = root or repo_root()
    return r / "PyPSA-PH"


def data_dir(root: Path | None = None) -> Path:
    r = root or repo_root()
    return r / "data"


def outputs_dir(root: Path | None = None) -> Path:
    r = root or repo_root()
    return r / "outputs"


def cache_dir(root: Path | None = None) -> Path:
    """Legacy notebook cache: outputs/cache/ with MODEL_TAG in filenames."""
    return outputs_dir(root) / "cache"


def runs_dir(root: Path | None = None) -> Path:
    """Versioned pipeline artefacts: outputs/runs/<run_id>/."""
    return outputs_dir(root) / "runs"


def run_dir(run_id: str, root: Path | None = None) -> Path:
    safe = str(run_id).replace("/", "_").replace(" ", "_")
    return runs_dir(root) / safe


def buses_csv(root: Path | None = None) -> Path:
    return pypsa_ph_dir(root) / "data" / "buses.csv"


def lines_csv(root: Path | None = None) -> Path:
    return pypsa_ph_dir(root) / "data" / "lines.csv"


def generators_csv(root: Path | None = None) -> Path:
    return pypsa_ph_dir(root) / "data" / "generators.csv"


def worldpop_csv(root: Path | None = None) -> Path:
    return (
        data_dir(root)
        / "worldpop_ph_2020_1km"
        / "phl_pd_2020_1km_ASCII_XYZ.csv"
    )
