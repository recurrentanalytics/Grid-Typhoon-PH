"""Artefact paths and small manifest helpers (aligned with notebook cache naming)."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from kritis_ph.config import RunConfig


def cache_stem(name: str, model_tag: str) -> str:
    safe_tag = str(model_tag).replace("/", "_").replace(" ", "_")
    return f"{name}__{safe_tag}"


def cache_csv_path(cfg: RunConfig, name: str) -> Path:
    stem = cache_stem(name, cfg.model_tag)
    return cfg.cache_dir() / f"{stem}.csv"


def cache_pkl_path(cfg: RunConfig, name: str) -> Path:
    stem = cache_stem(name, cfg.model_tag)
    return cfg.cache_dir() / f"{stem}.pkl"


def manifest_path(cfg: RunConfig, stage: str) -> Path:
    """Manifest under versioned run dir, e.g. .manifest_base_prep.json."""
    safe = stage.replace(" ", "_")
    return cfg.run_artifact_dir() / f".manifest_{safe}.json"


def write_manifest(cfg: RunConfig, stage: str, payload: dict[str, Any]) -> Path:
    """Atomically write JSON manifest after a stage completes."""
    path = manifest_path(cfg, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=".manifest_", suffix=".tmp.json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp and Path(tmp).is_file():
            Path(tmp).unlink(missing_ok=True)
    return path


def read_manifest(cfg: RunConfig, stage: str) -> dict[str, Any] | None:
    path = manifest_path(cfg, stage)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataframe(
    cfg: RunConfig, df: pd.DataFrame, name: str, **to_csv_kwargs: Any
) -> Path:
    path = cache_csv_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **to_csv_kwargs)
    print(f"[cache] Saved DataFrame {name!r} -> {path} ({len(df)} rows)")
    return path


def load_dataframe(cfg: RunConfig, name: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    path = cache_csv_path(cfg, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache CSV not found for {name!r}: {path}\n"
            "Run: python -m kritis_ph build --stages base_prep"
        )
    out = pd.read_csv(path, **read_csv_kwargs)
    print(f"[cache] Loaded DataFrame {name!r} <- {path} ({len(out)} rows)")
    return out


def save_pickle(cfg: RunConfig, obj: Any, name: str) -> Path:
    path = cache_pkl_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] Saved object {name!r} -> {path}")
    return path


def load_pickle(cfg: RunConfig, name: str) -> Any:
    path = cache_pkl_path(cfg, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache pickle not found for {name!r}: {path}\n"
            "Run: python -m kritis_ph build --stages base_prep"
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[cache] Loaded object {name!r} <- {path}")
    return obj


def load_pickle_optional(cfg: RunConfig, name: str) -> Any | None:
    path = cache_pkl_path(cfg, name)
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
