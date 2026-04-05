"""CLI: `python -m kritis_ph` or `kritis-ph` after editable install."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kritis_ph.config import RunConfig
from kritis_ph import paths as P
from kritis_ph.io import cache_csv_path, cache_pkl_path
from kritis_ph.stages import ALL_STAGES, STAGE_RUNNERS
from kritis_ph.stages import base_prep, storm_sims


def _parse_stages(spec: str) -> tuple[str, ...]:
    s = spec.strip().lower()
    if s == "all":
        return ALL_STAGES
    parts = tuple(p.strip() for p in spec.split(",") if p.strip())
    unknown = set(parts) - set(STAGE_RUNNERS)
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}")
    return parts


def _make_config(args: argparse.Namespace) -> RunConfig:
    root = Path(args.repo_root).resolve() if args.repo_root else P.repo_root()
    return RunConfig(
        repo_root=root,
        model_tag=args.model_tag,
        run_id=args.run_id,
    )


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _make_config(args)
    r = cfg.repo_root
    print("repo_root:", r)
    print("resolved_run_id:", cfg.resolved_run_id())
    print("cache_dir:", cfg.cache_dir())
    print("run_artifact_dir:", cfg.run_artifact_dir())
    checks = [
        ("buses.csv", P.buses_csv(r)),
        ("lines.csv", P.lines_csv(r)),
        ("generators.csv", P.generators_csv(r)),
        ("worldpop csv", P.worldpop_csv(r)),
    ]
    for label, p in checks:
        print(f"  {label}: {p}  [{'ok' if p.is_file() else 'MISSING'}]")

    print("\nPipeline prerequisites (this MODEL_TAG):")
    wp = P.worldpop_csv(r)
    print(f"  base_prep:     worldpop file present  [{'ok' if wp.is_file() else 'MISSING'}]")
    print(
        f"  base_prep:     cache complete         [{'ok' if base_prep.artifacts_ready(cfg) else 'not yet'}]"
    )
    gen_p = cache_csv_path(cfg, "gen_bus_cap")
    base_p = cache_pkl_path(cfg, "baseline_stats")
    print(
        f"  storm_sims:    gen_bus_cap cache      [{'ok' if gen_p.is_file() else 'MISSING — run baseline cell in notebook'}]"
    )
    print(
        f"  storm_sims:    baseline_stats cache   [{'ok' if base_p.is_file() else 'MISSING — run baseline cell in notebook'}]"
    )
    ph = cache_csv_path(cfg, "ph_points")
    sm = cache_csv_path(cfg, "storm_meta_run")
    print(
        f"  storm_sims:    ph_points cache        [{'ok' if ph.is_file() else 'MISSING — run notebook through storm_meta_run cell'}]"
    )
    print(
        f"  storm_sims:    storm_meta_run cache   [{'ok' if sm.is_file() else 'MISSING — same as above'}]"
    )
    print(
        f"  storm_sims:    storm summaries cache  [{'ok' if storm_sims.artifacts_ready(cfg) else 'not yet'}]"
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    cfg = _make_config(args)
    try:
        stages = _parse_stages(args.stages)
    except ValueError as e:
        print(f"kritis-ph build: {e}", file=sys.stderr)
        return 2
    print("repo_root:", cfg.repo_root)
    print("stages:", ", ".join(stages))
    for name in stages:
        print(f"[build] -> {name}")
        try:
            STAGE_RUNNERS[name](cfg)
        except FileNotFoundError as e:
            print(f"[build] stage {name!r} failed: {e}", file=sys.stderr)
            print(
                "Hint: run `python -m kritis_ph info` for a prerequisite checklist.",
                file=sys.stderr,
            )
            return 4
        except NotImplementedError as e:
            print(f"[build] stage {name!r} not implemented: {e}", file=sys.stderr)
            return 3
    print("[build] done.")
    return 0


def cmd_list_stages(_args: argparse.Namespace) -> int:
    for s in ALL_STAGES:
        print(s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kritis-ph")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show paths and whether core inputs exist")
    p_info.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repo root (default: directory containing kritis_ph package)",
    )
    p_info.add_argument(
        "--model-tag",
        default="v5r_det_threshold055",
        help="MODEL_TAG (cache filename stem)",
    )
    p_info.add_argument(
        "--run-id",
        default=None,
        help="Artefact run id (default: same as --model-tag)",
    )
    p_info.set_defaults(func=cmd_info)

    p_build = sub.add_parser("build", help="Run pipeline stages")
    p_build.add_argument("--repo-root", type=Path, default=None)
    p_build.add_argument(
        "--model-tag",
        default="v5r_det_threshold055",
    )
    p_build.add_argument("--run-id", default=None)
    p_build.add_argument(
        "--stages",
        default="init",
        help="Comma-separated stage names, or 'all' (default: init)",
    )
    p_build.set_defaults(func=cmd_build)

    p_ls = sub.add_parser("list-stages", help="Print registered stage names")
    p_ls.set_defaults(func=cmd_list_stages)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
