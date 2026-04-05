# KRITIS–Philippines (typhoon × grid screening)

## How to run it (simple version)

You only **need the notebook** for the full analysis. The command-line tool (`kritis_ph`) is **optional** — it can repeat heavy steps once your data and caches exist.

### Step 0 — Open the right folder in Cursor / VS Code

Use the project folder that contains **`pyproject.toml`** and **`notebooks/`** (this repo).

### Step 1 — Python environment (once)

In a terminal, **from the project root** (where `pyproject.toml` lives):

```bash
cd /path/to/this/repo
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

### Step 2 — WorldPop file (required for network + population prep)

The notebook expects this exact path:

`data/worldpop_ph_2020_1km/phl_pd_2020_1km_ASCII_XYZ.csv`

Download or copy your Philippines WorldPop 1 km ASCII/CSV into that folder with that filename (create the folders if needed). Without this file, the “base prep” part will error.

### Step 3 — Run the canonical notebook

1. Open **`notebooks/KRITIS_PH_canonical_notebook_commented.ipynb`**  
   (Ignore any copy of the same notebook at the repo root if you still have one — use the one under **`notebooks/`.**)
2. Select the Jupyter kernel that uses **`.venv/bin/python`** (see `Setup → Select Kernel` in VS Code / Cursor).
3. **Run all cells from top to bottom** (or “Run All”).  
   The notebook is written to load cached results when possible so you do not have to re-simulate everything every time.

That is the whole workflow for actually *doing* the study.

---

## Optional: command-line builds

Only use this if you already ran the notebook far enough to create caches, or you have placed inputs as below.

Check what is missing:

```bash
.venv/bin/python -m kritis_ph info
```

Examples:

```bash
.venv/bin/python -m kritis_ph build --stages init
.venv/bin/python -m kritis_ph build --stages base_prep    # needs WorldPop file
.venv/bin/python -m kritis_ph build --stages storm_sims   # needs notebook-derived catalogue + baseline caches
```

The CLI does **not** download IBTrACS or build `storm_meta_run` for you — those still come from the notebook unless you copy the matching CSVs into `outputs/cache/`.

---

## Layout

| Path | Purpose |
|------|---------|
| `src/kritis_ph/` | Python package + `python -m kritis_ph` CLI |
| `notebooks/` | Main analysis notebook |
| `PyPSA-PH/` | Benchmark grid data |
| `data/` | Inputs (WorldPop, EM-DAT, etc.) |
| `outputs/cache/` | Cached tables (regenerable) |
| `outputs/runs/` | Small manifest files per run |
| `outputs/figures/` | Exported figures |
