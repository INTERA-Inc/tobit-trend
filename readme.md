# Tobit Trend Analysis Tool

[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python workflow for groundwater chemistry and water-level trend analysis. 

## Repository layout

```text
project-root/
├── pyproject.toml
├── README.md
├── environment.yml
├── configs/
│   └── config.toml
├── input/
│   └── ...
└── src/
    └── tta/
        ├── run_tta.py
        ├── config.py
        ├── utils.py
        ├── preprocessing/
        │   ├── prepare_chromium_chemistry.py
        │   ├── prepare_chemistry_data.py
        │   ├── calculate_distance_00.py
        │   ├── chemistry_import_01.py
        │   ├── water_level_import_02.py
        │   ├── water_level_trends_03.py
        │   └── tobit_prep_04.py
        ├── model/
        │   └── tobit_model_04.py
        └── reporting/
            └── generate_report_05.py
```


## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate tta
```

Install the package from the project root:

```bash
pip install -e .
```

## Configuration

Edit the TOML file before running:

```text
configs/config.toml
```

## Running the workflow

Run from the project root:

```bash
tta configs/config.toml
```

Alternative:

```bash
python -m tta.run_tta configs/config.toml
```

Paths in the TOML are expected to be relative to the project root.

## Chemistry preparation

Raw HEIS chemistry TXT files are listed directly in `config.toml`. Chemistry preparation runs automatically inside the workflow, with the mode controlled by `chem_prep_mode`:

```toml
[prep_chemistry]
raw_chemistry_files = [
  "input/00_Data/Chemistry_Data/CY24/file1.txt",
  "input/00_Data/Chemistry_Data/CY24/file2.txt",
]
chem_prep_mode = "chromium"   # "chromium" | "single"
```

**`"chromium"` mode** — combines filtered total Chromium and Hexavalent Chromium into a single analyte stream. The full workflow runs once.

**`"single"` mode** — one or more analytes supplied as a list. The workflow runs end-to-end for each analyte in turn, producing a separate set of outputs per analyte:

```toml
chem_prep_mode = "single"
analyte = ["Nitrate"]                 # single analyte
analyte = ["Nitrate", "Uranium"]      # multiple analytes — workflow loops over each
# filtered_keep_value = "Y"          # optional FILTERED column restriction
```

## Outputs

Outputs are written to the configured output directory and run-version subfolder. Analyte-specific files include the analyte name as a slug (e.g. `Nitrate`, `Hex_Filt_Chromium`).

```text
prepared_chemistry_<analyte>_<run_id>.parquet
Chem_TrendData_<analyte>_<run_id>.parquet
TTA_full_term_stats_<analyte>_<run_id>.csv
TTA_Results_<analyte>_<run_id>.csv
TobitRegression_WLlag_<OU>_<analyte>_<run_id>.pdf
WL_regression_<run_id>.pdf               ← water-level only, produced once
tta.log
```

`tta.log` contains progress messages, warnings, errors, and detailed debug information from the run.

## Code walkthrough

### `run_tta.py`

Main workflow entry point. It:

1. reads the TOML configuration;
2. creates the output directory;
3. applies global well filters and optional `selected_wells`;
4. runs Steps 00–03 once (analyte-independent);
5. loops over each configured analyte, running Steps 01 and 04–05 per analyte;
6. writes all outputs with the analyte name in the filename;
7. writes progress and errors to `tta.log`.

### Step 00 — Distance calculations

Calculates well distance to the river and distance to river-stage gauges. Runs once — results are shared across all analytes.

### Steps 02–03 — Water-level preprocessing and trend analysis

Imports and processes water-level data (Step 02), then estimates groundwater-level trends and river-stage lag times (Step 03). Both steps run once and are shared across all analytes.

### Step 01 — Chemistry preprocessing *(per analyte)*

Reads raw HEIS TXT files and prepares the chemistry dataset for one analyte. Runs in two phases:

1. **Raw data preparation** — analyte selection and filtering via `prepare_chromium_chemistry` or `prepare_chemistry_data` depending on `chem_prep_mode`. The prepared dataset is written to `output_dir` as a parquet.
2. **Chemistry import** — non-detect handling, REVIEWQ and collection-purpose filtering, daily averaging, and joining of river-stage, well metadata, and screen interval data.

### Step 04 — Chemistry Tobit preparation and modelling *(per analyte)*

Prepares chemistry data for censored regression by applying trend breaks, assigning trend terms, and handling no-river-stage wells. Fits Tobit models and writes `TTA_full_term_stats_<analyte>_<run_id>.csv` and `TTA_Results_<analyte>_<run_id>.csv`.

### Step 05 — Reporting *(per analyte)*

Generates OU-level PDF reports. The analyte name is used in plot axis labels (read directly from the chemistry data) and in the output PDF filename.