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

Raw HEIS chemistry TXT files are listed directly in `config.toml`. Chemistry preparation runs automatically as the first part of Step 01, with the mode controlled by `chem_prep_mode`:

```toml
[prep_chemistry]
raw_chemistry_files = [
  "input/00_Data/Chemistry_Data/CY24/file1.txt",
  "input/00_Data/Chemistry_Data/CY24/file2.txt",
]
chem_prep_mode = "chromium"   # or "single"
```

**`"chromium"` mode** — combines filtered total Chromium and Hexavalent Chromium into a single analyte stream. Use for chromium plume monitoring.

**`"single"` mode** — retains one analyte by name. Use for analytes with a single constituent of concern (e.g. Nitrate):

```toml
chem_prep_mode = "single"
analyte = "Nitrate"
# filtered_keep_value = "Y"    # optional FILTERED column restriction
```

The prepared dataset is written to `<output_dir>/prepared_chemistry_<run_id>.parquet` for diagnostic inspection.

## Outputs

Outputs are written to the configured output directory and run-version subfolder.

Typical outputs include:

```text
prepared_chemistry_<run_id>.parquet
Chem_TrendData_<run_id>.parquet
TTA_full_term_stats_<run_id>.csv
TTA_Results_<run_id>.csv
TobitRegression_WLlag_<OU>_<run_id>.pdf
tta.log
```

`tta.log` contains progress messages, warnings, errors, and detailed debug information from the run.


## Code walkthrough

### `run_tta.py`

Main workflow entry point. It:

1. reads the TOML configuration;
2. creates the output directory;
3. applies global well filters and optional `selected_wells`;
4. runs each workflow step in sequence;
5. writes outputs;
6. launches reporting;
7. writes progress and errors to `tta.log`.

### Step 00 — Distance calculations

Calculates well distance to the river and distance to river-stage gauges. These results are used later to assign river-stage covariates and populate report metadata.

### Step 01 — Chemistry preprocessing

Reads raw HEIS TXT files and prepares the chemistry dataset for modelling. The step runs in two phases:

1. **Raw data preparation** — analyte selection and filtering via `prepare_chromium_chemistry` or `prepare_chemistry_data` depending on `chem_prep_mode`. The prepared dataset is written to `output_dir` as a parquet.
2. **Chemistry import** — non-detect handling, REVIEWQ and collection-purpose filtering, daily averaging, and joining of river-stage, well metadata, and screen interval data.

### Step 02 — Water-level preprocessing

Imports and processes water-level data. It joins river-stage data, well metadata, and screen intervals, creating the water-level dataset used for lag and trend analysis.

### Step 03 — Water-level trend analysis

Estimates the relationship between groundwater elevation, river stage and time. It calculates lag, number of observations, and p-values for the trend, river-stage term and date term.

### Step 04 — Chemistry Tobit preparation and modelling

Prepares chemistry data for censored regression by applying trend breaks, assigning trend terms, handling no-river-stage wells and fitting Tobit models.

### Step 05 — Reporting

Generates OU-level PDF reports.