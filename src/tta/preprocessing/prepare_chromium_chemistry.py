"""
prepare_chromium_chemistry.py

Prepare chromium chemistry data outside the main groundwater trend workflow.

Purpose
-------
This script extracts the current chromium-specific logic from Script 01 so that
the main trend workflow can become analyte-agnostic.

It reads raw HEIS chemistry TXT/CSV files, keeps:
    - filtered total Chromium records, and
    - Hexavalent Chromium records,

then standardises them into one combined analyte stream and writes:

    prepared_chemistry_<YEAR>.parquet

The output is intended to be used as the chemistry input to the main workflow.

Example
-------
Run from the project root:

    python src/preprocessing/prepare_chromium_chemistry.py \
        --chem-files input/00_Data/Chemistry_Data/CY24/qry_GW_REPORT_CY2024_GWSR_1.txt \
                     input/00_Data/Chemistry_Data/CY24/qry_GW_REPORT_CY2024_GWSR_2.txt \
                     input/00_Data/Chemistry_Data/CY24/qry_GW_REPORT_CY2024_GWSR_3.txt \
        --year 2024 \
        --output input/prepared_chemistry/prepared_chemistry_2024.parquet

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_CHROMIUM_ANALYTE = "Chromium"
DEFAULT_HEXCHROM_ANALYTE = "Hexavalent Chromium"
DEFAULT_FILTERED_KEEP_VALUE = "Y"
DEFAULT_COMBINED_ANALYTE_NAME = "Hex. Chromium and Fil. Chromium"
DEFAULT_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def read_chem_heis(
    path: str | Path,
    *,
    date_format: str = DEFAULT_DATE_FORMAT,
    sep: str = ",",
) -> pd.DataFrame:
    """
    Read one raw HEIS chemistry export.

    This mirrors the raw import cleanup previously embedded in Script 01:
    - removes unit rows where DEPTH == 'm';
    - removes rows without VAL;
    - parses EVENT and LOAD_DATE_TIME;
    - uses STD_MDA as MDL where available;
    - removes exact duplicate rows.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Chemistry file not found: {path}")

    df = pd.read_csv(path, sep=sep, low_memory=False)

    if "DEPTH" in df.columns:
        df = df.loc[df["DEPTH"] != "m"].copy()

    if "VAL" not in df.columns:
        raise KeyError(f"{path} is missing required column 'VAL'.")

    df = df.loc[~df["VAL"].isna()].copy()

    for col in ["EVENT", "LOAD_DATE_TIME"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace("EDT ", "", regex=False)
            s = s.str.replace("EST ", "", regex=False)
            df[col] = pd.to_datetime(s, format=date_format, errors="coerce")

    if "STD_MDA" in df.columns and "MDL" in df.columns:
        df["MDL"] = np.where(df["STD_MDA"].notna(), df["STD_MDA"], df["MDL"])

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate columns needed for chromium preparation.
    """
    required = {
        "NAME",
        "EVENT",
        "ANALYTE",
        "FILTERED",
        "VAL",
        "MDL",
    }

    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Chemistry data missing required columns: {sorted(missing)}")


def prepare_chromium_chemistry(
    chem_files: Sequence[str | Path],
    *,
    year: int,
    chromium_analyte: str = DEFAULT_CHROMIUM_ANALYTE,
    hexchrom_analyte: str = DEFAULT_HEXCHROM_ANALYTE,
    filtered_keep_value: str = DEFAULT_FILTERED_KEEP_VALUE,
    combined_analyte_name: str = DEFAULT_COMBINED_ANALYTE_NAME,
    date_format: str = DEFAULT_DATE_FORMAT,
    sep: str = ",",
) -> pd.DataFrame:
    """
    Prepare a single combined chromium chemistry dataset.

    Logic:
    1. Read all raw chemistry files.
    2. Keep records up to the requested analysis year.
    3. Keep:
       - total Chromium where FILTERED == filtered_keep_value;
       - Hexavalent Chromium regardless of FILTERED.
    4. Preserve original analyte in ANALYTE_ORG.
    5. Rename ANALYTE to the combined analyte name.
    """
    if not chem_files:
        raise ValueError("No chemistry files supplied.")

    chem_parts = [
        read_chem_heis(path, date_format=date_format, sep=sep) for path in chem_files
    ]

    chem = pd.concat(chem_parts, ignore_index=True)
    validate_required_columns(chem)

    chem["EVENT"] = pd.to_datetime(chem["EVENT"], errors="coerce")
    chem["VAL"] = pd.to_numeric(chem["VAL"], errors="coerce")
    chem["MDL"] = pd.to_numeric(chem["MDL"], errors="coerce")

    chem = chem.loc[chem["EVENT"].dt.year <= int(year)].copy()

    chrom = pd.concat(
        [
            chem.loc[
                (chem["ANALYTE"] == chromium_analyte)
                & (chem["FILTERED"] == filtered_keep_value)
            ].copy(),
            chem.loc[chem["ANALYTE"] == hexchrom_analyte].copy(),
        ],
        ignore_index=True,
    )

    if chrom.empty:
        raise ValueError(
            "Chromium preparation produced no records. "
            "Check analyte names, FILTERED values, and year cutoff."
        )
    chrom["ANALYTE_ORG"] = chrom["ANALYTE"]
    chrom["ANALYTE"] = combined_analyte_name

    chrom["DUPLICATE_PRIORITY"] = np.where(
        chrom["ANALYTE_ORG"] == hexchrom_analyte,
        1,
        0,
    )

    return chrom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare combined chromium chemistry data for the trend workflow."
    )

    parser.add_argument(
        "--chem-files",
        nargs="+",
        required=True,
        type=Path,
        help="Raw HEIS chemistry TXT/CSV files.",
    )

    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Analysis year cutoff. Records after this year are removed.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output parquet path, e.g. input/prepared_chemistry/prepared_chemistry_2024.parquet.",
    )

    parser.add_argument(
        "--chromium-analyte",
        default=DEFAULT_CHROMIUM_ANALYTE,
        help="Name of total chromium analyte in raw chemistry files.",
    )

    parser.add_argument(
        "--hexchrom-analyte",
        default=DEFAULT_HEXCHROM_ANALYTE,
        help="Name of hexavalent chromium analyte in raw chemistry files.",
    )

    parser.add_argument(
        "--filtered-keep-value",
        default=DEFAULT_FILTERED_KEEP_VALUE,
        help="FILTERED value to retain for total chromium.",
    )

    parser.add_argument(
        "--combined-analyte-name",
        default=DEFAULT_COMBINED_ANALYTE_NAME,
        help="Analyte name assigned to the prepared combined chemistry data.",
    )

    parser.add_argument(
        "--date-format",
        default=DEFAULT_DATE_FORMAT,
        help="Datetime format for EVENT and LOAD_DATE_TIME.",
    )

    parser.add_argument(
        "--sep",
        default=",",
        help="Input file delimiter. Default is comma.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepared = prepare_chromium_chemistry(
        chem_files=args.chem_files,
        year=args.year,
        chromium_analyte=args.chromium_analyte,
        hexchrom_analyte=args.hexchrom_analyte,
        filtered_keep_value=args.filtered_keep_value,
        combined_analyte_name=args.combined_analyte_name,
        date_format=args.date_format,
        sep=args.sep,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)

    print(f"Wrote prepared chemistry: {args.output}")
    print(f"Rows: {len(prepared):,}")
    print(f"Wells: {prepared['NAME'].nunique():,}")
    print(f"Analyte: {prepared['ANALYTE'].iloc[0]}")


if __name__ == "__main__":
    main()
