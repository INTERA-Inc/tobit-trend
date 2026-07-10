"""
Prepare single-analyte chemistry data for the TTA workflow.

Use this for analytes with one CoC (e.g. Nitrate).
For the dual-CoC chromium case (total Chromium + Hexavalent Chromium),
use prepare_chromium_chemistry.py instead.

Example
-------
Run from the project root.

cmd.exe (use ^ for line continuation):

    python src/tta/preprocessing/prepare_chemistry_data.py ^
        --chem-files input/00_Data/Chemistry_Data/CY24/qry_GW_REPORT_CY2024_GWSR_1.txt ^
        --analyte "Nitrate" ^
        --year 2024 ^
        --output input/prepared_chemistry/prepared_chemistry_nitrate_2024.parquet

PowerShell (use backtick for line continuation):

    python src/tta/preprocessing/prepare_chemistry_data.py `
        --chem-files input/00_Data/Chemistry_Data/CY24/qry_GW_REPORT_CY2024_GWSR_1.txt `
        --analyte "Nitrate" `
        --year 2024 `
        --output input/prepared_chemistry/prepared_chemistry_nitrate_2024.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from tta.preprocessing.prepare_chromium_chemistry import (
    DEFAULT_DATE_FORMAT,
    read_chem_heis,
    validate_required_columns,
)


def prepare_chemistry_data(
    chem_files: Sequence[str | Path],
    *,
    year: int,
    analyte: str,
    combined_analyte_name: Optional[str] = None,
    filtered_keep_value: Optional[str] = None,
    date_format: str = DEFAULT_DATE_FORMAT,
    sep: str = ",",
) -> pd.DataFrame:
    """
    Prepare a single-analyte chemistry dataset for the trend workflow.

    Parameters
    ----------
    chem_files : sequence of paths
        Raw HEIS chemistry TXT/CSV files.
    year : int
        Analysis year cutoff; records after this year are removed.
    analyte : str
        Analyte name to retain (e.g. "Nitrate").
    combined_analyte_name : str, optional
        Output ANALYTE label; defaults to ``analyte`` (no rename).
    filtered_keep_value : str, optional
        If set, only rows where FILTERED == this value are kept.
        If None, no FILTERED filter is applied.
    date_format : str
        Datetime format string for EVENT and LOAD_DATE_TIME columns.
    sep : str
        Input file delimiter.

    Returns
    -------
    pd.DataFrame
        Prepared chemistry DataFrame with ANALYTE_ORG preserved and
        ANALYTE set to ``combined_analyte_name``.
    """
    if not chem_files:
        raise ValueError("No chemistry files supplied.")

    if combined_analyte_name is None:
        combined_analyte_name = analyte

    chem_parts = [
        read_chem_heis(path, date_format=date_format, sep=sep) for path in chem_files
    ]
    chem = pd.concat(chem_parts, ignore_index=True)
    validate_required_columns(chem)

    chem["EVENT"] = pd.to_datetime(chem["EVENT"], errors="coerce")
    chem["VAL"] = pd.to_numeric(chem["VAL"], errors="coerce")
    chem["MDL"] = pd.to_numeric(chem["MDL"], errors="coerce")

    chem = chem.loc[chem["EVENT"].dt.year <= int(year)].copy()

    subset = chem.loc[chem["ANALYTE"] == analyte].copy()

    if filtered_keep_value is not None:
        subset = subset.loc[subset["FILTERED"] == filtered_keep_value].copy()

    if subset.empty:
        raise ValueError(
            f"Chemistry preparation produced no records for analyte '{analyte}'. "
            "Check analyte name, FILTERED value, and year cutoff."
        )

    subset["ANALYTE_ORG"] = subset["ANALYTE"]
    subset["ANALYTE"] = combined_analyte_name

    return subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare single-analyte chemistry data for the trend workflow."
    )
    parser.add_argument(
        "--chem-files",
        nargs="+",
        required=True,
        type=Path,
        help="Raw HEIS chemistry TXT/CSV files.",
    )
    parser.add_argument(
        "--analyte",
        required=True,
        help="Analyte name to retain (e.g. 'Nitrate').",
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
        help="Output parquet path.",
    )
    parser.add_argument(
        "--combined-analyte-name",
        default=None,
        help="Output ANALYTE label. Defaults to --analyte value.",
    )
    parser.add_argument(
        "--filtered-keep-value",
        default=None,
        help="If set, only rows with FILTERED == this value are kept.",
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

    prepared = prepare_chemistry_data(
        chem_files=args.chem_files,
        year=args.year,
        analyte=args.analyte,
        combined_analyte_name=args.combined_analyte_name,
        filtered_keep_value=args.filtered_keep_value,
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
