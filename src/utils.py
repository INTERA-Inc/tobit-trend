from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import logging
import sys
from collections.abc import Sequence
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run groundwater Tobit trend analysis workflow."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to TOML configuration file.",
    )
    return parser.parse_args()


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pandas()
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input format: {path}")


def build_output_dir(output_dir: str | Path | None, run_ver: str | None) -> Path | None:
    if output_dir is None:
        return None

    out_dir = Path(output_dir)
    if run_ver is not None:
        out_dir = out_dir / str(run_ver)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_selected_wells(selected_wells: list[str] | None) -> list[str]:
    if selected_wells is None:
        return []
    if not isinstance(selected_wells, list):
        raise TypeError("selected_wells must be a list of strings.")
    if not all(isinstance(w, str) for w in selected_wells):
        raise TypeError("selected_wells must be a list of strings.")

    return [w.strip() for w in selected_wells if w.strip()]


def filter_by_selected_wells(
    df: pd.DataFrame,
    selected_wells: list[str],
    col: str = "NAME",
) -> pd.DataFrame:
    if not selected_wells:
        return df

    if col not in df.columns:
        raise KeyError(f"Cannot filter selected_wells: missing column {col!r}")

    return df.loc[df[col].astype(str).str.strip().isin(selected_wells)].copy()


def assert_only_selected_wells(
    df: pd.DataFrame,
    selected_wells: list[str],
    col: str = "NAME",
    label: str = "dataframe",
) -> None:
    if not selected_wells:
        return

    if col not in df.columns:
        raise KeyError(f"{label} has no column {col!r}")

    found = set(df[col].dropna().astype(str).str.strip())
    extra = sorted(found - set(selected_wells))

    if extra:
        raise RuntimeError(f"{label} contains wells outside selected_wells: {extra}")


def assert_not_empty(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise RuntimeError(f"{label} is empty.")


def setup_logger(output_dir: Path) -> logging.Logger:
    """
    Set up workflow logger.

    Writes full log to:
        output_dir / "tta.log"

    Also prints progress messages to console.
    """
    log_file = output_dir / "tta.log"

    logger = logging.getLogger("tta")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # prevents duplicate messages if rerun in same session

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def apply_well_filters(
    well: pd.DataFrame,
    filter_cols: Sequence[str],
    filter_modes: Sequence[str],
    filter_values: Sequence[Sequence[str]],
) -> pd.DataFrame:
    if not filter_cols:
        return well.copy()

    if not (len(filter_cols) == len(filter_modes) == len(filter_values)):
        raise ValueError(
            "well_filter_cols, well_filter_modes, and well_filter_values "
            "must have the same length."
        )

    out = well.copy()

    for col, mode, values in zip(filter_cols, filter_modes, filter_values):
        if col not in out.columns:
            raise KeyError(f"Well filter column {col!r} not found in well table.")

        mode = str(mode).strip().lower()
        values_norm = {str(v).strip() for v in values}
        col_values = out[col].astype(str).str.strip()

        if mode == "include":
            out = out.loc[col_values.isin(values_norm)].copy()
        elif mode == "exclude":
            out = out.loc[~col_values.isin(values_norm)].copy()
        else:
            raise ValueError(
                f"Invalid well filter mode {mode!r}. Use 'include' or 'exclude'."
            )

    return out
