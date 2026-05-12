from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


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
