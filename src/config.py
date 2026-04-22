from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class TrendConfig:
    """
    Container for user-editable settings loaded from the project TOML file.

    Holds:
    - input and output file paths
    - preprocessing rules
    - model parameters
    """

    # paths
    cr_trend_parquet: Path
    wl_trends_flat_csv: Path
    wl_data_parquet: Path
    system_wells_csv: Path
    trend_breaks_csv: Path
    no_rs_csv: Path
    kw_csv: Path
    rum_csv: Path
    r_script_path: Path
    output_csv: Path

    # model
    dep: str
    indep: list[str]
    log: str
    maxlag: int
    n_min: int
    pnd_max: float
    mindate: str

    # prep rules
    PRIOR_YEAR: int
    CUTOFFS: dict[str, str]
    KW_DATE1: str
    KW_DATE2: str

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrendConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        return cls(
            # paths
            cr_trend_parquet=Path(raw["paths"]["cr_trend_parquet"]),
            wl_trends_flat_csv=Path(raw["paths"]["wl_trends_flat_csv"]),
            wl_data_parquet=Path(raw["paths"]["wl_data_parquet"]),
            system_wells_csv=Path(raw["paths"]["system_wells_csv"]),
            trend_breaks_csv=Path(raw["paths"]["trend_breaks_csv"]),
            no_rs_csv=Path(raw["paths"]["no_rs_csv"]),
            kw_csv=Path(raw["paths"]["kw_csv"]),
            rum_csv=Path(raw["paths"]["rum_csv"]),
            r_script_path=Path(raw["paths"]["r_script_path"]),
            output_csv=Path(raw["paths"]["output_csv"]),
            # model
            dep=raw["model"]["dep"],
            indep=list(raw["model"]["indep"]),
            log=raw["model"]["log"],
            maxlag=int(raw["model"]["maxlag"]),
            n_min=int(raw["model"]["n_min"]),
            pnd_max=float(raw["model"]["pnd_max"]),
            mindate=raw["model"]["mindate"],
            # prep rules
            PRIOR_YEAR=int(raw["data_rules"]["PRIOR_YEAR"]),
            CUTOFFS=dict(raw["CUTOFFS"]),
            KW_DATE1=raw["KW_DATES"]["date1"],
            KW_DATE2=raw["KW_DATES"]["date2"],
        )
