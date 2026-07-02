from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional
import tomllib

import pandas as pd


@dataclass(frozen=True)
class TrendConfig:
    """
    Container for user-editable settings loaded from the project TOML file.
    """

    # global settings
    run_id: str
    output_dir: Path
    selected_wells: list[str]
    global_min_date: pd.Timestamp
    global_max_date: pd.Timestamp

    # run_calculate_distance - script00
    well_info_well: Path
    gauge_locs: Path
    river_shapefile: Path

    # run_chemistry_import - script01
    chemistry_files: list[Path]
    river_stage_file: Path
    well_info_screen: Path
    well_filter_cols: list[str]
    well_filter_modes: list[str]
    well_filter_values: list[list[str]]
    mdl_sub_if_nonpositive_missing: float
    reviewq_remove_patterns: list[str]
    collection_purpose_exclude: list[str]
    duplicate_handling: str        # how duplicates are aggregated; placeholder ('daily_avg')
    write_chem_output: bool        # write preprocessed chemistry parquet to output_dir
    chem_max_date: pd.Timestamp   # per-step override; defaults to global_max_date

    # run_water_level_import - script02
    wl_file: Path
    wl_max_date: pd.Timestamp   # per-step override; defaults to global_max_date

    # run_water_level_trend_analysis - script03
    maxlag: int
    n_min: int
    pnd_max: float
    regression_start_date: str
    r_script_path: Path

    # run_tobit_trend_analysis - script04
    system_wells_csv: Path
    trend_breaks_csv: Path
    no_rs_csv: Path
    kw_csv: Path
    CUTOFFS: dict[str, pd.Timestamp]
    KW_DATE1: str
    KW_DATE2: str
    # tobit model
    dep: str

    # reporting
    gis_river_shapefile: Path
    gis_roads_shapefile: Path
    gis_ou_shapefile: Path
    map_crs: str

    # validation table
    save_validation_table: bool
    mcl_near_river: float
    mcl_far: float
    mcl_near_river_shapefile: Optional[Path]  # wells within 200 ft of river → MCL=mcl_near_river

    # water-level regression report
    wl_regression_report: bool

    @property
    def prior_year(self) -> int:
        """Year a well must have chemistry data in to be included; derived as global_max_date.year - 1."""
        return self.global_max_date.year - 1

    def validate_paths(self) -> None:
        """
        Raise FileNotFoundError if any required input file does not exist.

        Called automatically by ``from_toml()`` so the workflow fails fast
        rather than discovering missing files mid-run. ``output_dir`` is
        excluded because it is created by the workflow.
        """
        # Single-path fields — every Path field except output_dir and optional paths.
        # Optional[Path] fields are skipped when None (isinstance check handles this).
        _exclude = {"output_dir"}
        missing: list[str] = []

        for f in fields(self):
            if f.name in _exclude:
                continue
            val = getattr(self, f.name)
            if isinstance(val, Path) and not val.exists():
                missing.append(str(val))

        # chemistry_files is list[Path]
        for p in self.chemistry_files:
            if not p.exists():
                missing.append(str(p))

        if missing:
            lines = "\n  ".join(missing)
            raise FileNotFoundError(
                f"The following input files referenced in the config do not exist:\n  {lines}"
            )

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrendConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        gs = raw["global_settings"]
        global_max_date = pd.Timestamp(gs["global_max_date"])

        instance = cls(
            # global settings
            run_id=gs["run_id"],
            output_dir=Path(gs["output_dir"]),
            selected_wells=list(gs.get("selected_wells", [])),
            global_min_date=pd.Timestamp(gs["global_min_date"]),
            global_max_date=global_max_date,
            # run_calculate_distance - script00
            well_info_well=Path(raw["calculate_distance"]["well_info_well"]),
            gauge_locs=Path(raw["calculate_distance"]["gauge_locs"]),
            river_shapefile=Path(raw["calculate_distance"]["river_shapefile"]),
            # run_chemistry_import - script01
            chemistry_files=[Path(p) for p in raw["prep_chemistry"]["chemistry_files"]],
            river_stage_file=Path(raw["prep_chemistry"]["river_stage_file"]),
            well_info_screen=Path(raw["prep_chemistry"]["well_info_screen"]),
            well_filter_cols=list(raw["prep_chemistry"].get("well_filter_cols", [])),
            well_filter_modes=list(raw["prep_chemistry"].get("well_filter_modes", [])),
            well_filter_values=[
                list(v) for v in raw["prep_chemistry"].get("well_filter_values", [])
            ],
            mdl_sub_if_nonpositive_missing=float(
                raw["prep_chemistry"]["mdl_sub_if_nonpositive_missing"]
            ),
            reviewq_remove_patterns=list(
                raw["prep_chemistry"]["reviewq_remove_patterns"]
            ),
            collection_purpose_exclude=list(
                raw["prep_chemistry"]["collection_purpose_exclude"]
            ),
            duplicate_handling=str(
                raw["prep_chemistry"].get("duplicate_handling", "daily_avg")
            ),
            write_chem_output=bool(
                raw["prep_chemistry"].get("write_chem_output", True)
            ),
            chem_max_date=pd.Timestamp(
                raw["prep_chemistry"].get("max_date", gs["global_max_date"])
            ),
            # run_water_level_import - script02
            wl_file=Path(raw["prep_wl"]["wl_file"]),
            wl_max_date=pd.Timestamp(
                raw["prep_wl"].get("max_date", gs["global_max_date"])
            ),
            # run_water_level_trend_analysis - script03
            maxlag=int(raw["model"]["maxlag"]),
            n_min=int(raw["model"]["n_min"]),
            pnd_max=float(raw["model"]["pnd_max"]),
            regression_start_date=raw["model"]["regression_start_date"],
            r_script_path=Path(raw["tobit_trends"]["r_script_path"]),
            # run_tobit_trend_analysis - script04
            system_wells_csv=Path(raw["tobit_trends"]["system_wells_csv"]),
            trend_breaks_csv=Path(raw["tobit_trends"]["trend_breaks_csv"]),
            no_rs_csv=Path(raw["tobit_trends"]["no_rs_csv"]),
            kw_csv=Path(raw["tobit_trends"]["kw_csv"]),
            CUTOFFS={k: pd.Timestamp(v) for k, v in raw["CUTOFFS"].items()},
            KW_DATE1=raw["KW_DATES"]["date1"],
            KW_DATE2=raw["KW_DATES"]["date2"],
            # model
            dep=raw["model"]["dep"],
            # reporting
            gis_river_shapefile=Path(raw["reporting"]["gis_river_shapefile"]),
            gis_roads_shapefile=Path(raw["reporting"]["gis_roads_shapefile"]),
            gis_ou_shapefile=Path(raw["reporting"]["gis_ou_shapefile"]),
            map_crs=raw.get("reporting", {}).get("map_crs", "EPSG:2926"),
            # validation table
            save_validation_table=bool(
                raw.get("reporting", {}).get("save_validation_table", True)
            ),
            mcl_near_river=float(raw["reporting"]["mcl_near_river"]),
            mcl_far=float(raw["reporting"]["mcl_far"]),
            mcl_near_river_shapefile=(
                Path(raw["reporting"]["mcl_near_river_shapefile"])
                if raw.get("reporting", {}).get("mcl_near_river_shapefile")
                else None
            ),
            wl_regression_report=bool(
                raw.get("reporting", {}).get("wl_regression_report", False)
            ),
        )
        instance.validate_paths()
        return instance
