from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class TrendConfig:
    """
    Container for user-editable settings loaded from the project TOML file.
    """

    # global settings
    run_ver: str
    output_dir: Path

    # run_calculate_distance - script00
    well_info_well: Path
    gauge_locs: Path
    river_shapefile: Path

    # run_chemistry_import - script01
    chemistry_files: list[Path]
    river_stage_file: Path
    well_info_screen: Path
    chromium_analyte: str
    hexchrom_analyte: str
    filtered_keep_value: str
    combined_analyte_name: str
    mdl_sub_if_nonpositive_missing: float
    ou_keep: list[str]
    status_exclude: list[str]
    reviewq_remove_patterns: list[str]
    collection_purpose_exclude: list[str]
    trend_min_year: int
    CHEM_YEAR: int

    # run_water_level_import - script02
    wl_file: Path
    WL_YEAR: int

    # run_water_level_trend_analysis - script03
    maxlag: int
    log: str
    n_min: int
    pnd_max: float
    mindate: str
    r_script_path: Path

    # run_tobit_trend_analysis - script04
    system_wells_csv: Path
    trend_breaks_csv: Path
    no_rs_csv: Path
    kw_csv: Path
    rum_csv: Path
    PRIOR_YEAR: int
    CUTOFFS: dict[str, str]
    KW_DATE1: str
    KW_DATE2: str
    # tobit model
    dep: str
    indep: list[str]

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrendConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        return cls(
            # global settings
            run_ver=raw["global_settings"]["run_ver"],
            output_dir=Path(raw["global_settings"]["output_dir"]),
            # run_calculate_distance - script00
            well_info_well=Path(raw["calculate_distance"]["well_info_well"]),
            gauge_locs=Path(raw["calculate_distance"]["gauge_locs"]),
            river_shapefile=Path(raw["calculate_distance"]["river_shapefile"]),
            # run_chemistry_import - script01
            chemistry_files=[Path(p) for p in raw["prep_chem"]["chemistry_files"]],
            river_stage_file=Path(raw["prep_chem"]["river_stage_file"]),
            well_info_screen=Path(raw["prep_chem"]["well_info_screen"]),
            chromium_analyte=raw["prep_chem"]["chromium_analyte"],
            hexchrom_analyte=raw["prep_chem"]["hexchrom_analyte"],
            filtered_keep_value=raw["prep_chem"]["filtered_keep_value"],
            combined_analyte_name=raw["prep_chem"]["combined_analyte_name"],
            mdl_sub_if_nonpositive_missing=float(
                raw["prep_chem"]["mdl_sub_if_nonpositive_missing"]
            ),
            ou_keep=list(raw["prep_chem"]["ou_keep"]),
            status_exclude=list(raw["prep_chem"]["status_exclude"]),
            reviewq_remove_patterns=list(raw["prep_chem"]["reviewq_remove_patterns"]),
            collection_purpose_exclude=list(
                raw["prep_chem"]["collection_purpose_exclude"]
            ),
            trend_min_year=int(raw["prep_chem"]["trend_min_year"]),
            CHEM_YEAR=int(raw["prep_chem"]["CHEM_YEAR"]),
            # run_water_level_import - script02
            wl_file=Path(raw["prep_wl"]["wl_file"]),
            WL_YEAR=int(raw["prep_wl"]["WL_YEAR"]),
            # run_water_level_trend_analysis - script03
            log=raw["model"]["log"],
            maxlag=int(raw["model"]["maxlag"]),
            n_min=int(raw["model"]["n_min"]),
            pnd_max=float(raw["model"]["pnd_max"]),
            mindate=raw["model"]["mindate"],
            r_script_path=Path(raw["tobit_trends"]["r_script_path"]),
            # run_tobit_trend_analysis - script04
            system_wells_csv=Path(raw["tobit_trends"]["system_wells_csv"]),
            trend_breaks_csv=Path(raw["tobit_trends"]["trend_breaks_csv"]),
            no_rs_csv=Path(raw["tobit_trends"]["no_rs_csv"]),
            kw_csv=Path(raw["tobit_trends"]["kw_csv"]),
            rum_csv=Path(raw["tobit_trends"]["rum_csv"]),
            PRIOR_YEAR=int(raw["data_rules"]["PRIOR_YEAR"]),
            CUTOFFS=dict(raw["CUTOFFS"]),
            KW_DATE1=raw["KW_DATES"]["date1"],
            KW_DATE2=raw["KW_DATES"]["date2"],
            # model
            dep=raw["model"]["dep"],
            indep=list(raw["model"]["indep"]),
            # dist_file=Path(raw["prep_wl"]["dist_file"]),
            # stagedist_file=Path(raw["prep_wl"]["stagedist_file"]),
            # cr_trend_parquet=Path(raw["prep_wl"]["cr_trend_parquet"]),
            # wl_trends_flat_csv=Path(raw["prep_wl"]["wl_trends_flat_csv"]),
            # wl_data_parquet=Path(raw["prep_wl"]["wl_data_parquet"]),
            # system_wells_csv=Path(raw["prep_wl"]["system_wells_csv"]),
            # trend_breaks_csv=Path(raw["prep_wl"]["trend_breaks_csv"]),
            # no_rs_csv=Path(raw["prep_wl"]["no_rs_csv"]),
            # kw_csv=Path(raw["prep_wl"]["kw_csv"]),
            # rum_csv=Path(raw["prep_wl"]["rum_csv"]),
            # output_csv=Path(raw["prep_wl"]["output_csv"]),
            # # model
            # dep=raw["model"]["dep"],
            # indep=list(raw["model"]["indep"]),
            # # prep rules
        )
