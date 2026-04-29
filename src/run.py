import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

# loading tobit workflow config
from config import TrendConfig

# loading tobit workflow scripts
from preprocessing.calculate_distance_00 import run_calculate_distance
from preprocessing.chemistry_import_01 import (
    run_chemistry_import,
    ChemistryImportConfig,
)
from preprocessing.water_level_import_02 import run_water_level_import
from preprocessing.water_level_trends_03 import (
    run_water_level_trend_analysis,
    flatten_water_level_trends,
)
from preprocessing.tobit_CR_prep_04 import run_script04_prep
from model.tobit_CR_04_mod import do_tobit_rstyle


# from groundwater_trends.export import write_csv, compare_with_r


def load_table(path: str) -> pd.DataFrame:
    """Load tables, CSV or Parquet format."""
    if str(path).endswith(".parquet"):
        return pq.read_table(path).to_pandas()
    if str(path).endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input format: {path}")


def _build_output_dir(
    output_dir: str | Path | None, run_ver: str | None
) -> Path | None:
    """Build output directory path, creating it if it doesn't exist."""
    if output_dir is None:
        return None

    out_dir = Path(output_dir)
    if run_ver is not None:
        out_dir = out_dir / str(run_ver)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    # Load config
    config = TrendConfig.from_toml("configs/trend_config.toml")
    # Build output directory
    output_dir = _build_output_dir(config.output_dir, config.run_ver)
    print(f"Running Tobit Trend Analysis with output_dir={output_dir}...")
    #############################
    # 00 - CALCULATE DISTANCE   #
    #############################
    print("Running distance calculations...")
    dist, stagedist = run_calculate_distance(
        well=config.well_info_well,
        gauge=config.gauge_locs,
        river_shapefile=config.river_shapefile,
    )
    dist.to_csv(output_dir / "DIST.csv", index=False)
    stagedist.to_csv(output_dir / "STAGEDIST.csv", index=False)
    ##############################
    # 01 - PREP CHEMISTRY DATA   #
    ##############################
    print("Running chemistry import...")
    chem_cfg = ChemistryImportConfig(
        chromium_analyte=config.chromium_analyte,
        hexchrom_analyte=config.hexchrom_analyte,
        filtered_keep_value=config.filtered_keep_value,
        combined_analyte_name=config.combined_analyte_name,
        mdl_sub_if_nonpositive_missing=config.mdl_sub_if_nonpositive_missing,
        ou_keep=config.ou_keep,
        status_exclude=config.status_exclude,
        reviewq_remove_patterns=config.reviewq_remove_patterns,
        collection_purpose_exclude=config.collection_purpose_exclude,
        trend_min_year=config.trend_min_year,
    )

    chem_rs = run_chemistry_import(
        chem_files=config.chemistry_files,
        stage_comb=load_table(config.river_stage_file),
        dist=dist,  # from script 00
        stagedist=stagedist,  # from script 00
        well=load_table(config.well_info_well),
        screen=load_table(config.well_info_screen),
        yr=config.CHEM_YEAR,
        cfg=chem_cfg,
    )

    chem_rs.to_parquet(output_dir / "Cr_TrendData_2024.parquet", index=False)

    ################################
    # 02 - PREP WATER LEVEL DATA   #
    ################################
    print("Running water level import...")
    wl = load_table(config.wl_file)
    river_stage = load_table(config.river_stage_file)
    well = load_table(config.well_info_well)
    screen = load_table(config.well_info_screen)

    wl_rs = run_water_level_import(
        wl=wl,
        river_stage=river_stage,
        dist=dist,
        stagedist=stagedist,
        well=well,
        screen=screen,
        yr=config.WL_YEAR,
    )
    wl_rs.to_parquet(output_dir / "WL_TrendData_2024.parquet", index=False)
    wl_rs = load_table(output_dir / "WL_TrendData_2024.parquet")  # test load
    ############################
    # 03 - WATER LEVEL TRENDS  #
    ############################
    print("Running water level trend analysis...")
    res = run_water_level_trend_analysis(
        wl_rs=wl_rs,
        MAXLAG=config.maxlag,
        LOG=config.log,
        MINDATE=config.mindate,
        N=config.n_min,
        PND=config.pnd_max,
        r_script_path=config.r_script_path,
    )
    wl_trends_df = flatten_water_level_trends(res)
    wl_trends_df.to_parquet(output_dir / "WL_trends_2024.parquet", index=False)
    chem_rs = load_table(output_dir / "Cr_TrendData_2024.parquet")  # test load
    # wl_trends_df = load_table(output_dir / "WLTrends_flat.csv")  # test load
    ########################################
    # 04 - CHEMISTRY TOBIT TREND ANALYSIS  #
    ########################################
    print("Running chemistry tobit prep...")
    chem_rs, ulags, newrs_names = run_script04_prep(
        chem=chem_rs,  # from script 01
        wl_trends=wl_trends_df,  # from script 03
        SYSTEM_WELLS_CSV=config.system_wells_csv,
        TREND_BREAKS_CSV=config.trend_breaks_csv,
        NO_RS_CSV=config.no_rs_csv,
        KW_CSV=config.kw_csv,
        RUM_CSV=config.rum_csv,
        PRIOR_YEAR=config.PRIOR_YEAR,
        CUTOFFS=config.CUTOFFS,
        KW_DATE1=config.KW_DATE1,
        KW_DATE2=config.KW_DATE2,
    )

    print("Prepared rows:", len(chem_rs))
    print("Unique wells:", chem_rs["NAME"].nunique())
    print("ULAG wells:", len(ulags))
    print("NEWRS wells:", len(newrs_names))
    print("Done with prep, starting model...")

    res = do_tobit_rstyle(
        x=chem_rs,
        DEP=config.dep,
        INDEP=config.indep,
        LOG=config.log,
        MAXLAG=config.maxlag,
        N=config.n_min,
        PND=config.pnd_max,
        r_script_path=config.r_script_path,
        ulags=ulags,
        newrs_names=newrs_names,
    )

    df = pd.DataFrame(res)
    df.to_csv(output_dir / "TobitResults.csv", index=False)

    ############################
    # 05 - REPORTING/PLOTTING  #
    ############################


if __name__ == "__main__":
    main()
