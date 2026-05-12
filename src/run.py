import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# loading tobit workflow config
from config import TrendConfig

# load utils
from utils import (
    load_table,
    build_output_dir,
    normalize_selected_wells,
    filter_by_selected_wells,
    assert_only_selected_wells,
    assert_not_empty,
)

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
from reporting.generate_report import generate_report


def main():
    # Load config
    config = TrendConfig.from_toml("configs/trend_config.toml")
    # optional well selection - if specified, will filter to only these wells for all steps
    selected_wells = normalize_selected_wells(config.selected_wells)
    well = load_table(config.well_info_well)
    screen = load_table(config.well_info_screen)
    well = filter_by_selected_wells(well, selected_wells)
    screen = filter_by_selected_wells(screen, selected_wells)
    if selected_wells:
        found = set(well["NAME"].astype(str).str.strip())
        missing = sorted(set(selected_wells) - found)
        if missing:
            raise ValueError(f"selected_wells not found in WELL.csv: {missing}")
    # Build output directory
    output_dir = build_output_dir(config.output_dir, config.run_ver)
    print(f"Running Tobit Trend Analysis with output_dir={output_dir}...")

    #############################
    # 00 - CALCULATE DISTANCE   #
    #############################
    print("Running distance calculations...")
    dist, stagedist = run_calculate_distance(
        well=well,
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
        well=well,
        screen=screen,
        yr=config.CHEM_YEAR,
        cfg=chem_cfg,
    )

    chem_rs.to_parquet(output_dir / "Cr_TrendData_2024.parquet", index=False)

    ################################
    # 02 - PREP WATER LEVEL DATA   #
    ################################
    print("Running water level import...")
    wl = load_table(config.wl_file)
    wl = filter_by_selected_wells(wl, selected_wells)
    river_stage = load_table(config.river_stage_file)

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
    # wl_rs = load_table(output_dir / "WL_TrendData_2024.parquet")  # test load

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
    # chem_rs = load_table(output_dir / "Cr_TrendData_2024.parquet")  # test load
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
    df.to_csv(output_dir / f"TTA_Results_{config.run_ver}.csv", index=False)

    ############################
    # 05 - REPORTING/PLOTTING  #
    ############################
    no_rs = load_table(config.no_rs_csv)
    generate_report(
        output_dir=output_dir,
        wells=well,
        dist=dist,
        wl_trends=wl_trends_df,
        chem_trends=df,
        wl_rs=wl_rs,
        chem_rs=chem_rs,
        no_rs=no_rs,
        river_shapefile=config.gis_river_shapefile,
        roads_shapefile=config.gis_roads_shapefile,
        ou_shapefile=config.gis_ou_shapefile,
        ous=config.ou_keep,
        map_crs=config.map_crs,
        run_ver=config.run_ver,
    )


if __name__ == "__main__":
    main()
