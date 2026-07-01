import traceback

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from time import perf_counter

# loading tobit workflow config
from tta.config import TrendConfig

# load utils
from tta.reporting.wl_regression_report_05 import wl_regression_report
from tta.utils import (
    load_table,
    build_output_dir,
    normalize_selected_wells,
    filter_by_selected_wells,
    assert_only_selected_wells,
    assert_not_empty,
    setup_logger,
    apply_well_filters,
    parse_args,
)

# loading tobit workflow scripts
from tta.preprocessing.calculate_distance_00 import run_calculate_distance
from tta.preprocessing.chemistry_import_01 import (
    run_chemistry_import,
    ChemistryImportConfig,
)
from tta.preprocessing.water_level_import_02 import run_water_level_import
from tta.preprocessing.water_level_trends_03 import (
    run_water_level_trend_analysis,
    flatten_water_level_trends,
)
from tta.preprocessing.tobit_prep_04 import run_script04_prep
from tta.model.tobit_model_04 import do_tobit_rstyle
from tta.reporting.generate_report_05 import generate_report
from tta.reporting.wl_regression_report_05 import wl_regression_report
from tta.reporting.validation_table import build_validation_table, load_near_river_wells

MODEL_LOG = "log"
MODEL_INDEP = ["INTERP", "EVENT"]
WL_LOG = "log"


def main() -> None:
    logger = None
    try:
        args = parse_args()
        config = TrendConfig.from_toml(args.config)
        # Load config
        # config = TrendConfig.from_toml("configs/config.toml")
        # Build output directory
        output_dir = build_output_dir(config.output_dir, config.run_id)
        # Set up logging
        logger = setup_logger(output_dir)
        run_start = perf_counter()
        logger.info("Config file: %s", args.config)
        # optional well selection - if specified, will filter to only these wells for all steps
        selected_wells = normalize_selected_wells(config.selected_wells)
        # well filtering
        well = load_table(config.well_info_well)
        screen = load_table(config.well_info_screen)

        logger.debug("Input SCREEN file: %s", config.well_info_screen)
        logger.debug("Chemistry files: %s", config.chemistry_files)
        logger.debug("Water-level file: %s", config.wl_file)
        logger.debug("River-stage file: %s", config.river_stage_file)
        n_wells_raw = well["NAME"].nunique()
        # Permanent project-wide inclusion/exclusion filters.
        well = apply_well_filters(
            well=well,
            filter_cols=config.well_filter_cols,
            filter_modes=config.well_filter_modes,
            filter_values=config.well_filter_values,
        )

        if well.empty:
            raise ValueError("Global well filtering removed all wells.")

        # Apply selected_wells after project-wide filters.
        well = filter_by_selected_wells(well, selected_wells)

        if well.empty:
            raise ValueError("selected_wells filtering removed all wells.")

        screen = screen.loc[
            screen["NAME"].astype(str).isin(well["NAME"].astype(str))
        ].copy()
        logger.info("Global well filter retained %s wells", well["NAME"].nunique())
        logger.debug("Retained wells: %s", sorted(well["NAME"].astype(str).unique()))
        if selected_wells:
            found = set(well["NAME"].astype(str).str.strip())
            missing = sorted(set(selected_wells) - found)
            if missing:
                raise ValueError(f"selected_wells not found in WELL.csv: {missing}")
        logger.info(
            "Well filtering: %s raw wells -> %s retained wells",
            n_wells_raw,
            well["NAME"].nunique(),
        )
        logger.info(
            "Operational units retained: %s",
            ", ".join(sorted(well["OU"].dropna().astype(str).unique())),
        )

        logger.info("Starting Tobit Trend Analysis workflow")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Run version id: {config.run_id}")
        logger.info(
            f"Selected wells: {config.selected_wells if config.selected_wells else 'ALL'}"
        )
        logger.debug("Input WELL file: %s", config.well_info_well)

        #############################
        # 00 - CALCULATE DISTANCE   #
        #############################
        # print("Running distance calculations...")
        logger.info("Step 00: Running distance calculations")
        dist, stagedist = run_calculate_distance(
            well=well,
            gauge=config.gauge_locs,
            river_shapefile=config.river_shapefile,
        )
        # dist.to_csv(output_dir / "DIST.csv", index=False)
        # stagedist.to_csv(output_dir / "STAGEDIST.csv", index=False)
        logger.info(
            f"Step 00 complete: DIST rows={len(dist)}, STAGEDIST rows={len(stagedist)}"
        )

        ##############################
        # 01 - PREP CHEMISTRY DATA   #
        ##############################
        # print("Running chemistry import...")
        logger.info("Step 01: Running chemistry import")
        chem_cfg = ChemistryImportConfig(
            mdl_sub_if_nonpositive_missing=config.mdl_sub_if_nonpositive_missing,
            reviewq_remove_patterns=config.reviewq_remove_patterns,
            collection_purpose_exclude=config.collection_purpose_exclude,
            trend_min_year=config.global_min_date.year,
        )

        chem_rs = run_chemistry_import(
            chem_files=config.chemistry_files,
            stage_comb=load_table(config.river_stage_file),
            dist=dist,  # from script 00
            stagedist=stagedist,  # from script 00
            well=well,
            screen=screen,
            yr=config.chem_max_date.year,
            cfg=chem_cfg,
        )
        logger.info(
            f"Step 01 complete: chem_rs rows={len(chem_rs)}, wells={chem_rs['NAME'].nunique()}"
        )
        ################################
        # 02 - PREP WATER LEVEL DATA   #
        ################################
        logger.info("Step 02: Running water level import")
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
            yr=config.wl_max_date.year,
            trend_min_year=config.global_min_date.year,
        )
        # wl_rs.to_parquet(
        #     output_dir / f"WL_TrendData_{config.run_id}.parquet", index=False
        # )
        # wl_rs = load_table(output_dir / f"WL_TrendData_{config.run_id}.parquet")  # test load
        logger.info(
            f"Step 02 complete: wl_rs rows={len(wl_rs)}, wells={wl_rs['NAME'].nunique()}"
        )

        ############################
        # 03 - WATER LEVEL TRENDS  #
        ############################
        logger.info("Step 03: Running water level trend analysis")
        res = run_water_level_trend_analysis(
            wl_rs=wl_rs,
            MAXLAG=config.maxlag,
            LOG=WL_LOG,
            MINDATE=config.regression_start_date,
            N=config.n_min,
            PND=config.pnd_max,
            r_script_path=config.r_script_path,
        )
        wl_trends_df = flatten_water_level_trends(res)
        # wl_trends_df.to_parquet(
        #     output_dir / f"WL_trends_{config.run_id}.parquet", index=False
        # )
        logger.info(f"Step 03 complete: wl_trends rows={len(wl_trends_df)}")

        ########################################
        # 04 - CHEMISTRY TOBIT TREND ANALYSIS  #
        ########################################
        logger.info("Step 04: Running chemistry Tobit trend analysis")
        chem_rs, ulags, newrs_names, term_limits = run_script04_prep(
            chem=chem_rs,  # from script 01
            wl_trends=wl_trends_df,  # from script 03
            SYSTEM_WELLS_CSV=config.system_wells_csv,
            TREND_BREAKS_CSV=config.trend_breaks_csv,
            NO_RS_CSV=config.no_rs_csv,
            KW_CSV=config.kw_csv,
            PRIOR_YEAR=config.prior_year,
            CUTOFFS=config.CUTOFFS,
            KW_DATE1=config.KW_DATE1,
            KW_DATE2=config.KW_DATE2,
            max_date=config.chem_max_date,
            global_min_date=config.global_min_date,
        )

        logger.debug(f"Prepared rows: {len(chem_rs)}")
        logger.debug(f"Unique wells: {chem_rs['NAME'].nunique()}")
        logger.debug(f"ULAG wells: {len(ulags)}")
        logger.debug(f"NEWRS wells: {len(newrs_names)}")
        
        if config.write_chem_output:
            chem_rs.to_parquet(
                output_dir / f"Chem_TrendData_{config.run_id}.parquet", index=False
            )
            logger.info("Step 04 prep: chemistry output written to Chem_TrendData_%s.parquet", config.run_id)
        logger.info("Done with prep, starting model...")

        res = do_tobit_rstyle(
            x=chem_rs,
            DEP=config.dep,
            INDEP=MODEL_INDEP,
            LOG=MODEL_LOG,
            MAXLAG=config.maxlag,
            N=config.n_min,
            PND=config.pnd_max,
            r_script_path=config.r_script_path,
            ulags=ulags,
            newrs_names=newrs_names,
        )

        # Build DataFrame from results; drop vcov (numpy arrays don't serialize to CSV).
        df_tobit = pd.DataFrame(res)
        df_tobit.drop(columns=["vcov"], errors="ignore").to_csv(
            output_dir / f"TTA_full_term_stats_{config.run_id}.csv", index=False
        )
        df_tobit = pd.DataFrame(res).drop(columns=["vcov"], errors="ignore")
        logger.info(f"Step 04 complete: Tobit results rows={len(df_tobit)}")
        # logger.info(
        #     f"Step 04 complete: Results written to {output_dir / f'TTA_Results_{config.run_id}.csv'}"
        # )
        
        ##############################################
        # 04b - WRITE OUTPUT TABLE (tblSummary port) #
        ##############################################
        if config.save_validation_table:
            logger.info("Step 04b: Writing results table")
            near_river_wells = load_near_river_wells(config.mcl_near_river_shapefile)
            if config.mcl_near_river_shapefile and not near_river_wells:
                logger.warning(
                    "mcl_near_river_shapefile was set but no well names could be loaded; "
                    "all wells will use mcl_far=%.1f.",
                    config.mcl_far,
                )
            val_tbl = build_validation_table(
                results=res,
                chem_rs=chem_rs,
                well=well,
                dist=dist,
                cutoffs=config.CUTOFFS,
                chem_year=config.chem_max_date.year,
                near_river_wells=near_river_wells,
                mcl_near=config.mcl_near_river,
                mcl_far=config.mcl_far,
                dep=config.dep,
                term_limits=term_limits,
            )
            val_path = output_dir / f"TTA_Results_{config.run_id}.csv"
            val_tbl.to_csv(val_path, index=False)
            logger.info(
                "Step 04b complete: Results table rows=%d, written to %s",
                len(val_tbl),
                val_path,
            )

        ############################
        # 05 - REPORTING/PLOTTING  #
        ############################
        logger.info("Step 05: Running reporting")
        no_rs = load_table(config.no_rs_csv)
        ous = sorted(well["OU"].dropna().astype(str).unique())
        generate_report(
            output_dir=output_dir,
            wells=well,
            dist=dist,
            wl_trends=wl_trends_df,
            chem_trends=df_tobit,  # from script 04
            wl_rs=wl_rs,
            chem_rs=chem_rs,
            no_rs=no_rs,
            river_shapefile=config.gis_river_shapefile,
            roads_shapefile=config.gis_roads_shapefile,
            ou_shapefile=config.gis_ou_shapefile,
            ous=ous,
            map_crs=config.map_crs,
            run_id=config.run_id,
        )

        if config.wl_regression_report:
            wl_regression_report(
                run_id=config.run_id,
                output_dir=output_dir,
                dist=dist,
                wl_rs=wl_rs,
                wl_trends=wl_trends_df,
                wells=well,
                gis_river_shapefile=config.gis_river_shapefile,
                gis_roads_shapefile=config.gis_roads_shapefile,
                gis_ou_shapefile=config.gis_ou_shapefile,
                map_crs=config.map_crs
            )

        if config.wl_regression_report:
            logger.info("Step 05b: Running water-level regression report")
            wl_regression_report(
                run_id=config.run_id,
                output_dir=output_dir,
                dist=dist,
                wl_rs=wl_rs,
                wl_trends=wl_trends_df,
                wells=well,
                gis_river_shapefile=config.gis_river_shapefile,
                gis_roads_shapefile=config.gis_roads_shapefile,
                gis_ou_shapefile=config.gis_ou_shapefile,
                map_crs=config.map_crs,
            )
            logger.info("Step 05b complete: WL regression report generated")

        logger.info("Step 05 complete: reports generated for OUs: %s", ", ".join(ous))
        logger.info("Workflow completed successfully")
        elapsed = perf_counter() - run_start
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            "Total runtime: %02d:%02d:%04.1f (hh:mm:ss)",
            int(hours),
            int(minutes),
            seconds,
        )
        logger.info("Output directory: %s", output_dir)

    except Exception as exc:
        if logger is not None:
            logger.exception("Workflow failed: %s", exc)
        else:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
