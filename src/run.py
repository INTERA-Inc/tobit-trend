import pandas as pd
import pyarrow.parquet as pq

from config import TrendConfig

# from types import PrepData, TrendResult
from preprocessing.water_level_import_02 import run_water_level_import
from preprocessing.water_level_trends_03 import (
    run_water_level_trend_analysis,
    flatten_water_level_trends,
)
from preprocessing.tobit_CR_prep_04 import run_script04_prep
from model.tobit_CR_04_mod import do_tobit_rstyle

# from groundwater_trends.export import write_csv, compare_with_r


def main():
    config = TrendConfig.from_toml("configs/trend_config.toml")
    ## 00 - CALCULATE DISTANCE
    ## 01 - PREP CHEMISTRY DATA
    ## 02 - PREP WATER LEVEL DATA
    print("Running water level import...")
    wl = pd.read_parquet(config.wl_data_parquet)
    rs = pd.read_parquet(config.rs_data_parquet)
    wl_rs = run_water_level_import(WL=wl, RS=rs)
    ## 03 - WATER LEVEL TRENDS
    print("Running water level trend analysis...")
    res = run_water_level_trend_analysis(
        wl_rs=wl_rs,
        MAXLAG=config.maxlag,
        LOG=config.log,
        # TS=config.ts,
        MINDATE=config.mindate,
        N=config.n_min,
        PND=config.pnd_max,
        r_script_path=config.r_script_path,
    )
    # 3. Convert to DataFrame
    out_df = flatten_water_level_trends(res)
    out_df.to_csv("WL_trends_flattended.csv", index=False)

    ## 04 - CHEMISTRY TOBIT TREND ANALYIS
    # print("Running chemistry tobit prep...")
    # chem_rs, ulags, newrs_names = run_script04_prep(
    #     CR_TREND_PARQUET=config.cr_trend_parquet,
    #     WL_TRENDS_FLAT_CSV=config.wl_trends_flat_csv,
    #     SYSTEM_WELLS_CSV=config.system_wells_csv,
    #     TREND_BREAKS_CSV=config.trend_breaks_csv,
    #     NO_RS_CSV=config.no_rs_csv,
    #     KW_CSV=config.kw_csv,
    #     RUM_CSV=config.rum_csv,
    #     PRIOR_YEAR=config.PRIOR_YEAR,
    #     CUTOFFS=config.CUTOFFS,
    #     KW_DATE1=config.KW_DATE1,
    #     KW_DATE2=config.KW_DATE2,
    # )

    # print("Prepared rows:", len(chem_rs))
    # print("Unique wells:", chem_rs["NAME"].nunique())
    # print("ULAG wells:", len(ulags))
    # print("NEWRS wells:", len(newrs_names))
    # print("Done with prep, starting model...")

    # res = do_tobit_rstyle(
    #     x=chem_rs,
    #     DEP=config.dep,
    #     INDEP=config.indep,
    #     LOG=config.log,
    #     MAXLAG=config.maxlag,
    #     N=config.n_min,
    #     PND=config.pnd_max,
    #     r_script_path=config.r_script_path,
    #     ulags=ulags,
    #     newrs_names=newrs_names,
    # )

    # df = pd.DataFrame(res)
    # df.to_csv(config.output_csv, index=False)
    # print(f"Done: wrote {config.output_csv}")


if __name__ == "__main__":
    main()
