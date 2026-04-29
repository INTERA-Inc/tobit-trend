from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ChemistryImportConfig:
    chromium_analyte: str = "Chromium"
    hexchrom_analyte: str = "Hexavalent Chromium"
    filtered_keep_value: str = "Y"
    combined_analyte_name: str = "Hex. Chromium and Fil. Chromium"
    mdl_sub_if_nonpositive_missing: float = 1.0

    ou_keep: Sequence[str] = ("100-KR-4", "100-HR-3-D", "100-HR-3-H")
    status_exclude: Sequence[str] = ("DECOMMISSIONED-V", "DRILLING CANCELLED")

    reviewq_remove_patterns: Sequence[str] = ("Y", "R")
    collection_purpose_exclude: Sequence[str] = (
        "IH",
        "C",
        "VAR",
        "VP",
        "RC",
        "WM",
        "S",
        "PE",
        "T",
        "PD",
        "CNF",
    )

    trend_min_year: int = 2008


def read_chem_heis(path: str, eform: str = "%m/%d/%Y %H:%M:%S") -> pd.DataFrame:
    """
    Direct port of readCHEMHEIS() as uploaded.
    """
    df = pd.read_csv(path, low_memory=False)

    if "DEPTH" in df.columns:
        df = df.loc[df["DEPTH"] != "m"].copy()

    df = df.loc[~df["VAL"].isna()].copy()

    for col in ["EVENT", "LOAD_DATE_TIME"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace("EDT ", "", regex=False)
            s = s.str.replace("EST ", "", regex=False)
            df[col] = pd.to_datetime(s, format=eform, errors="coerce")

    if "STD_MDA" in df.columns and "MDL" in df.columns:
        df["MDL"] = np.where(df["STD_MDA"].notna(), df["STD_MDA"], df["MDL"])

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _year(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.year


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _prep_river_stage_for_well(
    stage_comb: pd.DataFrame,
    stagedist: pd.DataFrame,
    well_name: str,
) -> pd.DataFrame:
    gauge_vals = stagedist.loc[stagedist["NAME"] == well_name, "STAGE"]

    if len(gauge_vals) == 0:
        return pd.DataFrame(
            {
                "GAUGE": "PRD",
                "EVENT": stage_comb["EVENT"],
                "RS": stage_comb["PRD"],
                "UNITS": "m",
            }
        )

    gauge = gauge_vals.iloc[0]
    if gauge not in stage_comb.columns:
        raise KeyError(f"Gauge column {gauge!r} not found in stage_comb")

    return pd.DataFrame(
        {
            "GAUGE": gauge,
            "EVENT": stage_comb["EVENT"],
            "RS": stage_comb[gauge],
            "UNITS": "m",
        }
    )


def _prep_data_trends_rs(
    rs: pd.DataFrame,
    mindate: pd.Timestamp,
    maxdate: pd.Timestamp,
) -> pd.DataFrame:
    """
    Script-01-specific RS path equivalent to prepDataTrends(RS=..., RSCOL=...).
    """
    out = rs.loc[:, ["GAUGE", "EVENT", "RS", "UNITS"]].copy()
    out.columns = ["NAME", "EVENT", "STAGE", "UNITS"]
    out["EVENT"] = _normalize_date(out["EVENT"])
    out = out.loc[(out["EVENT"] >= mindate) & (out["EVENT"] <= maxdate)].copy()
    out = out.loc[~out["STAGE"].isna()].copy()

    out = out.groupby(["NAME", "EVENT", "UNITS"], as_index=False, dropna=False)[
        "STAGE"
    ].mean()

    full = pd.DataFrame(
        {"EVENT": pd.date_range(out["EVENT"].min(), out["EVENT"].max(), freq="D")}
    )
    full = full.merge(out, on="EVENT", how="left", sort=False)
    full["INTERP"] = full["STAGE"].interpolate(method="linear", limit_direction="both")
    full["NAME"] = out["NAME"].dropna().iloc[0]
    return full.sort_values("EVENT").reset_index(drop=True)


def _prep_data_trends_chem(
    chem: pd.DataFrame,
    mindate: pd.Timestamp,
    maxdate: pd.Timestamp,
) -> pd.DataFrame:
    """
    Script-01-specific CHEM path equivalent to prepDataTrends(CHEM=..., CHEMCOL=...).
    This script passes already daily-averaged chemistry, so duplicate logic is not needed here.
    """
    out = chem.loc[
        :, ["NAME", "EVENT", "ANALYTE", "FILTERED", "VAL", "LABQ", "UNIT", "MDL"]
    ].copy()
    out.columns = [
        "NAME",
        "EVENT",
        "ANALYTE",
        "FILTERED",
        "VAL",
        "QUAL",
        "UNITS",
        "MDL",
    ]

    out["EVENT"] = _normalize_date(out["EVENT"])
    out["VAL"] = pd.to_numeric(out["VAL"], errors="coerce")

    out = out.loc[(out["EVENT"] >= mindate) & (out["EVENT"] <= maxdate)].copy()

    out["NDS"] = False
    labq = out["QUAL"].fillna("").astype(str)
    out.loc[labq.str.contains("U", regex=True), "NDS"] = True

    out = out.sort_values("EVENT").reset_index(drop=True)
    return out


def _combine_data_rs_chem(rs: pd.DataFrame, chem: pd.DataFrame) -> pd.DataFrame:
    """
    Direct port of combineData() chemistry + river-stage + FILTERED branch.
    """
    x = rs.copy().rename(columns={"NAME": "RS_NAME", "UNITS": "RS_UNITS"})
    y = chem.copy().rename(columns={"NAME": "WELL_NAME", "UNITS": "CHEM_UNITS"})

    out_all = []

    for well_name, xwell in y.groupby("WELL_NAME", sort=False):
        well_out = []

        for analyte_name, yan in xwell.groupby("ANALYTE", sort=False):
            filt_out = []

            for _, z in yan.groupby("FILTERED", sort=False):
                tmp = x.merge(z, on="EVENT", how="left", sort=False)

                tmp["WELL_NAME"] = well_name
                tmp["ANALYTE"] = analyte_name

                rs_units_vals = x["RS_UNITS"].dropna().unique()
                tmp["RS_UNITS"] = rs_units_vals[0] if len(rs_units_vals) > 0 else np.nan

                tmp = tmp.loc[
                    :,
                    [
                        "WELL_NAME",
                        "EVENT",
                        "ANALYTE",
                        "FILTERED",
                        "VAL",
                        "CHEM_UNITS",
                        "QUAL",
                        "NDS",
                        "RS_NAME",
                        "INTERP",
                        "RS_UNITS",
                    ],
                ].copy()

                tmp = tmp.rename(columns={"WELL_NAME": "NAME"})
                filt_out.append(tmp)

            if filt_out:
                well_out.append(pd.concat(filt_out, ignore_index=True))

        if well_out:
            out_all.append(pd.concat(well_out, ignore_index=True))

    if not out_all:
        return pd.DataFrame(
            columns=[
                "NAME",
                "EVENT",
                "ANALYTE",
                "FILTERED",
                "VAL",
                "CHEM_UNITS",
                "QUAL",
                "NDS",
                "RS_NAME",
                "INTERP",
                "RS_UNITS",
            ]
        )

    return pd.concat(out_all, ignore_index=True)


def run_chemistry_import(
    chem_files: Sequence[str],
    stage_comb: pd.DataFrame,
    dist: pd.DataFrame,
    stagedist: pd.DataFrame,
    well: pd.DataFrame,
    screen: pd.DataFrame,
    yr: int,
    cfg: Optional[ChemistryImportConfig] = None,
) -> pd.DataFrame:
    """
    Direct port of 01_ImportData_Chemistry_clean.R using CSV/parquet inputs
    instead of RData/TXT objects loaded inside R.
    """
    if cfg is None:
        cfg = ChemistryImportConfig()

    # --Import Chemistry Dataset--#
    chem_parts = [read_chem_heis(path) for path in chem_files]
    CHEM = pd.concat(chem_parts, ignore_index=True)
    CHEM["EVENT"] = pd.to_datetime(CHEM["EVENT"], errors="coerce")
    CHEM = CHEM.loc[_year(CHEM["EVENT"]) <= yr].copy()

    # --Create Chromium and Hex Chromium Datasets--#
    CHROM = pd.concat(
        [
            CHEM.loc[
                (CHEM["ANALYTE"] == cfg.chromium_analyte)
                & (CHEM["FILTERED"] == cfg.filtered_keep_value)
            ].copy(),
            CHEM.loc[CHEM["ANALYTE"] == cfg.hexchrom_analyte].copy(),
        ],
        ignore_index=True,
    )

    CHROM["ANALYTE_ORG"] = CHROM["ANALYTE"]
    CHROM["ANALYTE"] = cfg.combined_analyte_name
    CHROM["MDL"] = np.where(
        (CHROM["VAL"] <= 0) & (CHROM["MDL"].isna()),
        cfg.mdl_sub_if_nonpositive_missing,
        CHROM["MDL"],
    )

    # --Subset Data for OUs of Interest--#
    WELLS = well.loc[well["OU"].isin(cfg.ou_keep)].copy()
    WELLS = WELLS.loc[~WELLS["STATUS"].isin(cfg.status_exclude), "NAME"].copy()
    CHROM = CHROM.loc[CHROM["NAME"].isin(WELLS)].copy()

    # --exclude samples where REVIEWQ include flags Y or R in it.--#
    reviewq = CHROM["REVIEWQ"].fillna("").astype(str)
    review_mask = pd.Series(False, index=CHROM.index)
    for pat in cfg.reviewq_remove_patterns:
        review_mask = review_mask | reviewq.str.contains(pat, regex=False)
    CHROM = CHROM.loc[~review_mask].copy()

    # [2] data filtering ============================================================
    CHROM = CHROM.loc[
        ~CHROM["COLLECTION_PURPOSE"].isin(cfg.collection_purpose_exclude)
    ].copy()

    # [3] Non-detect handling ======================================================
    CHROM["NDS"] = False
    CHROM["VAL0"] = CHROM["VAL"]

    labq = CHROM["LABQ"].fillna("").astype(str)
    CHROM.loc[labq.str.contains("U", regex=True), "NDS"] = True

    CHROM.loc[(CHROM["NDS"] == True) & CHROM["MDL"].notna(), "VAL"] = CHROM["MDL"]
    CHROM.loc[(CHROM["VAL"] <= 0) & CHROM["MDL"].notna(), "VAL"] = CHROM["MDL"]

    # [4] --Calculate Average Daily Concentration ==================================
    CHROM["YEAR"] = CHROM["EVENT"].dt.year
    CHROM["MONTH"] = CHROM["EVENT"].dt.month
    CHROM["DAY"] = CHROM["EVENT"].dt.day

    tmp_tbl = CHROM.loc[:, ["NAME", "YEAR", "MONTH", "DAY"]].copy()
    flag_dup = tmp_tbl.duplicated(keep=False)

    all_duplicates = CHROM.loc[flag_dup].copy()
    all_duplicates = all_duplicates.loc[
        all_duplicates["ANALYTE_ORG"] == cfg.hexchrom_analyte
    ].copy()

    CHROM_no_dup = CHROM.loc[~flag_dup].copy()

    CHROM = pd.concat([all_duplicates, CHROM_no_dup], ignore_index=True)
    CHROM = CHROM.sort_values(["NAME", "EVENT", "ANALYTE_ORG"]).reset_index(drop=True)

    CHEM_tmp = CHROM.groupby(
        ["NAME", "ANALYTE", "YEAR", "MONTH", "DAY"], as_index=False
    )["VAL"].mean()

    CHEM_unq = CHROM.loc[
        :,
        [
            "NAME",
            "EVENT",
            "FILTERED",
            "ANALYTE",
            "UNIT",
            "LABQ",
            "REVIEWQ",
            "VALIDATION_QUALIFIER",
            "COLLECTION_PURPOSE",
            "MDL",
            "YEAR",
            "MONTH",
            "DAY",
        ],
    ].copy()

    CHEM_unq2 = CHEM_unq.drop_duplicates(
        subset=["NAME", "ANALYTE", "YEAR", "MONTH", "DAY"], keep="first"
    ).copy()

    CHEM_f = CHEM_unq2.merge(
        CHEM_tmp,
        on=["NAME", "ANALYTE", "YEAR", "MONTH", "DAY"],
        how="left",
        sort=False,
    )

    CHROM = CHEM_f.loc[
        :,
        [
            "NAME",
            "EVENT",
            "FILTERED",
            "ANALYTE",
            "VAL",
            "UNIT",
            "LABQ",
            "REVIEWQ",
            "VALIDATION_QUALIFIER",
            "COLLECTION_PURPOSE",
            "MDL",
            "YEAR",
        ],
    ].copy()

    # --Combine Chemistry and River Gauge Data--#
    stage_comb = stage_comb.copy()
    stage_comb["EVENT"] = pd.to_datetime(stage_comb["EVENT"], errors="coerce")

    chrs = []
    for well_name, X in CHROM.groupby("NAME", sort=False):
        if len(X) > 2 and X["EVENT"].dt.year.max() >= cfg.trend_min_year:
            RS = _prep_river_stage_for_well(stage_comb, stagedist, well_name)

            DATA_RS = _prep_data_trends_rs(
                RS,
                mindate=pd.Timestamp(cfg.trend_min_year, 1, 1),
                maxdate=pd.Timestamp(yr, 12, 31),
            )
            DATA_CHEM = _prep_data_trends_chem(
                X,
                mindate=pd.Timestamp(cfg.trend_min_year, 1, 1),
                maxdate=pd.Timestamp(yr, 12, 31),
            )

            CHEM_RS_one = _combine_data_rs_chem(rs=DATA_RS, chem=DATA_CHEM)
            chrs.append(CHEM_RS_one)

    CHEM_RS = pd.concat(chrs, ignore_index=True) if chrs else pd.DataFrame()

    # --Combine Chemistry and Water-level Data with Distance to River--#
    CHEM_RS = CHEM_RS.merge(dist, on="NAME", how="left", sort=False)

    # --Add Well Coordinates--#
    COORDS = well.loc[:, ["NAME", "XCOORDS", "YCOORDS", "ZCOORDS"]].copy()
    CHEM_RS = CHEM_RS.merge(COORDS, on="NAME", how="left", sort=False)

    # --Add Well Screen Data--#
    CHEM_RS = CHEM_RS.merge(screen, on="NAME", how="left", sort=False)

    # --Add Well OU--#
    AREA = well.loc[:, ["NAME", "OU"]].copy()
    CHEM_RS = CHEM_RS.merge(AREA, on="NAME", how="left", sort=False)

    # --Subset Final Dataset--#
    CHEM_RS = CHEM_RS.loc[
        :,
        [
            "NAME",
            "XCOORDS",
            "YCOORDS",
            "TOP",
            "BOT",
            "OU",
            "DIST",
            "EVENT",
            "ANALYTE",
            "VAL",
            "CHEM_UNITS",
            "QUAL",
            "NDS",
            "RS_NAME",
            "INTERP",
            "RS_UNITS",
        ],
    ].copy()

    return CHEM_RS
