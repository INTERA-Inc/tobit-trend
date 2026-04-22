import os
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
import tempfile
import subprocess
import scipy
from scipy.stats import norm
import re


# ----------------------------
# Helpers
# ----------------------------
def current_version() -> str:
    return "0.7.3"


def to_datetime_date(s: pd.Series) -> pd.Series:
    # Cr_TrendData EVENT is date32[day] from parquet -> usually already datetime64[ns]
    dt = pd.to_datetime(s, errors="coerce")
    # treat as date (no time)
    return dt.dt.floor("D")


def year_of(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.year


def safe_str(s: pd.Series) -> pd.Series:
    return s.astype("string")


# ----------------------------
# TERM logic
# ----------------------------
def apply_system_cutoffs(
    df: pd.DataFrame, CUTOFFS: dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Assign initial TERM values based on system-level cutoff dates.

    Equivalent R logic:
    CHEM_RS[SYS, on=c('SYSTEM'), TERM := fifelse(EVENT < CUTOFF, 1, 2)]

    Splits each well's time series into two periods:
    TERM = 1 → before system remediation cutoff
    TERM = 2 → after cutoff

    Notes:
    - Applied to ALL rows (including rows with missing VAL).
    - Wells with SYSTEM = "NA" typically default to TERM = 2.
    """
    df = df.copy()
    df["TERM"] = 1

    # SYSTEM == "NA" stays 1
    mask_known = df["SYSTEM"].isin(CUTOFFS.keys())
    # For known systems: TERM=2 if EVENT >= cutoff
    for sys, cutoff in CUTOFFS.items():
        m = df["SYSTEM"] == sys
        df.loc[m, "TERM"] = np.where(df.loc[m, "EVENT"] < cutoff, 1, 2)

    return df


def compress_empty_terms_per_well(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unused TERM levels per well and renumber sequentially.

    Equivalent R logic:
    TM <- sort(unique(X[!is.na(VAL)]$TERM))
    for(i in 1:length(TM)){
        if(no rows with TERM == i and VAL not NA){
        decrement all TERM > i by 1
        }
    }

    Ensures TERM values are contiguous (1, 2, 3, …) within each well.
    Drops "empty" TERM periods (no observed concentration data).

    Why:
    - Some TERM bins may contain only NA values after filtering.
    - These empty periods are removed to avoid invalid regressions.

    Notes:
    - Only considers rows where VAL is not NA when determining emptiness.
    - Still modifies TERM for ALL rows in the well.
    """
    df = df.copy()
    out = []

    for name, g in df.groupby("NAME", sort=False):
        g = g.copy()

        tm = sorted(
            pd.to_numeric(g.loc[g["VAL"].notna(), "TERM"], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )

        # R: for(i in 1:length(TM))
        for i in range(1, len(tm) + 1):
            xsub = g.loc[
                g["VAL"].notna() & (pd.to_numeric(g["TERM"], errors="coerce") == i)
            ]
            if len(xsub) == 0:
                g.loc[pd.to_numeric(g["TERM"], errors="coerce") > i, "TERM"] = (
                    pd.to_numeric(
                        g.loc[pd.to_numeric(g["TERM"], errors="coerce") > i, "TERM"],
                        errors="coerce",
                    )
                    - 1
                )

        out.append(g)

    return pd.concat(out, ignore_index=True)


def _parse_trendbreak_date(series: pd.Series) -> pd.Series:
    """
    R-tolerant parser for TrendBreaks START/END.

    Handles:
    - m/d/YYYY
    - mm/dd/YYYY
    - junk trailing characters, e.g. '1/1/20111' -> '1/1/2011'
    - already clean ISO/datetime-like values

    Returns datetime64[ns] floored to day; invalid -> NaT.
    """
    s = pd.Series(series).copy()

    # First: direct parse for already-clean values like 2014-04-01
    dt = pd.to_datetime(s, errors="coerce")

    # Second: for anything still missing, extract the first m/d/yyyy pattern
    mask = dt.isna()
    if mask.any():
        s2 = s.astype("string").str.strip()

        # pull first valid-looking month/day/4-digit-year chunk
        extracted = s2.str.extract(r"(\d{1,2}/\d{1,2}/\d{4})", expand=False)

        dt2 = pd.to_datetime(extracted, format="%m/%d/%Y", errors="coerce")
        dt.loc[mask] = dt2.loc[mask]

    return pd.to_datetime(dt, errors="coerce").dt.floor("D")


def apply_manual_trend_breaks(
    df: pd.DataFrame, newtrends: pd.DataFrame
) -> pd.DataFrame:
    """
    Override TERM assignments using manual trend-break definitions.

    Equivalent R logic:
    for each break row:
        TERM := ifelse(EVENT >= START & EVENT < END, TREND, TERM)

    Applies user-defined TERM overrides for specific wells and date ranges.
    Replaces previously assigned TERM values within those intervals.

    Why:
    - Some wells require manual segmentation not captured by system cutoffs.
    - Allows domain-specific adjustments to trend periods.

    Notes:
    - Applied AFTER compression, so it can reintroduce non-sequential TERM values.
    - Applies to ALL rows (including VAL = NA).
    - Interval is [START, END) (inclusive of START, exclusive of END).
    - END = NA is treated as "current date".
    - Later rows in the CSV overwrite earlier ones if overlapping.
    """
    df = df.copy()
    nt = newtrends.copy()

    # normalize names like R string matching
    df["NAME"] = df["NAME"].astype("string").str.strip()
    df["EVENT"] = pd.to_datetime(df["EVENT"], errors="coerce").dt.floor("D")

    nt["NAME"] = nt["NAME"].astype("string").str.strip()

    required = {"NAME", "TREND", "START", "END"}
    missing = required - set(nt.columns)
    if missing:
        raise KeyError(f"TREND_BREAKS missing columns: {sorted(missing)}")

    nt["TREND"] = pd.to_numeric(nt["TREND"], errors="coerce")
    nt["START"] = _parse_trendbreak_date(nt["START"])
    nt["END"] = _parse_trendbreak_date(nt["END"])

    today = pd.Timestamp.today().floor("D")
    nt["END"] = nt["END"].fillna(today)

    # same practical behavior as R: rows with unusable START or TREND cannot apply
    nt = nt.loc[nt["NAME"].notna() & nt["START"].notna() & nt["TREND"].notna()].copy()

    # preserve file order, like the R for-loop over TSUB rows
    for _, row in nt.iterrows():
        m = (
            (df["NAME"] == row["NAME"])
            & (df["EVENT"] >= row["START"])
            & (df["EVENT"] < row["END"])
        )
        df.loc[m, "TERM"] = int(row["TREND"])

    return df


def apply_kw_extra_terms(
    df: pd.DataFrame, kw: pd.DataFrame, KW_DATE1: pd.Timestamp, KW_DATE2: pd.Timestamp
) -> pd.DataFrame:
    """
    Apply the KW-specific additional trend-period splits.

    Equivalent R logic:
    KWwells <- fread('05_Trends/Input/CY23/KW_selected_locations.csv')

    for (i in 1:nrow(KWwells)){

    NM <- KWwells$WELL_NAME[i]

    sub <- CHEM_RS[NAME == NM]
    CHEM_RS <- CHEM_RS[!NAME == NM]

    sub$TERM <- ifelse(sub$EVENT>=as.Date(ISOdate(2016,05,16)),sub$TERM + 1,sub$TERM)
    sub$TERM <- ifelse(sub$EVENT>=as.Date(ISOdate(2017,04,12)),sub$TERM + 1,sub$TERM)

    CHEM_RS <- rbind(CHEM_RS,sub)

    }

    Adds up to two extra TERM increments for selected KW wells.
    Rows before 2016-05-16 are unchanged.
    Rows from 2016-05-16 to 2017-04-11 get TERM + 1.
    Rows from 2017-04-12 onward get TERM + 2.

    Notes:
    - Applied only to wells listed in the KW input file.
    - Applied after manual trend breaks.
    - The increments are cumulative.
    """
    df = df.copy()
    kw_names = set(kw["WELL_NAME"].astype(str).tolist())

    m = df["NAME"].isin(kw_names)
    if m.any():
        df.loc[m, "TERM"] = df.loc[m, "TERM"] + (df.loc[m, "EVENT"] >= KW_DATE1).astype(
            int
        )
        df.loc[m, "TERM"] = df.loc[m, "TERM"] + (df.loc[m, "EVENT"] >= KW_DATE2).astype(
            int
        )
    return df


# ----------------------------
# ULAG logic from WLTrends_flat
# ----------------------------
def build_ulags(wl_trends_flat: pd.DataFrame) -> Dict[str, Optional[int]]:
    """
    Closer match to R:
      if well in names(WLLAG) and length(WL@SUM) > 1 then ULAG = WL@LAG else NULL

    In the flat export, the closest proxy is:
      SUM_rows > 1 and LAG present

    Do NOT filter on CLASS here; R does not.
    """
    df = wl_trends_flat.copy()
    df["KEY"] = df["KEY"].astype(str).str.strip()

    ok = df["SUM_rows"].fillna(0).astype(float).gt(1) & df["LAG"].notna()

    ulag = {k: int(round(float(v))) for k, v in df.loc[ok, ["KEY", "LAG"]].values}
    return ulag


# ----------------------------
# Main script-04 driver (up to modelling)
# ----------------------------
def run_script04_prep(
    CR_TREND_PARQUET: str,
    WL_TRENDS_FLAT_CSV: str,
    SYSTEM_WELLS_CSV: str,
    TREND_BREAKS_CSV: str,
    NO_RS_CSV: str,
    KW_CSV: str,
    RUM_CSV: str,
    PRIOR_YEAR: int,
    CUTOFFS: dict[str, pd.Timestamp],
    KW_DATE1: pd.Timestamp = pd.Timestamp("2016-05-16"),
    KW_DATE2: pd.Timestamp = pd.Timestamp("2017-04-12"),
) -> Tuple[pd.DataFrame, Dict[str, Optional[int]], set]:
    # Load
    chem = pd.read_parquet(CR_TREND_PARQUET)
    chem = chem.reset_index(drop=True).copy()
    chem["_src_order"] = np.arange(len(chem), dtype=int)
    wl_trends = pd.read_csv(WL_TRENDS_FLAT_CSV)

    sys = pd.read_csv(SYSTEM_WELLS_CSV)
    newtrends = pd.read_csv(TREND_BREAKS_CSV)
    no_rs = pd.read_csv(NO_RS_CSV)
    kw = pd.read_csv(KW_CSV)
    rum = pd.read_csv(RUM_CSV)

    # Standardize columns
    chem["NAME"] = safe_str(chem["NAME"])
    chem["EVENT"] = to_datetime_date(chem["EVENT"])
    # chem["SYSTEM"] = None  # added by join

    sys["NAME"] = safe_str(sys["NAME"])
    sys["SYSTEM"] = safe_str(sys["SYSTEM"])

    # Combine Chemistry and System Data
    chem = chem.merge(sys[["NAME", "SYSTEM"]], on="NAME", how="left")
    chem["SYSTEM"] = chem["SYSTEM"].fillna("NA")

    # Remove wells with no data in prior year
    chem["YEAR"] = year_of(chem["EVENT"])
    wells_prior = set(
        chem.loc[(chem["YEAR"] == PRIOR_YEAR) & chem["VAL"].notna(), "NAME"]
        .unique()
        .tolist()
    )
    chem = chem[chem["NAME"].isin(wells_prior)].copy()
    chem.drop(columns=["YEAR"], inplace=True)

    # TERM assignment
    ## Add Trend Term based on Cutoff Dates
    chem = apply_system_cutoffs(chem, CUTOFFS=CUTOFFS)
    ## Check to see if Data available for multiple Trends
    chem = compress_empty_terms_per_well(chem)
    ## Adjust Trend Breaks
    chem = apply_manual_trend_breaks(chem, newtrends)
    ## Add Trend Period for KW Remediation
    chem = apply_kw_extra_terms(chem, kw, KW_DATE1, KW_DATE2)

    # NEWRS list (NoRS + RUM)
    newrs_names = set(no_rs["NAME"].astype(str).tolist()) | set(
        rum["NAME"].astype(str).tolist()
    )

    # Extract Water-Level Lag Time ULAG
    ulags = build_ulags(wl_trends)

    return chem, ulags, newrs_names
