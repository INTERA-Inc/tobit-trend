import os
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import warnings
import tempfile
import subprocess
import scipy
from scipy.stats import norm
import re


# ----------------------------
# Helpers
# ----------------------------
def to_datetime_date(s: pd.Series) -> pd.Series:
    # Chem_TrendData EVENT is date32[day] from parquet -> usually already datetime64[ns]
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
        term_num = pd.to_numeric(g["TERM"], errors="coerce")

        tm = sorted(
            term_num[g["VAL"].notna()]
            .dropna()
            .unique()
            .tolist()
        )

        # R: for(i in 1:length(TM))
        for i in range(1, len(tm) + 1):
            xsub = g.loc[g["VAL"].notna() & (term_num == i)]
            if len(xsub) == 0:
                mask = term_num > i
                term_num = term_num.where(~mask, term_num - 1)

        g["TERM"] = term_num
        out.append(g)

    if not out:
        return pd.DataFrame(columns=df.columns)
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
    df: pd.DataFrame,
    newtrends: pd.DataFrame,
    max_date: pd.Timestamp,
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
    - END = NA is filled with ``max_date`` (caller passes analysis cutoff, e.g.
      ``pd.Timestamp(f"{CHEM_YEAR}-12-31")``), making results reproducible
      regardless of the calendar date the workflow is run.
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

    nt["END"] = nt["END"].fillna(max_date + pd.Timedelta(days=1))

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


def compute_term_limits(
    chem: pd.DataFrame,
    CUTOFFS: dict[str, pd.Timestamp],
    newtrends: pd.DataFrame,
    kw_names: set[str],
    KW_DATE1: pd.Timestamp,
    KW_DATE2: pd.Timestamp,
    global_min_date: pd.Timestamp,
    global_max_date: pd.Timestamp,
) -> dict[str, dict[int, tuple[str, str]]]:
    """
    Compute the actual date limits for each TERM period of each well.

    Uses configuration boundary dates (system cutoffs, manual break START/END,
    KW remediation dates) rather than chemistry sample dates, so limits reflect
    exact period definitions rather than the nearest quarterly sample.

    Parameters
    ----------
    chem
        Fully prepared chemistry DataFrame (post all TERM assignments).
    CUTOFFS
        Mapping of SYSTEM → system-level cutoff Timestamp.
    newtrends
        Raw TREND_BREAKS DataFrame (NAME, TREND, START, END).
    kw_names
        Well names subject to the KW remediation term splits.
    KW_DATE1, KW_DATE2
        KW remediation boundary dates (first and second split).
    global_min_date, global_max_date
        Analysis window start and end dates.

    Returns
    -------
    dict
        {well_name: {term_num: (start_date_str, end_date_str)}}
        Dates are ISO-format strings (YYYY-MM-DD).
    """
    # Parse manual breaks identically to apply_manual_trend_breaks.
    # breaks_by_well: {well_name: [(start, end_or_NaT), ...]}
    # When END is NaT (empty in CSV), the break is open-ended; the period
    # extends to global_max_date without adding a closing boundary.
    breaks_by_well: dict[str, list[tuple]] = {}
    required_break_cols = {"NAME", "START", "TREND", "END"}
    if not newtrends.empty and required_break_cols.issubset(newtrends.columns):
        nt = newtrends.copy()
        nt["NAME"] = nt["NAME"].astype("string").str.strip()
        nt["START"] = _parse_trendbreak_date(nt["START"])
        nt["END"] = _parse_trendbreak_date(nt["END"])
        # END remains NaT when empty — handled below as open-ended (→ global_max_date)
        nt = nt.dropna(subset=["NAME", "START", "TREND"])
        for _, row in nt.iterrows():
            name = str(row["NAME"])
            breaks_by_well.setdefault(name, []).append((row["START"], row["END"]))

    kw_date1 = pd.Timestamp(KW_DATE1)
    kw_date2 = pd.Timestamp(KW_DATE2)

    result: dict[str, dict[int, tuple[str, str]]] = {}

    for well_name, well_df in chem.groupby("NAME"):
        well_name_str = str(well_name)
        system = well_df["SYSTEM"].iloc[0] if "SYSTEM" in well_df.columns else None
        system = (
            None
            if (not system or str(system) in ("NA", "nan", "", "None"))
            else str(system)
        )

        # Collect all configuration-defined boundary dates for this well.
        bound_set: set[pd.Timestamp] = set()

        if system and system in CUTOFFS:
            bound_set.add(CUTOFFS[system])

        for start, end in breaks_by_well.get(well_name_str, []):
            if pd.notna(start):
                bound_set.add(pd.Timestamp(start))
            if pd.notna(end):
                end_ts = pd.Timestamp(end)
                # Add END as a closing boundary only when it falls strictly inside
                # the analysis window.  When END is NaT (empty in CSV) or is beyond
                # global_max_date, the break is treated as open-ended: the period
                # naturally closes at global_max_date (the final period_ends element).
                if end_ts < global_max_date:
                    bound_set.add(end_ts)

        if well_name_str in kw_names:
            bound_set.add(kw_date1)
            bound_set.add(kw_date2)

        boundaries = sorted(bound_set)

        # Build theoretical period intervals from those boundaries.
        period_starts = [global_min_date] + boundaries
        period_ends = [b - pd.Timedelta(days=1) for b in boundaries] + [global_max_date]

        term_limits: dict[int, tuple[str, str]] = {}

        for ps, pe in zip(period_starts, period_ends):
            period_mask = (well_df["EVENT"] >= ps) & (well_df["EVENT"] <= pe)
            period_df = well_df.loc[period_mask]
            if period_df.empty:
                continue

            valid_terms = period_df["TERM"].dropna()
            if valid_terms.empty:
                continue

            term_num = int(pd.Series(valid_terms.values).mode().iloc[0])

            if term_num not in term_limits:
                term_limits[term_num] = (str(ps.date()), str(pe.date()))
            # If this TERM number appears again in a later non-contiguous block
            # (e.g. a manual break creates TERM3 in the middle, and the original
            # TERM2 resumes afterwards), skip it.  Recording only the first
            # contiguous block correctly reflects the config-defined boundary for
            # each TERM without spanning over intermediate periods.

        result[well_name_str] = term_limits

    return result


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
    chem: pd.DataFrame,
    wl_trends: pd.DataFrame,
    SYSTEM_WELLS_CSV: str,
    TREND_BREAKS_CSV: str,
    NO_RS_CSV: str,
    KW_CSV: str,
    PRIOR_YEAR: int,
    CUTOFFS: dict[str, pd.Timestamp],
    KW_DATE1: pd.Timestamp = pd.Timestamp("2016-05-16"),
    KW_DATE2: pd.Timestamp = pd.Timestamp("2017-04-12"),
    max_date: pd.Timestamp = pd.Timestamp("2024-12-31"),
    global_min_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[int]], set, dict]:
    # Load
    chem = chem.reset_index(drop=True).copy()
    chem["_src_order"] = np.arange(len(chem), dtype=int)
    # wl_trends = pd.read_parquet(WL_TRENDS)

    sys = pd.read_csv(SYSTEM_WELLS_CSV)
    newtrends = pd.read_csv(TREND_BREAKS_CSV)
    no_rs = pd.read_csv(NO_RS_CSV)
    kw = pd.read_csv(KW_CSV)

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
    chem = apply_manual_trend_breaks(chem, newtrends, max_date=max_date)
    ## Add Trend Period for KW Remediation
    chem = apply_kw_extra_terms(chem, kw, KW_DATE1, KW_DATE2)

    # NEWRS list = NoRS file (includes former RUM locs)
    newrs_names = set(no_rs["NAME"].astype(str).str.strip().tolist())

    # Extract Water-Level Lag Time ULAG
    ulags = build_ulags(wl_trends)

    # Compute exact TERM period boundaries from configuration dates.
    kw_names = set(kw["WELL_NAME"].astype(str).tolist())
    _global_min = global_min_date if global_min_date is not None else chem["EVENT"].min()
    term_limits = compute_term_limits(
        chem=chem,
        CUTOFFS=CUTOFFS,
        newtrends=newtrends,
        kw_names=kw_names,
        KW_DATE1=KW_DATE1,
        KW_DATE2=KW_DATE2,
        global_min_date=_global_min,
        global_max_date=max_date,
    )

    return chem, ulags, newrs_names, term_limits