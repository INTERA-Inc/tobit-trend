"""
Build a validation table that matches the R tblSummary() output column for column.

One row per well (last TERM only).  Mean/UCL/LCL are model-predicted values
for the target year, back-transformed from log scale.  The CI is based on the
delta method with a t-distribution, exactly as in R's CLinterval().
"""
from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------

def _days_in_year(year: int) -> int:
    """Number of days in *year*, matching R's daysperYear()."""
    return 366 if calendar.isleap(year) else 365


def _compute_time_axis(dt: pd.DataFrame) -> np.ndarray:
    """
    Port of R's covars() TIME construction:
      DT$TIME[1] <- as.numeric(format(min(DT$EVENT[1]), '%j'))
      DT$TIME    <- (DT$EVENT - DT$EVENT[1]) + DT$TIME[1]

    dt must be sorted by EVENT with a 0-based integer index.
    Returns a 1-D float array of length len(dt).
    """
    first_event = dt["EVENT"].iloc[0]
    julian_day = int(first_event.strftime("%j"))
    return (dt["EVENT"] - first_event).dt.days.values.astype(float) + julian_day


# ---------------------------------------------------------------------------
# Smoothing-spline INTERP predictor (port of R covars() for 2-INDEP)
# ---------------------------------------------------------------------------

def _spline_covars(
    dt: pd.DataFrame,
    times_predict: np.ndarray,
    cov: str = "INTERP",
) -> np.ndarray:
    """
    Return predicted covariate values at *times_predict* using a smoothing
    spline fitted on (TIME, cov) from *dt*.

    For prediction times beyond the observed range, substitutes the median of
    the last 10 years of observed data (matching R's ifelse cap in covars()).

    dt must be sorted by EVENT with a 0-based integer index.
    """
    time_axis = _compute_time_axis(dt)

    mask = dt[cov].notna().values
    time_0 = time_axis[mask]
    interp_0 = dt.loc[mask, cov].values

    if len(time_0) < 2:
        fill = float(np.nanmean(interp_0)) if len(interp_0) > 0 else np.nan
        return np.full(len(times_predict), fill)

    # chem_rs can contain duplicate RS dates for wells with both FILTERED=True and
    # FILTERED=False chemistry records: _combine_data_rs_chem does one LEFT JOIN per
    # FILTERED group, so every RS date appears once per group.  INTERP is identical
    # for both rows (same date → same interpolated stage), so averaging is lossless.
    if len(np.unique(time_0)) < len(time_0):
        _df = pd.DataFrame({"TIME": time_0, cov: interp_0})
        _df = _df.groupby("TIME", sort=True)[cov].mean().reset_index()
        time_0 = _df["TIME"].values
        interp_0 = _df[cov].values

    if len(time_0) < 2:
        return np.full(len(times_predict), float(np.nanmean(interp_0)))

    # Prefer make_smoothing_spline (scipy ≥ 1.10, matches R's GCV criterion).
    # Fall back to UnivariateSpline when unavailable.
    try:
        from scipy.interpolate import make_smoothing_spline
        spline = make_smoothing_spline(time_0, interp_0)
        pred = spline(times_predict)
    except ImportError:
        from scipy.interpolate import UnivariateSpline
        k = min(3, len(time_0) - 1)
        spline = UnivariateSpline(time_0, interp_0, k=k)
        pred = spline(times_predict)

    # Cap extrapolation at the boundary with median of last 10 years.
    max_time = float(time_axis.max())
    cutoff_year = pd.Timestamp.now().year - 10
    recent_mask = dt["EVENT"].dt.year >= cutoff_year
    recent_vals = dt.loc[recent_mask & dt[cov].notna(), cov]
    if len(recent_vals) > 0:
        median_cap = float(recent_vals.median())
    else:
        median_cap = float(np.nanmean(interp_0))

    pred[times_predict >= max_time] = median_cap
    return pred


# ---------------------------------------------------------------------------
# CLinterval for a single year interval (port of R CLinterval())
# ---------------------------------------------------------------------------

def _cl_interval_year(
    dt: pd.DataFrame,
    use_interp: bool,
    coef: np.ndarray,
    vcov: np.ndarray,
    df: int,
    from_: float,
    to_: float,
    alpha: float,
    mindate: pd.Timestamp,
    cov: str = "INTERP",
    n_ticks: int = 100,
) -> dict[str, float]:
    """
    Port of R's CLinterval() restricted to interval='year'.

    Returns {'Fit', 'LCL', 'UCL'} on the log scale (before back-transformation).
    coef = [Intercept, INTERP, EVENT] (2-INDEP) or [Intercept, EVENT] (1-INDEP).
    vcov must match coef dimensions.
    """
    eps = 0.5 * (to_ - from_) / (n_ticks + 1)
    times_predict = np.linspace(from_ + eps, to_ - eps, n_ticks)

    # Trim prediction times before the first data point in the first interval.
    if from_ == 0:
        min_julian = int(dt["EVENT"].min().strftime("%j"))
        if min_julian > 0:
            times_predict = times_predict[times_predict >= min_julian]
    if len(times_predict) == 0:
        times_predict = np.array([(from_ + to_) / 2.0])

    # Build means vector  [1, mean_INTERP, mean_EVENT_abs]  or  [1, mean_EVENT_abs]
    mindate_numeric = float((mindate - pd.Timestamp("1970-01-01")).days)

    if use_interp:
        interp_pred = _spline_covars(dt, times_predict, cov)
        mean_interp = float(np.mean(interp_pred))
        mean_time = float(np.mean(times_predict))
        mean_event_abs = mindate_numeric + mean_time
        means_vec = np.array([1.0, mean_interp, mean_event_abs])
    else:
        mean_time = float(np.mean(times_predict))
        mean_event_abs = mindate_numeric + mean_time
        means_vec = np.array([1.0, mean_event_abs])

    y_pred = float(means_vec @ coef)
    var_pred = float(means_vec @ vcov @ means_vec)
    sigma = float(np.sqrt(max(0.0, var_pred)))

    # R: cl <- y.predict + c(LCL=1, UCL=-1) * t.crit * sigma
    t_crit = float(t_dist.ppf(alpha / 2.0, df))

    return {
        "Fit": y_pred,
        "LCL": y_pred + 1.0 * t_crit * sigma,
        "UCL": y_pred - 1.0 * t_crit * sigma,
    }


# ---------------------------------------------------------------------------
# calcUCL for the last year only (port of R calcUCL(), interval='year')
# ---------------------------------------------------------------------------

def _calc_ucl_last_year(
    coef: np.ndarray,
    vcov: np.ndarray,
    df: int,
    dt: pd.DataFrame,
    lag: float,
    use_interp: bool,
    log: str,
    chem_year: int,
    dep: str = "VAL",
    cov: str = "INTERP",
    con: float = 95.0,
) -> tuple[float, float, float]:
    """
    Compute model-predicted (Mean, UCL, LCL) on original concentration scale
    for *chem_year*.  This is a Python port of R's calcUCL(..., interval='year')
    restricted to the last (target) year column.

    Returns (Mean, UCL, LCL).  All NaN on failure.
    """
    dt = dt.sort_values("EVENT").reset_index(drop=True)

    # Apply lag to covariate, matching R: DT[[cov]] <- lagCol(DT[[cov]], -LAG)
    if use_interp and pd.notna(lag) and lag > 0:
        from tta.model.tobit_model_04 import lag_col_rstyle
        dt = dt.copy()
        dt[cov] = lag_col_rstyle(dt[cov].values, -int(lag))

    # MINDATE = Jan 1 of first year that has non-NA dep data.
    dt_nonmiss = dt.dropna(subset=[dep])
    if dt_nonmiss.empty:
        return np.nan, np.nan, np.nan
    first_year = int(dt_nonmiss["EVENT"].dt.year.min())
    mindate = pd.Timestamp(year=first_year, month=1, day=1)
    dt = dt[dt["EVENT"] >= mindate].reset_index(drop=True)

    # Build cumulative year table from first_year to chem_year.
    years = list(range(first_year, chem_year + 1))
    durations = [_days_in_year(y) for y in years]
    cumulative = np.cumsum(durations, dtype=float)
    starts = np.concatenate([[0.0], cumulative[:-1]])

    from_ = float(starts[-1])
    to_ = float(cumulative[-1])
    alpha = (100.0 - con) / 100.0

    cl = _cl_interval_year(
        dt=dt,
        use_interp=use_interp,
        coef=coef,
        vcov=vcov,
        df=df,
        from_=from_,
        to_=to_,
        alpha=alpha,
        mindate=mindate,
        cov=cov,
    )

    # Back-transform from log scale.
    if log == "log":
        return np.exp(cl["Fit"]), np.exp(cl["UCL"]), np.exp(cl["LCL"])
    if log == "log10":
        return 10.0 ** cl["Fit"], 10.0 ** cl["UCL"], 10.0 ** cl["LCL"]
    return cl["Fit"], cl["UCL"], cl["LCL"]


# ---------------------------------------------------------------------------
# MCL shapefile loader
# ---------------------------------------------------------------------------

def load_near_river_wells(shapefile: Optional[Path], name_col: str = "WELL_NAME") -> set[str]:
    """
    Return the set of well names contained in *shapefile*.

    The shapefile is expected to have a *name_col* attribute column, matching
    R's ``STD$WELL_NAME`` layer (wells within 200 ft of the river).  Returns
    an empty set if *shapefile* is None or missing.
    """
    if shapefile is None:
        return set()
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shapefile)
        if name_col in gdf.columns:
            return set(gdf[name_col].dropna().astype(str))
        logger.warning(
            "MCL shapefile %s has no column '%s'; MCL defaulting to far value for all wells.",
            shapefile,
            name_col,
        )
        return set()
    except Exception as exc:
        logger.warning("Could not load MCL near-river shapefile %s: %s", shapefile, exc)
        return set()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Static columns; TERM{n}_min / TERM{n}_max are appended dynamically.
_STATIC_COLUMNS = [
    "Name", "XCoords", "YCoords", "Distance", "OU", "System", "Model",
    "Lag", "LAG_ORIGIN", "Analyte", "Trend", "Mean", "UCL", "LCL", "STD",
    "Mean_Exceed", "UCL_Exceed", "Trend_Break", "Use_RiverStage",
    "RiverStage_Coeff", "RiverStage_Coeff_stderror",
    "Date_Coeff", "Date_Coeff_stderror",
    "Intercept", "Intercept_stderror",
    "Transformation", "N", "PND",
    "TERM",
]


def build_validation_table(
    results: list[dict],
    chem_rs: pd.DataFrame,
    well: pd.DataFrame,
    dist: pd.DataFrame,
    cutoffs: dict[str, pd.Timestamp],
    chem_year: int,
    near_river_wells: set[str],
    mcl_near: float,
    mcl_far: float,
    dep: str = "VAL",
    cov: str = "INTERP",
    p: float = 0.05,
    con: float = 95.0,
    term_limits: Optional[dict[str, dict[int, tuple[str, str]]]] = None,
) -> pd.DataFrame:
    """
    Build one-row-per-well validation table matching R's tblSummary() output.

    Parameters
    ----------
    results
        Raw list[dict] returned by do_tobit_rstyle (must include 'vcov' key).
    chem_rs
        Prepared chemistry DataFrame (post Step 04) with at minimum:
        NAME, EVENT, VAL, NDS, INTERP, TERM, ANALYTE, OU, SYSTEM columns.
    well
        Well info DataFrame with NAME, XCOORDS, YCOORDS columns.
    dist
        Distance DataFrame with NAME, DIST columns.
    cutoffs
        Mapping of SYSTEM → Trend_Break Timestamp (from config.CUTOFFS).
    chem_year
        Target year for Mean/UCL/LCL prediction (from config.chem_max_date.year).
    near_river_wells
        Set of well names assigned MCL=mcl_near (within 200 ft of river).
    mcl_near
        MCL value for near-river wells.
    mcl_far
        MCL value for all other wells.
    dep
        Dependent variable column name ('VAL').
    cov
        River stage covariate column name ('INTERP').
    p
        Significance threshold for trend classification.
    con
        Confidence level for UCL/LCL (95).
    term_limits
        Exact TERM period boundaries from preprocessing, keyed by well name.
        {well_name: {term_num: (start_date_str, end_date_str)}}.
        When provided, TERM{n}_min / TERM{n}_max reflect the configuration
        cutoff dates rather than the nearest quarterly chemistry sample.

    Returns
    -------
    pd.DataFrame
        One row per well, with columns matching R's tblSummary.
    """
    # Build lookup tables from well and dist DataFrames.
    xcoord_col = "XCOORDS" if "XCOORDS" in well.columns else None
    ycoord_col = "YCOORDS" if "YCOORDS" in well.columns else None

    well_idx = well.set_index("NAME")
    dist_idx = dist.set_index("NAME")["DIST"].to_dict() if "DIST" in dist.columns else {}

    # Find last ITER (TERM) per well in results.
    last_result: dict[str, dict] = {}
    for r in results:
        w = r.get("WELL")
        if w is None:
            continue
        it = r.get("ITER") or 0
        if w not in last_result or it > (last_result[w].get("ITER") or 0):
            last_result[w] = r

    rows: list[dict] = []

    for well_name, row in last_result.items():
        last_iter = row.get("ITER")

        # All data for the well across every TERM — matches R's X@DATA which is the
        # full well dataset, not just the last TERM.  Used for N, PND, year-mean
        # fallback, and the INTERP spline in _calc_ucl_last_year.
        dt_all = chem_rs[chem_rs["NAME"] == well_name].sort_values("EVENT").reset_index(drop=True)

        if dt_all.empty:
            logger.warning("No chemistry rows found for well %s; skipping.", well_name)
            continue

        # --- Metadata (stable across TERMs; take from first available row) ---
        _analytes = dt_all["ANALYTE"].dropna().unique().tolist() if "ANALYTE" in dt_all.columns else []
        analyte = " & ".join(_analytes) if _analytes else np.nan
        ou = str(dt_all["OU"].iloc[0]) if "OU" in dt_all.columns else np.nan
        system = str(dt_all["SYSTEM"].iloc[0]) if "SYSTEM" in dt_all.columns else np.nan

        try:
            xcoords = float(well_idx.loc[well_name, xcoord_col]) if xcoord_col else np.nan
        except (KeyError, TypeError):
            xcoords = np.nan
        try:
            ycoords = float(well_idx.loc[well_name, ycoord_col]) if ycoord_col else np.nan
        except (KeyError, TypeError):
            ycoords = np.nan

        dist_val = dist_idx.get(well_name, np.nan)
        mcl = mcl_near if well_name in near_river_wells else mcl_far
        cut = cutoffs.get(system) if (system and system not in ("", "nan")) else np.nan

        # --- N, PND, and TERM info from all observed rows across every TERM ---
        dt_nonmiss = dt_all.dropna(subset=[dep])
        n_obs = len(dt_nonmiss)
        n_nds = int(dt_nonmiss["NDS"].sum()) if "NDS" in dt_nonmiss.columns else 0
        pnd_str = f"{round(n_nds / n_obs * 100) if n_obs > 0 else 0}%"

        term_nums = (
            sorted(dt_nonmiss["TERM"].dropna().unique().tolist())
            if "TERM" in dt_nonmiss.columns else []
        )
        well_term_limits = (term_limits or {}).get(str(well_name), {})
        n_terms = len(well_term_limits) if well_term_limits else len(term_nums)
        term_dates: dict = {}

        # Populate config-defined limits for every configured TERM, regardless
        # of whether that TERM has any non-null chemistry observations.  This
        # ensures the validation columns reflect all configured input TERMs,
        # not just those that happen to have detected samples.
        for _ti, (start_str, end_str) in well_term_limits.items():
            term_dates[f"TERM{_ti}_min"] = start_str
            term_dates[f"TERM{_ti}_max"] = end_str

        # For TERMs present in the data but absent from the config limits
        # (e.g. when term_limits was not computed), fall back to data-driven dates.
        for _t in term_nums:
            _ti = int(_t)
            if _ti not in well_term_limits:
                _t_events = dt_nonmiss.loc[dt_nonmiss["TERM"] == _t, "EVENT"]
                term_dates[f"TERM{_ti}_min"] = (
                    str(_t_events.min().date()) if len(_t_events) > 0 else np.nan
                )
                term_dates[f"TERM{_ti}_max"] = (
                    str(_t_events.max().date()) if len(_t_events) > 0 else np.nan
                )

        # --- Model metadata ---
        lag_origin = row.get("lag_origin", np.nan)
        lag = row.get("LAG")
        fit_ok = bool(row.get("fit_ok", False))
        model_type = row.get("model_type", "")
        use_interp = model_type == "INTERP+EVENT"
        log_transform = row.get("LOG", "log") or "log"
        model_label = row.get("MODEL", np.nan)

        # --- Case 1: No lag evaluation (length(X@LAG)==0 in R) ---
        # In Python this maps to LAG being NA/None AND model not fit.
        no_lag = lag is None or (isinstance(lag, float) and np.isnan(lag))
        if no_lag and not fit_ok:
            yr_data = dt_all[dt_all["EVENT"].dt.year == chem_year].dropna(subset=[dep])
            mean_val = float(yr_data[dep].mean()) if len(yr_data) > 0 else np.nan
            rows.append(_no_trend_row(
                well_name, xcoords, ycoords, dist_val, ou, system, model_label,
                analyte, mean_val, mcl, cut, log_transform, n_obs, pnd_str,
                lag_origin, n_terms, term_dates,
            ))
            continue

        # --- p-value extraction ---
        p_trend = row.get("p_trend")
        p_event = row.get("p_event")
        beta_event = row.get("beta_event")

        p_trend_val = float(p_trend) if p_trend is not None and not (isinstance(p_trend, float) and np.isnan(p_trend)) else 1.0
        p_event_val = float(p_event) if p_event is not None and not (isinstance(p_event, float) and np.isnan(p_event)) else 1.0

        # --- Trend classification (matches R tblSummary logic) ---
        if not fit_ok:
            trend = "Not Performed"
        elif p_trend_val < p and p_event_val < p:
            if pd.notna(beta_event) and float(beta_event) < 0:
                trend = "Decreasing"
            else:
                trend = "Increasing"
        else:
            trend = "Not Significant"

        # --- Mean/UCL/LCL ---
        if trend == "Not Performed":
            yr_data = dt_all[dt_all["EVENT"].dt.year == chem_year].dropna(subset=[dep])
            mean_conc = float(yr_data[dep].mean()) if len(yr_data) > 0 else np.nan
            ucl_conc = np.nan
            lcl_conc = np.nan
        else:
            vcov = row.get("vcov")
            if vcov is None or not isinstance(vcov, np.ndarray):
                mean_conc = ucl_conc = lcl_conc = np.nan
                logger.warning("vcov missing for %s; Mean/UCL/LCL set to NaN.", well_name)
            else:
                if use_interp:
                    coef_vec = np.array([
                        float(row.get("beta_intercept", np.nan)),
                        float(row.get("beta_interp", np.nan)),
                        float(row.get("beta_event", np.nan)),
                    ])
                else:
                    coef_vec = np.array([
                        float(row.get("beta_intercept", np.nan)),
                        float(row.get("beta_event", np.nan)),
                    ])
                try:
                    mean_conc, ucl_conc, lcl_conc = _calc_ucl_last_year(
                        coef=coef_vec,
                        vcov=vcov,
                        df=int(row.get("df") or 1),
                        dt=dt_all,   # full well history, matching R's X@DATA
                        lag=float(lag) if pd.notna(lag) else np.nan,
                        use_interp=use_interp,
                        log=log_transform,
                        chem_year=chem_year,
                        dep=dep,
                        cov=cov,
                        con=con,
                    )
                except Exception as exc:
                    logger.warning("CI calc failed for %s: %s", well_name, exc)
                    mean_conc = ucl_conc = lcl_conc = np.nan

        mean_exceed = bool(mean_conc > mcl) if pd.notna(mean_conc) else np.nan
        ucl_exceed = bool(ucl_conc > mcl) if pd.notna(ucl_conc) else np.nan

        rows.append({
            "Name": well_name,
            "XCoords": xcoords,
            "YCoords": ycoords,
            "Distance": dist_val,
            "OU": ou,
            "System": system,
            "Model": model_label,
            "Lag": lag if not no_lag else np.nan,
            "LAG_ORIGIN": lag_origin,
            "Analyte": analyte,
            "Trend": trend,
            "Mean": mean_conc,
            "UCL": ucl_conc,
            "LCL": lcl_conc,
            "STD": mcl,
            "Mean_Exceed": mean_exceed,
            "UCL_Exceed": ucl_exceed,
            "Trend_Break": cut,
            "Use_RiverStage": use_interp if fit_ok else np.nan,
            "RiverStage_Coeff": row.get("beta_interp") if use_interp else np.nan,
            "RiverStage_Coeff_stderror": row.get("se_interp") if use_interp else np.nan,
            "Date_Coeff": row.get("beta_event") if fit_ok else np.nan,
            "Date_Coeff_stderror": row.get("se_event") if fit_ok else np.nan,
            "Intercept": row.get("beta_intercept") if fit_ok else np.nan,
            "Intercept_stderror": row.get("se_intercept") if fit_ok else np.nan,
            "Transformation": log_transform,
            "N": n_obs,
            "PND": pnd_str,
            "TERM": n_terms,
            **term_dates,
        })

    if not rows:
        return pd.DataFrame(columns=_STATIC_COLUMNS)
    # Derive the column list from the highest TERM *number* seen across all
    # rows, not from n_terms (which is the count of TERMs per well).  The two
    # differ when TERM numbering is non-contiguous or doesn't start at 1 —
    # e.g. a well whose only non-null VAL data falls in TERM 4 has n_terms=1
    # but needs a TERM4_min/max column, not TERM1_min/max.
    max_term_num = 0
    for r in rows:
        for k in r:
            if k.startswith("TERM") and k.endswith("_min"):
                try:
                    max_term_num = max(max_term_num, int(k[4:-4]))
                except ValueError:
                    pass
    term_date_cols = [
        col
        for t in range(1, max_term_num + 1)
        for col in (f"TERM{t}_min", f"TERM{t}_max")
    ]
    return pd.DataFrame(rows, columns=_STATIC_COLUMNS + term_date_cols)


def _no_trend_row(
    well_name: str,
    xcoords: float,
    ycoords: float,
    dist_val: float,
    ou: str,
    system: str,
    model_label,
    analyte: str,
    mean_val: float,
    mcl: float,
    cut,
    log_transform: str,
    n_obs: int,
    pnd_str: str,
    lag_origin,
    n_terms: int,
    term_dates: dict,
) -> dict:
    """Shared structure for no-lag / not-performed rows."""
    mean_exceed = bool(mean_val > mcl) if pd.notna(mean_val) else np.nan
    return {
        "Name": well_name,
        "XCoords": xcoords,
        "YCoords": ycoords,
        "Distance": dist_val,
        "OU": ou,
        "System": system,
        "Model": model_label,
        "Lag": np.nan,
        "LAG_ORIGIN": lag_origin,
        "Analyte": analyte,
        "Trend": np.nan,
        "Mean": mean_val,
        "UCL": np.nan,
        "LCL": np.nan,
        "STD": mcl,
        "Mean_Exceed": mean_exceed,
        "UCL_Exceed": np.nan,
        "Trend_Break": cut,
        "Use_RiverStage": np.nan,
        "RiverStage_Coeff": np.nan,
        "RiverStage_Coeff_stderror": np.nan,
        "Date_Coeff": np.nan,
        "Date_Coeff_stderror": np.nan,
        "Intercept": np.nan,
        "Intercept_stderror": np.nan,
        "Transformation": log_transform,
        "N": n_obs,
        "PND": pnd_str,
        "TERM": n_terms,
        **term_dates,
    }
