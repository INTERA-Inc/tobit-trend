# Reviewed version: duplicate top-level definitions removed; last active definitions preserved.
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from scipy.special import log_ndtr
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Config
PRIOR_YEAR = 2023

CUTOFFS = {
    "DX": pd.Timestamp("2011-01-01"),
    "HX": pd.Timestamp("2011-11-01"),
    "KX_KW_KR4": pd.Timestamp("2009-04-01"),
}

KW_DATE1 = pd.Timestamp("2016-05-16")
KW_DATE2 = pd.Timestamp("2017-04-12")


# ----------------------------
# Helpers
# ----------------------------
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
# TERM logic (mirrors R)
# ----------------------------
def apply_system_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
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
    Mirrors the R loop that shifts TERM down if a term has zero non-NA VAL for a well.
    """
    df = df.copy()
    out = []

    for name, g in df.groupby("NAME", sort=False):
        g = g.copy()
        # consider terms that have data
        terms = sorted(g.loc[g["VAL"].notna(), "TERM"].unique().tolist())

        # R loops i=1..length(TM) and if LEN==0 then decrement later terms.
        # Equivalent: ensure each integer term from 1..max_term has at least one VAL, else shift down.
        max_term = int(g["TERM"].max()) if len(g) else 1
        for i in range(1, max_term + 1):
            xsub = g[(g["VAL"].notna()) & (g["TERM"] == i)]
            if len(xsub) == 0:
                g.loc[g["TERM"] > i, "TERM"] = g.loc[g["TERM"] > i, "TERM"] - 1

        out.append(g)

    return pd.concat(out, ignore_index=True)


def _parse_mdy_loose(series: pd.Series) -> pd.Series:
    """
    Parse mm/dd/YYYY with tolerance for junk like trailing characters.
    Returns datetime64[ns] floored to day; invalid -> NaT.
    """
    s = series.astype(str).str.strip()

    # Extract first mm/dd/yyyy occurrence if present
    # e.g. "01/02/20111" -> "01/02/2011"
    extracted = s.str.extract(r"(\d{1,2}/\d{1,2}/\d{4})", expand=False)

    dt = pd.to_datetime(extracted, format="%m/%d/%Y", errors="coerce")
    return dt.dt.floor("D")


def apply_manual_trend_breaks(
    df: pd.DataFrame, newtrends: pd.DataFrame
) -> pd.DataFrame:
    """
    NEWTRENDS: columns NAME, TREND, START, END (END can be NA => today)
    Applies: if EVENT in [START, END) then TERM = TREND
    """
    df = df.copy()
    nt = newtrends.copy()

    nt["NAME"] = nt["NAME"].astype(str)
    nt["TREND"] = pd.to_numeric(nt["TREND"], errors="coerce")

    nt["START"] = _parse_mdy_loose(nt["START"])
    nt["END"] = _parse_mdy_loose(nt["END"])

    # Match R behavior: END NA -> today
    today = pd.Timestamp.today().floor("D")
    nt["END"] = nt["END"].fillna(today)

    # If START is missing, skip that rule (can't apply)
    nt = nt[nt["START"].notna() & nt["TREND"].notna()].copy()

    for name, g_nt in nt.groupby("NAME"):
        m_name = df["NAME"].astype(str) == name
        if not m_name.any():
            continue

        for _, row in g_nt.iterrows():
            m = m_name & (df["EVENT"] >= row["START"]) & (df["EVENT"] < row["END"])
            df.loc[m, "TERM"] = int(row["TREND"])

    return df


def apply_kw_extra_terms(df: pd.DataFrame, kw: pd.DataFrame) -> pd.DataFrame:
    """
    For each WELL_NAME in KW list:
      TERM += 1 if EVENT >= 2016-05-16
      TERM += 1 if EVENT >= 2017-04-12
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
    Mimic:
      if well in names(WLLAG) and length(WL@SUM)>1 then ULAG=WL@LAG else NULL
    Using flat CSV approximation:
      use lag only if CLASS == Trend_Summary AND SUM_rows > 1 AND LAG not null
    """
    df = wl_trends_flat.copy()
    df["KEY"] = df["KEY"].astype(str)

    # In WLTrends_flat, KEY might be the well name directly (as in WLLAG[[NAME]])
    ok = (
        (df["CLASS"] == "Trend_Summary")
        & df["SUM_rows"].fillna(0).astype(float).gt(1)
        & df["LAG"].notna()
    )
    ulag = {k: int(v) for k, v in df.loc[ok, ["KEY", "LAG"]].values}

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
    chem = apply_system_cutoffs(chem)
    chem = compress_empty_terms_per_well(chem)
    chem = apply_manual_trend_breaks(chem, newtrends)
    chem = apply_kw_extra_terms(chem, kw)

    # NEWRS list (NoRS + RUM)
    newrs_names = set(no_rs["NAME"].astype(str).tolist()) | set(
        rum["NAME"].astype(str).tolist()
    )

    # ULAG map
    ulags = build_ulags(wl_trends)

    return chem, ulags, newrs_names


# ============================================================
# TREND MODELLING - Helpers
# ============================================================


def _bool_nds(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .isin({"TRUE", "T", "1", "Y", "YES"})
        .fillna(False)
    )


def _to_event_numeric(event: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(event).dt.floor("D")
    return (dt.astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.int64)


def _loess_residuals_rstyle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Approximate R residuals(loess(y ~ x)).
    We use LOWESS as the closest readily available analogue.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(y) < 4:
        return np.full(len(y), np.nan)

    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.full(len(y), np.nan)

    try:
        fit = lowess(y, x, frac=2 / 3, it=0, return_sorted=False)
        if not np.all(np.isfinite(fit)):
            return np.full(len(y), np.nan)
        return y - fit
    except Exception:
        return np.full(len(y), np.nan)


def crosscor_rstyle(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    lag: int,
) -> float:
    """
    Approximate sspaTrendAnalysis::crosscor
    - lag <- round(lag)
    - x.lag <- x1 - lag
    - x <- intersect(x2, x.lag)
    - matched by first exact index
    - detrend with loess
    - acf(..., lag.max=0) == ordinary correlation of matched residuals
    """
    lag = int(round(lag))

    x1 = np.asarray(x1, dtype=np.int64)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=np.int64)
    y2 = np.asarray(y2, dtype=float)

    ok1 = np.isfinite(y1)
    ok2 = np.isfinite(y2)
    x1, y1 = x1[ok1], y1[ok1]
    x2, y2 = x2[ok2], y2[ok2]

    if len(x1) < 4 or len(x2) < 4:
        return np.nan

    x_lag = x1 - lag
    x = np.intersect1d(x2, x_lag)

    if len(x) < 4:
        return np.nan

    # replicate which.max(u == x1/x2): first matching index
    idx1_map = {}
    for i, v in enumerate(x1):
        if v not in idx1_map:
            idx1_map[v] = i

    idx2_map = {}
    for i, v in enumerate(x2):
        if v not in idx2_map:
            idx2_map[v] = i

    i1 = np.array([idx1_map.get(u + lag, -1) for u in x], dtype=int)
    i2 = np.array([idx2_map.get(u, -1) for u in x], dtype=int)

    keep = (i1 >= 0) & (i2 >= 0)
    if keep.sum() < 4:
        return np.nan

    i1 = i1[keep]
    i2 = i2[keep]

    xx1 = x1[i1].astype(float)
    yy1 = y1[i1]
    xx2 = x2[i2].astype(float)
    yy2 = y2[i2]

    r1 = _loess_residuals_rstyle(xx1, yy1)
    r2 = _loess_residuals_rstyle(xx2, yy2)

    ok = np.isfinite(r1) & np.isfinite(r2)
    if ok.sum() < 4:
        return np.nan

    r1 = r1[ok]
    r2 = r2[ok]

    if np.nanstd(r1) < 1e-12 or np.nanstd(r2) < 1e-12:
        return np.nan

    return float(np.corrcoef(r1, r2)[0, 1])


def estimate_lag_from_series_rstyle(
    event: pd.Series,
    dep: pd.Series,
    cov: pd.Series,
    nds: pd.Series | None = None,
    max_lag: int = 90,
    min_n: int = 7,
    max_pnd: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """
    Approximate sspaTrendAnalysis::doLag:
      X_0 <- subset(x, !is.na(DEP))
      PNDS <- sum(NDS)/nrow(X_0)
      if n >= N and PNDS <= PND:
          x1 <- as.numeric(X_0$EVENT)
          x2 <- as.numeric(y$EVENT)
          y1 <- X_0[[DEP]]
          y2 <- y[[INDEP]]
          lags <- 0:MAXLAG
          acf <- sapply(...)
          lag <- ccf$lag[which(abs(ccf$acf) == max(abs(ccf$acf)))]
    """
    d = pd.DataFrame(
        {
            "EVENT": pd.to_datetime(event).dt.floor("D"),
            "DEP": pd.to_numeric(dep, errors="coerce"),
            "COV": pd.to_numeric(cov, errors="coerce"),
        }
    ).copy()

    if nds is None:
        d["NDS"] = False
    else:
        d["NDS"] = _bool_nds(pd.Series(nds))

    x0 = d[d["DEP"].notna()].copy()
    if len(x0) == 0:
        return np.nan, pd.DataFrame(columns=["lag", "acf"])

    pnds = x0["NDS"].mean()
    if len(x0) < min_n or pnds > max_pnd:
        return np.nan, pd.DataFrame(columns=["lag", "acf"])

    x1 = _to_event_numeric(x0["EVENT"])
    x2 = _to_event_numeric(d["EVENT"])
    y1 = x0["DEP"].to_numpy(dtype=float)
    y2 = d["COV"].to_numpy(dtype=float)

    rows = []
    for lag in range(0, int(max_lag) + 1):
        acf = crosscor_rstyle(x1, y1, x2, y2, lag=lag)
        rows.append({"lag": lag, "acf": acf})

    cod = pd.DataFrame(rows)
    valid = cod[cod["acf"].notna()].copy()
    if valid.empty:
        return np.nan, cod

    m = valid["acf"].abs().max()
    # R returns all ties; for Python model loop choose first/smallest lag deterministically
    best_lag = float(valid.loc[valid["acf"].abs() == m, "lag"].iloc[0])
    return best_lag, cod


def lag_col_rstyle(x, lag=0):
    """
    Match extracted R lagCol(X, LAG) exactly.

    R behavior:
      if LAG == 0 or NA: return X
      if LAG < 0: prepend NAs, then take X[1:(n+LAG)]
      if LAG > 0: drop first LAG rows, append NAs
    """
    s = pd.Series(x).copy()
    n = len(s)

    if pd.isna(lag) or int(lag) == 0:
        return s

    lag = int(lag)

    if abs(lag) >= n:
        return pd.Series([np.nan] * n, index=s.index)

    if lag < 0:
        k = -lag
        vals = [np.nan] * k + s.iloc[: n - k].tolist()
    else:
        k = lag
        vals = s.iloc[k:].tolist() + [np.nan] * k

    return pd.Series(vals, index=s.index)


def _standardize_nonconstant_columns(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    mu = np.zeros(X.shape[1], dtype=float)
    sd = np.ones(X.shape[1], dtype=float)
    Z = X.copy()

    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanstd(col) < 1e-12:
            Z[:, j] = col
            mu[j] = 0.0
            sd[j] = 1.0
        else:
            mu[j] = np.nanmean(col)
            sd[j] = np.nanstd(col)
            if sd[j] < 1e-12:
                sd[j] = 1.0
            Z[:, j] = (col - mu[j]) / sd[j]

    return Z, mu, sd


def _tobit_nll_left(params, y, X, cens, L):
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)

    if not np.isfinite(sigma) or sigma <= 0:
        return 1e100

    mu = X @ beta
    z_obs = (y - mu) / sigma
    z_cen = (L - mu) / sigma

    ll = np.empty_like(y, dtype=float)
    uu = ~cens
    cc = cens

    ll[uu] = norm.logpdf(z_obs[uu]) - log_sigma
    ll[cc] = log_ndtr(z_cen[cc])

    if not np.all(np.isfinite(ll)):
        return 1e100

    return float(-np.sum(ll))


def _fit_tobit_left(y, X, cens, L):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    cens = np.asarray(cens, dtype=bool)
    L = np.asarray(L, dtype=float)

    ok = np.isfinite(y) & np.isfinite(L) & np.all(np.isfinite(X), axis=1)
    y = y[ok]
    X = X[ok]
    cens = cens[ok]
    L = L[ok]

    n, p = X.shape
    if n <= p:
        return None

    y_mu = np.nanmean(y)
    y_sd = np.nanstd(y)
    if not np.isfinite(y_sd) or y_sd < 1e-12:
        return None

    yZ = (y - y_mu) / y_sd
    LZ = (L - y_mu) / y_sd
    XZ, X_mu, X_sd = _standardize_nonconstant_columns(X)

    uu = ~cens
    if uu.sum() >= p:
        try:
            beta0, *_ = np.linalg.lstsq(XZ[uu], yZ[uu], rcond=None)
            resid = yZ[uu] - XZ[uu] @ beta0
            sigma0 = max(np.nanstd(resid), 1e-2)
        except Exception:
            beta0 = np.zeros(p, dtype=float)
            sigma0 = max(np.nanstd(yZ), 1e-1)
    else:
        beta0 = np.zeros(p, dtype=float)
        sigma0 = max(np.nanstd(yZ), 1e-1)

    theta0 = np.r_[beta0, np.log(sigma0)]

    res = minimize(
        _tobit_nll_left,
        theta0,
        args=(yZ, XZ, cens, LZ),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 1000},
    )

    if (not res.success) or (not np.isfinite(res.fun)):
        return None

    betaZ = res.x[:-1]
    sigmaZ = np.exp(res.x[-1])

    beta = np.zeros_like(betaZ)
    for j in range(p):
        if np.nanstd(X[:, j]) < 1e-12:
            beta[j] = betaZ[j] * y_sd
        else:
            beta[j] = betaZ[j] * y_sd / X_sd[j]

    intercept_shift = 0.0
    for j in range(p):
        if np.nanstd(X[:, j]) >= 1e-12:
            intercept_shift += betaZ[j] * X_mu[j] / X_sd[j]

    if np.nanstd(X[:, 0]) < 1e-12:
        beta[0] = y_mu + y_sd * betaZ[0] - y_sd * intercept_shift

    sigma = sigmaZ * y_sd
    loglik = -float(res.fun)

    cov_beta = None
    try:
        Hinv = np.asarray(res.hess_inv)
        if Hinv.shape == (p + 1, p + 1) and np.all(np.isfinite(Hinv)):
            J = np.zeros((p, p + 1), dtype=float)
            for j in range(p):
                if np.nanstd(X[:, j]) < 1e-12:
                    J[j, j] = y_sd
                else:
                    J[j, j] = y_sd / X_sd[j]
            cov_beta = J @ Hinv @ J.T
    except Exception:
        cov_beta = None

    return {
        "beta": beta,
        "sigma": sigma,
        "loglik": loglik,
        "cov_beta": cov_beta,
        "n": n,
        "k_full": p + 1,  # includes sigma
        "success": True,
        "opt": res,
    }


def _tobit_nll_rstyle(params, y, X, left):
    """
    Match sspaTrendAnalysis::censReg.LL for left-censored data.
    y and X should already be standardized if you want to mimic R exactly.
    """
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)

    if not np.isfinite(sigma) or sigma <= 0:
        return 1e100

    mu = X @ beta
    r = (y - mu) / sigma

    ll = np.empty_like(y, dtype=float)

    left = np.asarray(left, dtype=bool)
    between = ~left

    ll[left] = log_ndtr(r[left])
    ll[between] = norm.logpdf(r[between]) - log_sigma

    if not np.all(np.isfinite(ll)):
        return 1e100

    return float(-np.sum(ll))


def _fit_tobit_rstyle(y, X, left):
    """
    Match custom sspaTrendAnalysis::censReg / censReg.LL:
    - response is log(VAL) from formula
    - left is logical censor indicator
    - y and X standardized before optimization
    - coefficients unstandardized after optimization
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    left = np.asarray(left, dtype=bool)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[ok]
    X = X[ok]
    left = left[ok]

    n, p = X.shape
    if n <= p:
        return None

    y_mu = np.nanmean(y)
    y_sd = np.nanstd(y)
    if not np.isfinite(y_sd) or y_sd < 1e-12:
        return None
    y0 = (y - y_mu) / y_sd

    X0, X_mu, X_sd = _standardize_nonconstant_columns(X)

    try:
        beta0, *_ = np.linalg.lstsq(X0, y0, rcond=None)
        resid = y0 - X0 @ beta0
        msr = np.mean(resid**2)
        log_sigma0 = np.log(msr) if msr > 0 else -300.0
    except Exception:
        beta0 = np.zeros(p, dtype=float)
        log_sigma0 = 0.0

    theta0 = np.r_[beta0, log_sigma0]

    res = minimize(
        _tobit_nll_rstyle,
        theta0,
        args=(y0, X0, left),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 1000},
    )

    if (not res.success) or (not np.isfinite(res.fun)):
        return None

    beta0_hat = res.x[:-1]
    log_sigma0_hat = res.x[-1]

    # unstandardize coefficients
    beta_hat = np.zeros_like(beta0_hat)
    for j in range(p):
        if np.nanstd(X[:, j]) < 1e-12:
            beta_hat[j] = beta0_hat[j] * y_sd
        else:
            beta_hat[j] = beta0_hat[j] * y_sd / X_sd[j]

    intercept_shift = 0.0
    for j in range(p):
        if np.nanstd(X[:, j]) >= 1e-12:
            intercept_shift += beta0_hat[j] * X_mu[j] / X_sd[j]

    if np.nanstd(X[:, 0]) < 1e-12:
        beta_hat[0] = y_mu + y_sd * beta0_hat[0] - y_sd * intercept_shift

    sigma_hat = np.exp(log_sigma0_hat) * y_sd
    loglik = -float(res.fun)

    cov_beta = None
    Hinv = None
    try:
        Hinv = np.asarray(res.hess_inv, dtype=float)
        if Hinv.shape == (p + 1, p + 1) and np.all(np.isfinite(Hinv)):
            J = np.zeros((p, p + 1), dtype=float)
            for j in range(p):
                if np.nanstd(X[:, j]) < 1e-12:
                    J[j, j] = y_sd
                else:
                    J[j, j] = y_sd / X_sd[j]

            cov_beta = J @ Hinv @ J.T

            # keep only finite covariance matrices
            if not np.all(np.isfinite(cov_beta)):
                cov_beta = None
    except Exception as e:
        print("cov_beta construction failed:", repr(e))
        cov_beta = None

    return {
        "beta": beta_hat,
        "sigma": sigma_hat,
        "loglik": loglik,
        "cov_beta": cov_beta,
        "n": n,
        "success": True,
        "opt": res,
    }


def _run_tobit_rstyle(
    df: pd.DataFrame,
    y_col: str,
    X_full: np.ndarray,
    X_null: np.ndarray,
    X_2: np.ndarray | None,
    X_3: np.ndarray | None,
    cens_col: str,
    min_n: int,
    max_pnd: float,
):
    x0 = df[df[y_col].notna()].copy()

    if "NDS" not in x0.columns:
        x0["NDS"] = False
    x0["NDS"] = _bool_nds(x0["NDS"])

    if len(x0) == 0:
        return {"CEN": None, "CEN_0": None, "CEN_2": None, "CEN_3": None}

    pnds = x0["NDS"].mean()
    if len(x0) < min_n or pnds > max_pnd:
        return {"CEN": None, "CEN_0": None, "CEN_2": None, "CEN_3": None}

    idx = x0.index.to_numpy()
    pos = df.index.get_indexer(idx)

    y = x0[y_col].to_numpy(dtype=float)
    left = x0[cens_col].to_numpy(dtype=bool)

    full = _fit_tobit_rstyle(y, X_full[pos], left)
    null = _fit_tobit_rstyle(y, X_null[pos], left)

    fit2 = _fit_tobit_rstyle(y, X_2[pos], left) if X_2 is not None else None
    fit3 = _fit_tobit_rstyle(y, X_3[pos], left) if X_3 is not None else None

    return {"CEN": full, "CEN_0": null, "CEN_2": fit2, "CEN_3": fit3}


# ============================================================
# Main model runner
# ============================================================


def plot_r_python_diagnostics(merged: pd.DataFrame):
    both = merged[merged["_merge"] == "both"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Lag comparison
    ax = axes[0, 0]
    if {"LAG_py", "LAG_r"}.issubset(both.columns):
        x = pd.to_numeric(both["LAG_r"], errors="coerce")
        y = pd.to_numeric(both["LAG_py"], errors="coerce")
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], alpha=0.6)
        if ok.any():
            lo = min(x[ok].min(), y[ok].min())
            hi = max(x[ok].max(), y[ok].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("R lag")
        ax.set_ylabel("Python lag")
        ax.set_title("Lag comparison")

    # 2) p_trend comparison
    ax = axes[0, 1]
    if {"p_trend_py", "p_trend_r"}.issubset(both.columns):
        x = pd.to_numeric(both["p_trend_r"], errors="coerce")
        y = pd.to_numeric(both["p_trend_py"], errors="coerce")
        ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], alpha=0.6)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("R p_trend")
        ax.set_ylabel("Python p_trend")
        ax.set_title("Trend p-value comparison")

    # 3) AIC diff histogram
    ax = axes[1, 0]
    if "AIC_diff" in both.columns:
        x = pd.to_numeric(both["AIC_diff"], errors="coerce").dropna()
        if len(x):
            ax.hist(x, bins=40)
        ax.set_title("AIC difference (Python - R)")
        ax.set_xlabel("AIC diff")

    # 4) BIC diff histogram
    ax = axes[1, 1]
    if "BIC_diff" in both.columns:
        x = pd.to_numeric(both["BIC_diff"], errors="coerce").dropna()
        if len(x):
            ax.hist(x, bins=40)
        ax.set_title("BIC difference (Python - R)")
        ax.set_xlabel("BIC diff")

    plt.tight_layout()
    plt.show()


def plot_lag_trace(lag_diag: pd.DataFrame, key: str):
    d = lag_diag[lag_diag["KEY"] == key].copy()
    if d.empty:
        raise ValueError(f"No lag diagnostics found for {key}")

    d = d.sort_values("lag")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(d["lag"], d["acf"], marker="o")
    ax1.set_xlabel("Lag (days)")
    ax1.set_ylabel("Correlation")
    ax1.set_title(f"Lag trace: {key}")

    best = d.loc[d["abs_acf"].idxmax()]
    ax1.axvline(best["lag"], linestyle="--")
    plt.tight_layout()
    plt.show()


# ============================================================
# Plotting
# ============================================================


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _nd_summary_for_well(chem_rs: pd.DataFrame, well: str) -> pd.DataFrame:
    df = chem_rs.loc[chem_rs["NAME"] == well].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["ITER", "n_total_obs", "n_nd", "pct_nd", "nd_rule_ok"]
        )

    df["VAL"] = pd.to_numeric(df["VAL"], errors="coerce")
    df["NDS"] = _bool_nds_plot(df["NDS"]) if "NDS" in df.columns else False

    # mirror modelling eligibility: only rows with usable concentration values
    df = df[df["VAL"].notna()].copy()

    out = (
        df.groupby("TERM", as_index=False)
        .agg(
            n_total_obs=("VAL", "size"),
            n_nd=("NDS", "sum"),
        )
        .rename(columns={"TERM": "ITER"})
    )

    out["pct_nd"] = np.where(
        out["n_total_obs"] > 0,
        out["n_nd"] / out["n_total_obs"],
        np.nan,
    )
    out["nd_rule_ok"] = np.where(out["pct_nd"] < 0.5, "PASS", "FAIL")
    return out


def _bool_nds_plot(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .isin({"TRUE", "T", "1", "Y", "YES", "U"})
        .fillna(False)
    )


def _to_event_numeric_rstyle_plot(event: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(event).dt.floor("D")
    return (dt.astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)


def _apply_lag_for_plot(series: pd.Series, lag) -> pd.Series:
    """
    R doTobit applies lagCol(col, -LAG).
    Net effect for positive selected lag is shift(-lag).
    """
    s = pd.Series(series).copy()
    if lag is None or pd.isna(lag):
        return s
    lag = int(round(float(lag)))
    if lag == 0:
        return s
    return s.shift(-lag)


def _build_term_predictions_from_results(
    df_well: pd.DataFrame,
    results_df: pd.DataFrame,
    source_label: str,
    use_interp_col: str = "INTERP",
) -> pd.DataFrame:
    """
    Build fitted values from result coefficients already reported on the R scale:
      yhat_log = beta_intercept + beta_interp * INTERP_LAG + beta_event * EVENT_NUM_R
    where EVENT_NUM_R is raw R-style Date numeric.
    """
    out = []
    df_well = df_well.copy().sort_values(["TERM", "EVENT"])

    if results_df is None or len(results_df) == 0:
        return pd.DataFrame(
            columns=[
                "NAME",
                "TERM",
                "EVENT",
                "EVENT_NUM_R",
                "INTERP_LAG",
                "yhat_log",
                "yhat",
                "source",
            ]
        )

    for term, g in df_well.groupby("TERM", sort=True):
        g = g.copy().sort_values("EVENT")
        row = results_df[
            (results_df["WELL"] == g["NAME"].iloc[0])
            & (pd.to_numeric(results_df["ITER"], errors="coerce") == int(term))
        ]

        if row.empty:
            continue

        row = row.iloc[0]
        b0 = row.get("beta_intercept", np.nan)
        b1 = row.get("beta_interp", np.nan)
        b2 = row.get("beta_event", np.nan)
        lag = row.get("LAG", np.nan)
        model_type = row.get("model_type", None)

        if pd.isna(b0) or pd.isna(b2):
            continue

        g["EVENT_NUM_R"] = _to_event_numeric_rstyle_plot(g["EVENT"])

        # infer model type if absent
        if model_type is None or pd.isna(model_type):
            model_type = "INTERP+EVENT" if pd.notna(b1) else "EVENT"

        if model_type == "INTERP+EVENT" and pd.notna(b1):
            g["INTERP_LAG"] = _apply_lag_for_plot(
                pd.to_numeric(g[use_interp_col], errors="coerce"), lag
            )
            g = g[g["INTERP_LAG"].notna()].copy()
            if g.empty:
                continue

            g["yhat_log"] = (
                b0
                + b1 * g["INTERP_LAG"].to_numpy(dtype=float)
                + b2 * g["EVENT_NUM_R"].to_numpy(dtype=float)
            )
        else:
            g["INTERP_LAG"] = np.nan
            g["yhat_log"] = b0 + b2 * g["EVENT_NUM_R"].to_numpy(dtype=float)

        g["yhat"] = np.exp(g["yhat_log"])
        g["source"] = source_label
        out.append(
            g[
                [
                    "NAME",
                    "TERM",
                    "EVENT",
                    "EVENT_NUM_R",
                    "INTERP_LAG",
                    "yhat_log",
                    "yhat",
                    "source",
                ]
            ]
        )

    if not out:
        return pd.DataFrame(
            columns=[
                "NAME",
                "TERM",
                "EVENT",
                "EVENT_NUM_R",
                "INTERP_LAG",
                "yhat_log",
                "yhat",
                "source",
            ]
        )

    return pd.concat(out, ignore_index=True)


def _summary_table_for_well_compare(
    well: str,
    chem_rs: pd.DataFrame,
    results_py: pd.DataFrame,
    results_r: pd.DataFrame | None = None,
) -> pd.DataFrame:
    py = results_py.loc[results_py["WELL"] == well].copy()
    py = py.sort_values("ITER")

    keep_py = [
        c
        for c in [
            "KEY",
            "WELL",
            "ITER",
            "LAG",
            "p_trend",
            "p_interp",
            "p_event",
            "n_obs",
        ]
        if c in py.columns
    ]
    py = py[keep_py].copy()
    py = py.rename(
        columns={
            "LAG": "LAG_py",
            "p_trend": "p_trend_py",
            "p_interp": "p_interp_py",
            "p_event": "p_event_py",
            "n_obs": "n_obs_py",
        }
    )

    if results_r is not None:
        r = results_r.loc[results_r["WELL"] == well].copy()
        r = r.sort_values("ITER")
        keep_r = [
            c
            for c in [
                "KEY",
                "WELL",
                "ITER",
                "LAG",
                "p_trend",
                "p_interp",
                "p_event",
                "n_obs",
            ]
            if c in r.columns
        ]
        r = r[keep_r].copy()
        r = r.rename(
            columns={
                "LAG": "LAG_r",
                "p_trend": "p_trend_r",
                "p_interp": "p_interp_r",
                "p_event": "p_event_r",
                "n_obs": "n_obs_r",
            }
        )
        tab = py.merge(r, on=["KEY", "WELL", "ITER"], how="outer")
    else:
        tab = py

    nd = _nd_summary_for_well(chem_rs, well)
    tab = tab.merge(nd, on="ITER", how="left")

    return tab.sort_values("ITER").reset_index(drop=True)


def _draw_summary_table_compare(ax, table_df: pd.DataFrame):
    ax.axis("off")

    if table_df.empty:
        ax.text(0.01, 0.95, "No summary available", va="top", ha="left")
        return

    show_cols = [
        c
        for c in [
            "ITER",
            "n_total_obs",
            "n_nd",
            "pct_nd",
            "nd_rule_ok",
            "LAG_py",
            "LAG_r",
            "p_trend_py",
            "p_trend_r",
            "p_interp_py",
            "p_interp_r",
            "p_event_py",
            "p_event_r",
            "n_obs_py",
            "n_obs_r",
        ]
        if c in table_df.columns
    ]

    disp = table_df[show_cols].copy()

    for c in disp.columns:
        if c.startswith("p_"):
            disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
        elif c == "pct_nd":
            disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{100*x:.0f}%")
        elif c.startswith("LAG"):
            disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{int(round(x))}")
        elif c.startswith("n_"):
            disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{int(round(x))}")

    tbl = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)


def plot_well_trend_report(
    well: str,
    chem_rs: pd.DataFrame,
    results_py: pd.DataFrame,
    results_r: pd.DataFrame | None = None,
    gwl_df: pd.DataFrame | None = None,
    gwl_well_col: str = "NAME",
    gwl_date_col: str = "EVENT",
    gwl_value_col: str = "GWL",
    figsize: tuple = (15, 10),
    log_y: bool = True,
    savepath: str | None = None,
):
    """
    Plot well report with Python and R trend fits side by side.

    Required:
      chem_rs
      results_py

    Optional:
      results_r
      gwl_df
    """
    df = chem_rs.loc[chem_rs["NAME"] == well].copy()
    if df.empty:
        raise ValueError(f"No chemistry data found for well: {well}")

    if isinstance(results_r, str):
        results_r = pd.read_csv(results_r)

    df["EVENT"] = pd.to_datetime(df["EVENT"])
    df["VAL"] = pd.to_numeric(df["VAL"], errors="coerce")
    df["INTERP"] = pd.to_numeric(df["INTERP"], errors="coerce")
    df["NDS"] = _bool_nds_plot(df["NDS"]) if "NDS" in df.columns else False
    df = df.sort_values(["TERM", "EVENT"]).copy()

    pred_py = _build_term_predictions_from_results(
        df, results_py, source_label="Python"
    )
    pred_r = (
        _build_term_predictions_from_results(df, results_r, source_label="R")
        if results_r is not None
        else pd.DataFrame()
    )

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2.4, 1.5, 1.3], hspace=0.25, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    # --------------------------------------------------------
    # Panel 1: Chromium observations + Python/R fits
    # --------------------------------------------------------
    det = df.loc[df["VAL"].notna() & (~df["NDS"])].copy()
    nd = df.loc[df["VAL"].notna() & (df["NDS"])].copy()

    if not det.empty:
        ax1.scatter(
            det["EVENT"],
            det["VAL"],
            label="Observed detect",
            s=24,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    if not nd.empty:
        ax1.scatter(
            nd["EVENT"],
            nd["VAL"],
            label="Observed non-detect",
            s=52,
            marker="v",
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            zorder=4,
        )

    if not pred_py.empty:
        for term, g in pred_py.groupby("TERM", sort=True):
            sort_cols = ["EVENT"] + (
                ["_src_order"] if "_src_order" in g.columns else []
            )
            g = g.sort_values(sort_cols, kind="mergesort").copy()
            ax1.plot(
                g["EVENT"], g["yhat"], linewidth=2, label=f"Python fit T{int(term)}"
            )

    if not pred_r.empty:
        for term, g in pred_r.groupby("TERM", sort=True):
            sort_cols = ["EVENT"] + (
                ["_src_order"] if "_src_order" in g.columns else []
            )
            g = g.sort_values(sort_cols, kind="mergesort").copy()
            ax1.plot(
                g["EVENT"],
                g["yhat"],
                linewidth=2,
                linestyle="--",
                label=f"R fit T{int(term)}",
            )

    term_starts = (
        df.groupby("TERM", as_index=False)["EVENT"]
        .min()
        .sort_values("EVENT")["EVENT"]
        .tolist()
    )
    for b in term_starts[1:]:
        ax1.axvline(pd.to_datetime(b), linestyle="--", linewidth=1)

    nd_tab = _nd_summary_for_well(chem_rs, well)
    if not nd_tab.empty:
        txt = "\n".join(
            [
                f"T{int(r.ITER)}: ND {int(r.n_nd)}/{int(r.n_total_obs)} ({100*r.pct_nd:.0f}%) {r.nd_rule_ok}"
                for _, r in nd_tab.iterrows()
            ]
        )
        ax1.text(
            0.01,
            0.99,
            txt,
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

    ax1.set_title(f"{well} — Python vs R Chromium trend comparison")
    ax1.set_ylabel("Chromium concentration")
    if log_y:
        pos = df["VAL"].dropna()
        if len(pos) > 0 and (pos > 0).all():
            ax1.set_yscale("log")
    ax1.legend(loc="best", fontsize=8, ncol=2)

    # --------------------------------------------------------
    # Panel 2: River Stage + optional GWL
    # --------------------------------------------------------
    rs = df[["EVENT", "INTERP"]].dropna().drop_duplicates().sort_values("EVENT")
    if not rs.empty:
        ax2.plot(rs["EVENT"], rs["INTERP"], label="River Stage", linewidth=1.8)

    if gwl_df is not None:
        g = gwl_df.copy()
        g[gwl_date_col] = pd.to_datetime(g[gwl_date_col])
        g = g.loc[g[gwl_well_col] == well].copy()
        g[gwl_value_col] = pd.to_numeric(g[gwl_value_col], errors="coerce")
        g = g.dropna(subset=[gwl_date_col, gwl_value_col]).sort_values(gwl_date_col)
        if not g.empty:
            ax2.plot(
                g[gwl_date_col], g[gwl_value_col], label="Observed GWL", linewidth=1.5
            )

    for b in term_starts[1:]:
        ax2.axvline(pd.to_datetime(b), linestyle="--", linewidth=1)

    ax2.set_ylabel("Hydrology")
    ax2.legend(loc="best", fontsize=8)

    # --------------------------------------------------------
    # Panel 3: Comparison table
    # --------------------------------------------------------
    tab = _summary_table_for_well_compare(
        well,
        chem_rs=chem_rs,
        results_py=results_py,
        results_r=results_r,
    )
    _draw_summary_table_compare(ax3, tab)

    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.autofmt_xdate()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig


# ------------------------------------------------------------
# Batch plotting
# ------------------------------------------------------------


def plot_many_wells(
    wells: list[str],
    chem_rs: pd.DataFrame,
    results_py: pd.DataFrame,
    results_r: pd.DataFrame | None = None,
    gwl_df: pd.DataFrame | None = None,
    out_dir: str | None = None,
):
    figs = {}
    for well in wells:
        savepath = None if out_dir is None else f"{out_dir}/{well}_trend_report.png"
        fig = plot_well_trend_report(
            well=well,
            chem_rs=chem_rs,
            results_py=results_py,
            results_r=results_r,
            gwl_df=gwl_df,
            savepath=savepath,
        )
        figs[well] = fig
    return figs


# ============================
# Patched overrides (2026-03-25)
# ============================


def _to_event_numeric_rstyle(event: pd.Series) -> np.ndarray:
    """
    Mimic R as.numeric(Date): whole days since 1970-01-01.
    """
    dt = pd.to_datetime(event).dt.floor("D")
    return (dt.astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)


def _wald_pvalue(beta_idx: int, fit: dict) -> float:
    cov_beta = fit.get("cov_beta", None)
    beta = fit.get("beta", None)
    if cov_beta is None or beta is None:
        return np.nan
    if beta_idx >= len(beta):
        return np.nan
    var = cov_beta[beta_idx, beta_idx]
    if not np.isfinite(var) or var <= 0:
        return np.nan
    z = beta[beta_idx] / np.sqrt(var)
    return float(2 * norm.sf(abs(z)))


def _extract_model_rstyle(df: pd.DataFrame, fits: dict, indep_len: int, lag: float):
    """
    Approximate extractModel() for fields needed in flat output.
    Includes coefficient-level Wald p-values for INTERP and EVENT.
    """
    full = fits.get("CEN")
    null = fits.get("CEN_0")
    fit2 = fits.get("CEN_2")
    fit3 = fits.get("CEN_3")

    if full is None:
        return None

    n = int(full["n"])

    if indep_len == 2:
        D = 2 * (full["loglik"] - null["loglik"]) if null is not None else np.nan
        p_trend = float(1 - chi2.cdf(D, 2)) if np.isfinite(D) and D >= 0 else np.nan

        return {
            "LAG": lag,
            "p_trend": p_trend,
            "AIC": 2 * (len(full["beta"]) - 1) - 2 * full["loglik"],
            "BIC": np.log(n) * (len(full["beta"]) - 1) - 2 * full["loglik"],
            "AIC_null": (
                2 * len(null["beta"]) - 2 * null["loglik"]
                if null is not None
                else np.nan
            ),
            "BIC_null": (
                np.log(n) * len(null["beta"]) - 2 * null["loglik"]
                if null is not None
                else np.nan
            ),
            "AIC_cov1": (
                2 * (len(fit2["beta"]) - 1) - 2 * fit2["loglik"]
                if fit2 is not None
                else np.nan
            ),
            "BIC_cov1": (
                np.log(n) * (len(fit2["beta"]) - 1) - 2 * fit2["loglik"]
                if fit2 is not None
                else np.nan
            ),
            "AIC_cov2": (
                2 * (len(fit3["beta"]) - 1) - 2 * fit3["loglik"]
                if fit3 is not None
                else np.nan
            ),
            "BIC_cov2": (
                np.log(n) * (len(fit3["beta"]) - 1) - 2 * fit3["loglik"]
                if fit3 is not None
                else np.nan
            ),
            "beta": full["beta"],
            "sigma": full["sigma"],
            "logLik": full["loglik"],
            "n_obs": n,
            "p_interp": _wald_pvalue(1, full),
            "p_event": _wald_pvalue(2, full),
        }

    D = 2 * (full["loglik"] - null["loglik"]) if null is not None else np.nan
    p_trend = float(1 - chi2.cdf(D, 1)) if np.isfinite(D) and D >= 0 else np.nan

    return {
        "LAG": lag,
        "p_trend": p_trend,
        "AIC": 2 * (len(full["beta"]) - 1) - 2 * full["loglik"],
        "BIC": np.log(n) * (len(full["beta"]) - 1) - 2 * full["loglik"],
        "AIC_null": (
            2 * len(null["beta"]) - 2 * null["loglik"] if null is not None else np.nan
        ),
        "BIC_null": (
            np.log(n) * len(null["beta"]) - 2 * null["loglik"]
            if null is not None
            else np.nan
        ),
        "beta": full["beta"],
        "sigma": full["sigma"],
        "logLik": full["loglik"],
        "n_obs": n,
        "p_interp": np.nan,
        "p_event": _wald_pvalue(1, full),
    }


def run_script04_models(
    chem_rs: pd.DataFrame,
    ulags: dict,
    newrs_names: set,
    max_lag: int = 90,
    min_n: int = 7,
    max_pnd: float = 1.0,
    lag_debug: bool = False,
):
    """
    R-aligned Script 04 model runner.

    Important:
    - lag selection is R-style (date-based cross-correlation)
    - lag application is R-style row shift via lagCol(..., -LAG)
    - fit uses stable EVENT scale: years since start of term
    - reported beta_event / beta_intercept are converted back to R Date scale
    """
    rows = []
    lag_rows = []

    for well, g_well in chem_rs.groupby("NAME", sort=True):
        g_well = g_well.sort_values("EVENT").copy()
        g_well["EVENT"] = pd.to_datetime(g_well["EVENT"]).dt.floor("D")
        use_rs = well not in newrs_names
        ulag = ulags.get(well, None)

        for term, g_term in g_well.groupby("TERM", sort=True):
            g_term = (
                g_term.sort_values(["EVENT", "_src_order"], kind="mergesort")
                .reset_index(drop=True)
                .copy()
            )
            key = f"{well}__ITER{int(term)}"

            g_term["VAL"] = pd.to_numeric(g_term["VAL"], errors="coerce")
            g_term["INTERP"] = pd.to_numeric(g_term["INTERP"], errors="coerce")
            g_term["NDS"] = _bool_nds(g_term["NDS"]) if "NDS" in g_term else False

            # Any remaining <=0 cannot be logged.
            g_term.loc[g_term["VAL"] <= 0, "VAL"] = np.nan
            g_term["Y"] = np.log(g_term["VAL"])

            lag = np.nan
            lag_source = "NONE"
            cod = pd.DataFrame()

            if use_rs:
                if ulag is None:
                    lag, cod = estimate_lag_from_series_rstyle(
                        event=g_term["EVENT"],
                        dep=g_term["Y"],
                        cov=g_term["INTERP"],
                        nds=g_term["NDS"],
                        max_lag=max_lag,
                        min_n=min_n,
                        max_pnd=max_pnd,
                    )
                    lag_source = "FALLBACK" if pd.notna(lag) else "NONE"
                elif pd.notna(ulag) and float(ulag) > 0:
                    _tmp_lag, cod = estimate_lag_from_series_rstyle(
                        event=g_term["EVENT"],
                        dep=g_term["Y"],
                        cov=g_term["INTERP"],
                        nds=g_term["NDS"],
                        max_lag=max_lag,
                        min_n=min_n,
                        max_pnd=max_pnd,
                    )
                    lag = int(round(float(ulag)))
                    lag_source = "ULAG"
                else:
                    lag = 0
                    lag_source = "ULAG"

                if pd.isna(lag):
                    rows.append(
                        {
                            "KEY": key,
                            "WELL": well,
                            "ITER": int(term),
                            "LAG": np.nan,
                            "lag_source": lag_source,
                            "p_trend": np.nan,
                            "p_interp": np.nan,
                            "p_event": np.nan,
                            "AIC": np.nan,
                            "BIC": np.nan,
                            "logLik": np.nan,
                            "n_obs": int(g_term["Y"].notna().sum()),
                            "n_cens": int(g_term.loc[g_term["Y"].notna(), "NDS"].sum()),
                            "model_type": "INTERP+EVENT",
                            "fit_ok": False,
                        }
                    )
                    if lag_debug and not cod.empty:
                        tmp = cod.copy()
                        tmp["KEY"] = key
                        tmp["WELL"] = well
                        tmp["ITER"] = int(term)
                        lag_rows.append(tmp)
                    continue

                dt = g_term.copy()
                if lag > 0:
                    dt["INTERP_LAG"] = lag_col_rstyle(dt["INTERP"], -int(lag))
                    dt = dt[dt["INTERP_LAG"].notna()].copy()
                else:
                    dt["INTERP_LAG"] = dt["INTERP"]

                # Stable fit scale
                dt["EVENT_NUM"] = (
                    pd.to_datetime(dt["EVENT"]) - pd.to_datetime(dt["EVENT"]).min()
                ).dt.days / 365.25

                X_full = np.column_stack(
                    [
                        np.ones(len(dt)),
                        dt["INTERP_LAG"].to_numpy(dtype=float),
                        dt["EVENT_NUM"].to_numpy(dtype=float),
                    ]
                )
                X_null = np.ones((len(dt), 1), dtype=float)
                X_2 = np.column_stack(
                    [np.ones(len(dt)), dt["INTERP_LAG"].to_numpy(dtype=float)]
                )
                X_3 = np.column_stack(
                    [np.ones(len(dt)), dt["EVENT_NUM"].to_numpy(dtype=float)]
                )

                fits = _run_tobit_rstyle(
                    dt,
                    "Y",
                    X_full,
                    X_null,
                    X_2,
                    X_3,
                    cens_col="NDS",
                    min_n=min_n,
                    max_pnd=max_pnd,
                )
                model = _extract_model_rstyle(dt, fits, indep_len=2, lag=lag)

                if model is None:
                    rows.append(
                        {
                            "KEY": key,
                            "WELL": well,
                            "ITER": int(term),
                            "LAG": lag,
                            "lag_source": lag_source,
                            "p_trend": np.nan,
                            "p_interp": np.nan,
                            "p_event": np.nan,
                            "AIC": np.nan,
                            "BIC": np.nan,
                            "logLik": np.nan,
                            "n_obs": int(dt["Y"].notna().sum()),
                            "n_cens": int(dt.loc[dt["Y"].notna(), "NDS"].sum()),
                            "model_type": "INTERP+EVENT",
                            "fit_ok": False,
                        }
                    )
                else:
                    # Convert coefficients from stable time scale (years since term start)
                    # to R reporting scale (raw Date day number)
                    event0_days = _to_event_numeric_rstyle(dt["EVENT"]).min()
                    beta_intercept_r = model["beta"][0] - model["beta"][2] * (
                        event0_days / 365.25
                    )
                    beta_event_r = model["beta"][2] / 365.25

                    rows.append(
                        {
                            "KEY": key,
                            "WELL": well,
                            "ITER": int(term),
                            "LAG": lag,
                            "lag_source": lag_source,
                            "p_trend": model.get("p_trend", np.nan),
                            "p_interp": model.get("p_interp", np.nan),
                            "p_event": model.get("p_event", np.nan),
                            "AIC": model.get("AIC", np.nan),
                            "BIC": model.get("BIC", np.nan),
                            "logLik": model.get("logLik", np.nan),
                            "n_obs": model.get("n_obs", np.nan),
                            "n_cens": int(dt.loc[dt["Y"].notna(), "NDS"].sum()),
                            "model_type": "INTERP+EVENT",
                            "beta_intercept": beta_intercept_r,
                            "beta_interp": model["beta"][1],
                            "beta_event": beta_event_r,
                            "sigma": model["sigma"],
                            "fit_ok": True,
                        }
                    )

            else:
                dt = g_term.copy()
                # Stable fit scale
                dt["EVENT_NUM"] = (
                    pd.to_datetime(dt["EVENT"]) - pd.to_datetime(dt["EVENT"]).min()
                ).dt.days / 365.25

                X_full = np.column_stack(
                    [np.ones(len(dt)), dt["EVENT_NUM"].to_numpy(dtype=float)]
                )
                X_null = np.ones((len(dt), 1), dtype=float)

                fits = _run_tobit_rstyle(
                    dt,
                    "Y",
                    X_full,
                    X_null,
                    None,
                    None,
                    cens_col="NDS",
                    min_n=min_n,
                    max_pnd=max_pnd,
                )
                model = _extract_model_rstyle(dt, fits, indep_len=1, lag=0)

                if model is None:
                    rows.append(
                        {
                            "KEY": key,
                            "WELL": well,
                            "ITER": int(term),
                            "LAG": 0,
                            "lag_source": "NONE",
                            "p_trend": np.nan,
                            "p_interp": np.nan,
                            "p_event": np.nan,
                            "AIC": np.nan,
                            "BIC": np.nan,
                            "logLik": np.nan,
                            "n_obs": int(dt["Y"].notna().sum()),
                            "n_cens": int(dt.loc[dt["Y"].notna(), "NDS"].sum()),
                            "model_type": "EVENT",
                            "fit_ok": False,
                        }
                    )
                else:
                    event0_days = _to_event_numeric_rstyle(dt["EVENT"]).min()
                    beta_intercept_r = model["beta"][0] - model["beta"][1] * (
                        event0_days / 365.25
                    )
                    beta_event_r = model["beta"][1] / 365.25

                    rows.append(
                        {
                            "KEY": key,
                            "WELL": well,
                            "ITER": int(term),
                            "LAG": 0,
                            "lag_source": "NONE",
                            "p_trend": model.get("p_trend", np.nan),
                            "p_interp": np.nan,
                            "p_event": model.get("p_event", np.nan),
                            "AIC": model.get("AIC", np.nan),
                            "BIC": model.get("BIC", np.nan),
                            "logLik": model.get("logLik", np.nan),
                            "n_obs": model.get("n_obs", np.nan),
                            "n_cens": int(dt.loc[dt["Y"].notna(), "NDS"].sum()),
                            "model_type": "EVENT",
                            "beta_intercept": beta_intercept_r,
                            "beta_interp": np.nan,
                            "beta_event": beta_event_r,
                            "sigma": model["sigma"],
                            "fit_ok": True,
                        }
                    )

            if lag_debug and not cod.empty:
                tmp = cod.copy()
                tmp["KEY"] = key
                tmp["WELL"] = well
                tmp["ITER"] = int(term)
                lag_rows.append(tmp)

    results = pd.DataFrame(rows)
    if lag_debug:
        lag_diag = (
            pd.concat(lag_rows, ignore_index=True) if lag_rows else pd.DataFrame()
        )
        return results, lag_diag
    return results


def compare_r_python_results(
    py_results: pd.DataFrame,
    r_csv_path: Optional[str] = None,
    out_prefix: Optional[str] = None,
    r_df: Optional[pd.DataFrame] = None,
):
    """
    Compare Python results to R diagnostics CSV or DataFrame.
    """
    if r_df is not None:
        r = r_df.copy()
    elif r_csv_path is not None:
        r = pd.read_csv(r_csv_path).copy()
    else:
        raise ValueError("Provide r_csv_path or r_df")

    p = py_results.copy()
    r["KEY"] = r["KEY"].astype(str)
    p["KEY"] = p["KEY"].astype(str)

    merged = p.merge(
        r,
        on="KEY",
        how="outer",
        suffixes=("_py", "_r"),
        indicator=True,
    )

    compare_cols = [
        "LAG",
        "p_trend",
        "AIC",
        "BIC",
        "logLik",
        "beta_intercept",
        "beta_interp",
        "beta_event",
        "p_interp",
        "p_event",
        "n_obs",
    ]

    for col in compare_cols:
        pyc = f"{col}_py"
        rc = f"{col}_r"
        if pyc in merged.columns and rc in merged.columns:
            merged[f"{col}_diff"] = pd.to_numeric(
                merged[pyc], errors="coerce"
            ) - pd.to_numeric(merged[rc], errors="coerce")

    both = merged[merged["_merge"] == "both"].copy()

    summary = {
        "n_python": int(len(p)),
        "n_r": int(len(r)),
        "merge_counts": merged["_merge"].value_counts(dropna=False).to_dict(),
    }

    if {"LAG_py", "LAG_r"}.issubset(both.columns):
        lag_ok = both["LAG_py"].notna() & both["LAG_r"].notna()
        summary["lag_n"] = int(lag_ok.sum())
        summary["lag_match_rate"] = (
            float((both.loc[lag_ok, "LAG_py"] == both.loc[lag_ok, "LAG_r"]).mean())
            if lag_ok.sum()
            else np.nan
        )

    for col in compare_cols:
        dcol = f"{col}_diff"
        if dcol in both.columns:
            vals = pd.to_numeric(both[dcol], errors="coerce")
            summary[f"{col}_mae"] = (
                float(np.nanmean(np.abs(vals))) if vals.notna().any() else np.nan
            )

    if out_prefix:
        merged.to_csv(f"{out_prefix}_merged_comparison.csv", index=False)

    return merged, summary


def _build_term_predictions(
    df_well: pd.DataFrame,
    results_df: pd.DataFrame,
    use_interp_col: str = "INTERP",
) -> pd.DataFrame:
    """
    Build fitted values per well-term using reported coefficients.
    Reported coefficients are on R raw-Date scale, so convert back to the
    stable fit scale before predicting.
    """
    out = []
    df_well = df_well.copy().sort_values(["TERM", "EVENT"])

    for term, g in df_well.groupby("TERM", sort=True):
        g = g.copy().sort_values("EVENT")
        row = results_df[
            (results_df["WELL"] == g["NAME"].iloc[0])
            & (results_df["ITER"] == int(term))
        ]

        if row.empty:
            continue

        row = row.iloc[0]
        if pd.isna(row.get("beta_intercept", np.nan)):
            continue

        lag = row.get("LAG", np.nan)
        model_type = row.get("model_type", "INTERP+EVENT")

        # Stable fit scale
        g["EVENT_NUM"] = (
            pd.to_datetime(g["EVENT"]) - pd.to_datetime(g["EVENT"]).min()
        ).dt.days / 365.25

        # Convert reported R-scale coefficients back to stable fit scale
        event0_days = _to_event_numeric_rstyle(g["EVENT"]).min()
        b0_r = row["beta_intercept"]
        b2_r = row.get("beta_event", np.nan)
        if pd.isna(b2_r):
            continue
        b2_fit = b2_r * 365.25
        b0_fit = b0_r + b2_fit * (event0_days / 365.25)

        if model_type == "INTERP+EVENT":
            g["INTERP_LAG"] = lag_col_rstyle(
                pd.to_numeric(g[use_interp_col], errors="coerce"),
                -lag if pd.notna(lag) and float(lag) > 0 else 0,
            )
            g = g[g["INTERP_LAG"].notna()].copy()
            if g.empty:
                continue

            b1 = row.get("beta_interp", np.nan)
            if pd.isna(b1):
                continue

            g["yhat_log"] = (
                b0_fit
                + b1 * g["INTERP_LAG"].to_numpy(dtype=float)
                + b2_fit * g["EVENT_NUM"].to_numpy(dtype=float)
            )
        else:
            g["INTERP_LAG"] = np.nan
            g["yhat_log"] = b0_fit + b2_fit * g["EVENT_NUM"].to_numpy(dtype=float)

        g["yhat"] = np.exp(g["yhat_log"])
        out.append(
            g[["NAME", "TERM", "EVENT", "EVENT_NUM", "INTERP_LAG", "yhat_log", "yhat"]]
        )

    if not out:
        return pd.DataFrame(
            columns=[
                "NAME",
                "TERM",
                "EVENT",
                "EVENT_NUM",
                "INTERP_LAG",
                "yhat_log",
                "yhat",
            ]
        )

    return pd.concat(out, ignore_index=True)
