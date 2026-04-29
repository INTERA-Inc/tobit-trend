from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import math
import os
import tempfile
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t


# --------------------------------------------------------------------------------------
# R S4 class ports
# --------------------------------------------------------------------------------------


@dataclass
class CovariateModel:
    MODEL: str
    NAME: str
    LAG: Union[float, np.ndarray]
    SUM: Any
    vcov: Any
    AIC: pd.DataFrame
    BIC: pd.DataFrame
    df: Union[float, np.ndarray]
    RES: Union[np.ndarray, List[float]]
    PRED: Union[np.ndarray, List[float]]
    p_trend: Union[float, np.ndarray]


@dataclass
class TrendSummary:
    MODEL: str
    FORM: str
    LOG: str
    NAME: str
    LAG: Union[float, np.ndarray]
    TS: Any
    MINDATE: pd.Timestamp
    SUM: Any
    vcov: Any
    AIC: pd.DataFrame
    BIC: pd.DataFrame
    df: Union[float, np.ndarray]
    RES: Union[np.ndarray, List[float]]
    PRED: Union[np.ndarray, List[float]]
    p_trend: Union[float, np.ndarray]
    COD: pd.DataFrame
    DATA: pd.DataFrame


# --------------------------------------------------------------------------------------
# Small R-like helpers
# --------------------------------------------------------------------------------------


def match_arg(value: Optional[str], choices: List[str], arg_name: str = "arg") -> str:
    if value is None:
        return choices[0]
    if value in choices:
        return value
    prefix = [c for c in choices if c.startswith(value)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) == 0:
        raise ValueError(f"{arg_name} must be one of {choices}; got {value!r}")
    raise ValueError(f"{arg_name} is ambiguous: {value!r} matches {prefix}")


def create_formula(LHS: str, RHS: List[str], LOG: str) -> str:
    LOG = match_arg(LOG, ["log", "log10", "NA"], "LOG")
    if LOG == "log":
        form = f"log({LHS})~"
    elif LOG == "log10":
        form = f"log10({LHS})~"
    else:
        form = f"{LHS}~"
    form = form + "+".join(RHS)
    return form


def parse_regression(
    X: pd.DataFrame,
    LHS: str,
    RHS: List[str],
    LOG: str = "log",
    TS: Any = None,
) -> Dict[str, Any]:
    LOG = match_arg(LOG, ["log", "log10", "NA"], "LOG")
    FORM = create_formula(LHS=LHS, RHS=RHS, LOG=LOG)

    if "TERM" not in X.columns:
        return {"TS": TS, "LOG": LOG, "FORM": FORM, "DATA": X}
    return {
        "TS": TS,
        "LOG": LOG,
        "FORM": FORM,
        "TERMS": sorted(pd.unique(X["TERM"])),
        "DATA": X,
    }


def all_vars(formula_text: str) -> List[str]:
    s = formula_text.replace(" ", "")
    out: List[str] = []
    token = ""
    for ch in s:
        if ch.isalnum() or ch == "_":
            token += ch
        else:
            if token:
                out.append(token)
                token = ""
    if token:
        out.append(token)
    return [x for x in out if x not in {"log", "log10"}]


def empty_dt(columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        return pd.DataFrame()
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})


def lag_col(
    X: Union[pd.Series, np.ndarray, List[float]], LAG: Union[int, float]
) -> np.ndarray:
    arr = np.asarray(X)
    if LAG == 0 or pd.isna(LAG):
        return arr.copy()
    LAG = int(LAG)
    if LAG < 0:
        k = -LAG
        return np.concatenate([np.repeat(np.nan, k), arr[: len(arr) - k]])
    return np.concatenate([arr[LAG:], np.repeat(np.nan, LAG)])


def _event_to_numeric(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series)
    return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)).to_numpy(
        dtype=float
    )


# --------------------------------------------------------------------------------------
# Exact crosscor port using the tested R bridge pattern from the attached chemistry port.
# This avoids guessing loess/acf behavior.
# --------------------------------------------------------------------------------------


def crosscor_r_bridge(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    lag: int,
    r_script_path: str,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, "in.csv")
        outfile = os.path.join(tmpdir, "out.csv")

        n1 = len(x1)
        n2 = len(x2)
        n = max(n1, n2)

        df = pd.DataFrame(
            {
                "x1": list(x1) + [np.nan] * (n - n1),
                "y1": list(y1) + [np.nan] * (n - n1),
                "x2": list(x2) + [np.nan] * (n - n2),
                "y2": list(y2) + [np.nan] * (n - n2),
                "lag": [lag] * n,
            }
        )
        df.to_csv(infile, index=False)

        cmd = ["Rscript", r_script_path, infile, outfile]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"Rscript failed\nCMD: {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        return pd.read_csv(outfile)


def do_lag(
    x: pd.DataFrame,
    y: pd.DataFrame,
    DEP: str,
    INDEP: str,
    MAXLAG: int,
    N: int,
    PND: float,
    r_script_path: str,
) -> Dict[str, Any]:
    X_0 = x.loc[~pd.isna(x[DEP])].copy()

    if "NDS" not in X_0.columns:
        X_0 = X_0.copy()
        X_0["NDS"] = False

    PNDS = X_0["NDS"].sum() / len(X_0) if len(X_0) > 0 else np.nan

    if len(X_0) >= N and PNDS <= PND:
        x1 = _event_to_numeric(X_0["EVENT"])
        x2 = _event_to_numeric(y["EVENT"])
        y1 = X_0[DEP].to_numpy()
        y2 = y[INDEP].to_numpy()
        lags = range(0, MAXLAG + 1)
        ccf_parts = [
            crosscor_r_bridge(x1, y1, x2, y2, lag, r_script_path) for lag in lags
        ]
        ccf = pd.concat(ccf_parts, ignore_index=True)
        max_abs = np.nanmax(np.abs(ccf["acf"].to_numpy()))
        lag = ccf.loc[np.abs(ccf["acf"]) == max_abs, "lag"].to_numpy()
    else:
        ccf = empty_dt(["acf", "lag"])
        lag = np.array([], dtype=float)

    return {"COD": ccf, "LAG": lag}


# --------------------------------------------------------------------------------------
# OLS helpers
# --------------------------------------------------------------------------------------


def _build_response(df: pd.DataFrame, dep: str, log_mode: str) -> np.ndarray:
    y = df[dep].to_numpy(dtype=float)
    if log_mode == "log":
        return np.log(y)
    if log_mode == "log10":
        return np.log10(y)
    return y


def _build_design(df: pd.DataFrame, indep: List[str]) -> np.ndarray:
    cols = [np.ones(len(df), dtype=float)]
    for var in indep:
        if var == "1":
            continue
        if var == "EVENT":
            cols.append(_event_to_numeric(df["EVENT"]))
        else:
            cols.append(df[var].to_numpy(dtype=float))
    return np.column_stack(cols)


class LMFit:
    def __init__(
        self,
        coefficients,
        residuals,
        fitted_values,
        df_residual,
        vcov,
        llf,
        terms,
        X,
        y,
    ):
        self.coefficients = coefficients
        self.residuals = residuals
        self.fitted_values = fitted_values
        self.df_residual = df_residual
        self.vcov = vcov
        self.llf = llf
        self.terms = terms
        self.X = X
        self.y = y

    def summary_coefficients(self) -> np.ndarray:
        se = np.sqrt(np.diag(self.vcov))
        tvals = self.coefficients / se
        pvals = 2.0 * student_t.sf(np.abs(tvals), self.df_residual)
        out = np.column_stack([self.coefficients, se, tvals, pvals])
        return out


def _parse_formula(formula_text: str):
    lhs, rhs = formula_text.split("~", 1)
    lhs = lhs.strip()
    rhs_terms = [t.strip() for t in rhs.split("+")]
    return lhs, rhs_terms


def _dep_from_lhs(lhs: str) -> str:
    if lhs.startswith("log(") and lhs.endswith(")"):
        return lhs[4:-1]
    if lhs.startswith("log10(") and lhs.endswith(")"):
        return lhs[6:-1]
    return lhs


def _log_mode_from_lhs(lhs: str) -> str:
    if lhs.startswith("log(") and lhs.endswith(")"):
        return "log"
    if lhs.startswith("log10(") and lhs.endswith(")"):
        return "log10"
    return "NA"


class LMFit:
    def __init__(
        self,
        coefficients,
        residuals,
        fitted_values,
        df_residual,
        vcov,
        llf,
        terms,
    ):
        self.coefficients = coefficients
        self.residuals = residuals
        self.fitted_values = fitted_values
        self.df_residual = df_residual
        self.vcov = vcov
        self.llf = llf
        self.terms = terms

    def summary_coefficients(self) -> np.ndarray:
        se = np.sqrt(np.diag(self.vcov))
        tvals = self.coefficients / se
        pvals = 2.0 * student_t.sf(np.abs(tvals), self.df_residual)
        return np.column_stack([self.coefficients, se, tvals, pvals])


def run_ols_formula(data: pd.DataFrame, formula_text: str) -> LMFit:
    lhs, rhs_terms = _parse_formula(formula_text)
    dep = _dep_from_lhs(lhs)
    log_mode = _log_mode_from_lhs(lhs)

    needed = [dep] + [t for t in rhs_terms if t not in {"1", "EVENT"}]
    if "EVENT" in rhs_terms:
        needed.append("EVENT")

    work = data[needed].dropna().copy()

    y = _build_response(work, dep, log_mode)
    X = _build_design(work, rhs_terms)

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid, :]
    y = y[valid]

    coef, residuals_ss, rank, s = np.linalg.lstsq(X, y, rcond=None)

    fitted = X @ coef
    resid = y - fitted

    n = len(y)
    p = X.shape[1]
    df_residual = float(n - p)

    rss = float(np.sum(resid**2))
    sigma2 = rss / df_residual if df_residual > 0 else np.nan

    xtx_inv = np.linalg.inv(X.T @ X)
    vcov = sigma2 * xtx_inv if np.isfinite(sigma2) else np.full((p, p), np.nan)

    if n > 0 and rss > 0:
        llf = -0.5 * n * (np.log(2 * np.pi) + 1 + np.log(rss / n))
    else:
        llf = np.nan

    return LMFit(
        coefficients=np.asarray(coef, dtype=float),
        residuals=np.asarray(resid, dtype=float),
        fitted_values=np.asarray(fitted, dtype=float),
        df_residual=df_residual,
        vcov=np.asarray(vcov, dtype=float),
        llf=float(llf),
        terms=formula_text,
    )


def run_ols(
    data: pd.DataFrame, DEP: str, FORM: str, LOG: str, N: int = 8
) -> Dict[str, Any]:
    X_0 = data.loc[~pd.isna(data[DEP])].copy()

    TERMS = all_vars(FORM)
    fit = run_ols_formula(X_0, FORM)
    fit_0 = run_ols_formula(X_0, f"{FORM.split('~', 1)[0]}~1")

    if len(TERMS) == 3:
        FORM2 = create_formula(LHS=TERMS[0], RHS=[TERMS[-2]], LOG=LOG)
        FORM3 = create_formula(LHS=TERMS[0], RHS=[TERMS[-1]], LOG=LOG)
        fit_2 = run_ols_formula(X_0, FORM2)
        fit_3 = run_ols_formula(X_0, FORM3)
    else:
        fit_2 = np.nan
        fit_3 = np.nan

    return {"CEN": fit, "CEN_0": fit_0, "CEN_2": fit_2, "CEN_3": fit_3}


def extract_model(
    x: pd.DataFrame,
    y: Dict[str, Any],
    DEP: str,
    INDEP: List[str],
    LAG: Union[int, float, np.ndarray],
    MODEL: str = "OLS",
) -> CovariateModel:
    MODEL = match_arg(MODEL, ["Tobit", "OLS"], "MODEL")
    NM = x["NAME"].dropna().unique()
    NM_val = NM[0] if len(NM) else ""
    x = x.loc[~pd.isna(x[DEP])].copy()

    if len(INDEP) == 2:
        if pd.isna(y["CEN"]):
            return CovariateModel(
                MODEL=MODEL,
                NAME=NM_val,
                LAG=LAG,
                SUM=np.empty((0, 0)),
                vcov=np.empty((0, 0)),
                AIC=empty_dt(),
                BIC=empty_dt(),
                df=np.array([]),
                RES=np.array([]),
                PRED=np.array([]),
                p_trend=np.array([]),
            )

        D = 2 * (float(y["CEN"].llf) - float(y["CEN_0"].llf))
        df_lr = 3 - 1
        p = 1 - chi2.cdf(D, df_lr)

        aic_vals = np.array(
            [
                2 * (len(y["CEN"].coefficients) - 1) - 2 * float(y["CEN"].llf),
                2 * len(y["CEN_0"].coefficients) - 2 * float(y["CEN_0"].llf),
                2 * (len(y["CEN_2"].coefficients) - 1) - 2 * float(y["CEN_2"].llf),
                2 * (len(y["CEN_3"].coefficients) - 1) - 2 * float(y["CEN_3"].llf),
            ]
        )
        aic_dt = pd.DataFrame(
            {
                "Formula": [INDEP[0] + " + " + INDEP[1], "0", INDEP[0], INDEP[1]],
                "AIC": aic_vals,
            }
        )
        aic_dt["RL"] = np.exp((aic_dt["AIC"].min() - aic_dt["AIC"]) / 2)

        bic_dt = pd.DataFrame(
            {
                "Formula": [INDEP[0] + " + " + INDEP[1], "0", INDEP[0], INDEP[1]],
                "BIC": [
                    math.log(len(x)) * (len(y["CEN"].coefficients) - 1)
                    - 2 * float(y["CEN"].llf),
                    math.log(len(x)) * len(y["CEN_0"].coefficients)
                    - 2 * float(y["CEN_0"].llf),
                    math.log(len(x)) * (len(y["CEN_2"].coefficients) - 1)
                    - 2 * float(y["CEN_2"].llf),
                    math.log(len(x)) * (len(y["CEN_3"].coefficients) - 1)
                    - 2 * float(y["CEN_3"].llf),
                ],
            }
        )

        SUM = y["CEN"].summary_coefficients()
        return CovariateModel(
            MODEL=MODEL,
            NAME=NM_val,
            LAG=LAG,
            SUM=SUM,
            vcov=y["CEN"].vcov[:3, :3],
            AIC=aic_dt,
            BIC=bic_dt,
            df=y["CEN"].df_residual,
            RES=y["CEN"].residuals,
            PRED=y["CEN"].fitted_values,
            p_trend=float(p),
        )

    if len(INDEP) == 1:
        if pd.isna(y["CEN"]):
            return CovariateModel(
                MODEL=MODEL,
                NAME=NM_val,
                LAG=LAG,
                SUM=np.empty((0, 0)),
                vcov=np.empty((0, 0)),
                AIC=empty_dt(),
                BIC=empty_dt(),
                df=np.array([]),
                RES=np.array([]),
                PRED=np.array([]),
                p_trend=np.array([]),
            )

        D = 2 * (float(y["CEN"].llf) - float(y["CEN_0"].llf))
        df_lr = 2 - 1
        p = 1 - chi2.cdf(D, df_lr)

        aic_dt = pd.DataFrame(
            {
                "Formula": [INDEP[0], "0"],
                "AIC": [
                    2 * (len(y["CEN"].coefficients) - 1) - 2 * float(y["CEN"].llf),
                    2 * len(y["CEN_0"].coefficients) - 2 * float(y["CEN_0"].llf),
                ],
            }
        )
        aic_dt["RL"] = np.exp((aic_dt["AIC"].min() - aic_dt["AIC"]) / 2)

        bic_dt = pd.DataFrame(
            {
                "Formula": [INDEP[0], "0"],
                "BIC": [
                    math.log(len(x)) * (len(y["CEN"].coefficients) - 1)
                    - 2 * float(y["CEN"].llf),
                    math.log(len(x)) * len(y["CEN_0"].coefficients)
                    - 2 * float(y["CEN_0"].llf),
                ],
            }
        )

        SUM = y["CEN"].summary_coefficients()
        return CovariateModel(
            MODEL=MODEL,
            NAME=NM_val,
            LAG=LAG,
            SUM=SUM,
            vcov=y["CEN"].vcov[:2, :2],
            AIC=aic_dt,
            BIC=bic_dt,
            df=y["CEN"].df_residual,
            RES=y["CEN"].residuals,
            PRED=y["CEN"].fitted_values,
            p_trend=float(p),
        )

    raise ValueError("Unexpected number of independent variables")


# --------------------------------------------------------------------------------------
# Exact structural port of doOLS()
# --------------------------------------------------------------------------------------


def _empty_trend_summary(
    X: Dict[str, Any], DATA: pd.DataFrame, TS: Any, LOG: str, MINDATE: pd.Timestamp
) -> TrendSummary:
    return TrendSummary(
        MODEL="OLS",
        FORM=X["FORM"],
        LOG=LOG,
        NAME=str(DATA["NAME"].iloc[0]),
        LAG=np.array([]),
        TS=TS,
        MINDATE=pd.to_datetime(MINDATE),
        SUM=np.empty((0, 0)),
        vcov=np.empty((0, 0)),
        AIC=empty_dt(),
        BIC=empty_dt(),
        df=np.array([]),
        RES=np.array([]),
        PRED=np.array([]),
        p_trend=np.array([]),
        COD=empty_dt(),
        DATA=DATA,
    )


def do_ols(
    X: Dict[str, Any],
    cov: str = "INTERP",
    MAXLAG: int = 90,
    MINDATE: Union[str, pd.Timestamp] = pd.Timestamp("1994-01-01"),
    ULAG: Optional[int] = None,
    N: int = 8,
    LOG: str = "log",
    r_script_path: Optional[str] = None,
    PND: float = 0.0,
) -> Union[TrendSummary, Dict[str, TrendSummary]]:
    LOG = match_arg(LOG, ["log", "log10", "NA"], "LOG")
    DATA = X["DATA"].copy()
    TERMS = all_vars(X["FORM"])
    DEP = TERMS[0]
    INDEP = TERMS[1:]
    TS = X.get("TS")
    ITER = X["TERMS"] if "TERM" in DATA.columns else np.array([np.nan])

    DATA[DEP] = np.where(
        pd.to_datetime(DATA["EVENT"]) < pd.to_datetime(MINDATE), np.nan, DATA[DEP]
    )

    if len(INDEP) == 1:
        if len(ITER) > 1:
            TREND: Dict[str, TrendSummary] = {}
            for i in range(1, len(ITER) + 1):
                SUB = DATA.loc[DATA["TERM"] == i].copy()
                NM = f"ITER{ITER[i - 1]}"
                DT = SUB.copy()
                DT = DT.loc[~pd.isna(DT[cov])].copy()
                CEN = run_ols(DT, DEP, X["FORM"], LOG, N)
                MODEL = extract_model(DT, CEN, DEP, INDEP, LAG=0, MODEL="OLS")
                TREND[NM] = TrendSummary(
                    MODEL=MODEL.MODEL,
                    FORM=X["FORM"],
                    LOG=LOG,
                    NAME=MODEL.NAME,
                    LAG=MODEL.LAG,
                    TS=TS,
                    MINDATE=pd.to_datetime(MINDATE),
                    SUM=MODEL.SUM,
                    vcov=MODEL.vcov,
                    AIC=MODEL.AIC,
                    BIC=MODEL.BIC,
                    df=MODEL.df,
                    RES=MODEL.RES,
                    PRED=MODEL.PRED,
                    p_trend=MODEL.p_trend,
                    COD=empty_dt(),
                    DATA=DATA,
                )
            return TREND

        CEN = run_ols(DATA, DEP, X["FORM"], LOG, N)
        MODEL = extract_model(DATA, CEN, DEP, INDEP, LAG=0, MODEL="OLS")
        return TrendSummary(
            MODEL=MODEL.MODEL,
            FORM=X["FORM"],
            LOG=LOG,
            NAME=MODEL.NAME,
            LAG=MODEL.LAG,
            TS=TS,
            MINDATE=pd.to_datetime(MINDATE),
            SUM=MODEL.SUM,
            vcov=MODEL.vcov,
            AIC=MODEL.AIC,
            BIC=MODEL.BIC,
            df=MODEL.df,
            RES=MODEL.RES,
            PRED=MODEL.PRED,
            p_trend=MODEL.p_trend,
            COD=empty_dt(),
            DATA=DATA,
        )

    if len(ITER) > 1:
        if r_script_path is None:
            raise ValueError("r_script_path is required when lag optimization is used")
        LIST = [g.copy() for _, g in DATA.groupby("TERM", sort=False)]
        CCF = []
        for x_sub in LIST:
            if ULAG is None or ULAG > 0:
                CCF.append(do_lag(x_sub, DATA, DEP, cov, MAXLAG, N, PND, r_script_path))
            else:
                CCF.append({"COD": empty_dt(), "LAG": 0, "COD_smooth": np.nan})
    elif len(INDEP) == 2:
        if ULAG is None or ULAG > 0:
            if r_script_path is None:
                raise ValueError(
                    "r_script_path is required when lag optimization is used"
                )
            CCF = do_lag(DATA, DATA, DEP, cov, MAXLAG, N, PND, r_script_path)
        else:
            CCF = {"COD": empty_dt(), "LAG": 0, "COD_smooth": np.nan}
    else:
        CCF = {"COD": empty_dt(), "LAG": 0, "COD_smooth": np.nan}

    if len(ITER) > 1:
        TREND = {}
        for i in range(1, len(ITER) + 1):
            SUB = DATA.loc[DATA["TERM"] == i].copy()
            CCF_SUB = CCF[i - 1]
            NM = f"ITER{ITER[i - 1]}"

            if (
                len(np.atleast_1d(CCF_SUB["LAG"])) == 0
                or len(SUB) < 365
                or SUB[DEP].notna().sum() < 10
            ):
                TREND[NM] = _empty_trend_summary(
                    X, DATA, TS, LOG, pd.to_datetime(MINDATE)
                )
            else:
                if ULAG is not None:
                    CCF_SUB["LAG"] = ULAG
                DT = SUB.copy()
                lag_value = CCF_SUB["LAG"]
                if isinstance(lag_value, np.ndarray):
                    lag_value = lag_value[0]
                if lag_value > 0:
                    DT[cov] = lag_col(DT[cov].to_numpy(), -lag_value)
                    DT = DT.loc[~pd.isna(DT[cov])].copy()
                FIT = run_ols(DT, DEP, X["FORM"], LOG, N)
                MODEL = extract_model(DT, FIT, DEP, INDEP, LAG=lag_value, MODEL="OLS")
                TREND[NM] = TrendSummary(
                    MODEL=MODEL.MODEL,
                    FORM=X["FORM"],
                    LOG=LOG,
                    NAME=MODEL.NAME,
                    LAG=MODEL.LAG,
                    TS=TS,
                    MINDATE=pd.to_datetime(MINDATE),
                    SUM=MODEL.SUM,
                    vcov=MODEL.vcov,
                    AIC=MODEL.AIC,
                    BIC=MODEL.BIC,
                    df=MODEL.df,
                    RES=MODEL.RES,
                    PRED=MODEL.PRED,
                    p_trend=MODEL.p_trend,
                    COD=CCF_SUB["COD"],
                    DATA=DATA,
                )
        return TREND

    if len(np.atleast_1d(CCF["LAG"])) == 0:
        return _empty_trend_summary(X, DATA, TS, LOG, pd.to_datetime(MINDATE))

    if ULAG is not None:
        CCF["LAG"] = ULAG
    DT = DATA.copy()
    lag_value = CCF["LAG"]
    if isinstance(lag_value, np.ndarray):
        lag_value = lag_value[0]
    if lag_value > 0:
        DT[cov] = lag_col(DT[cov].to_numpy(), -lag_value)
        DT = DT.loc[~pd.isna(DT[cov])].copy()
    FIT = run_ols(DT, DEP, X["FORM"], LOG, N)
    MODEL = extract_model(DT, FIT, DEP, INDEP, LAG=lag_value, MODEL="OLS")
    return TrendSummary(
        MODEL=MODEL.MODEL,
        FORM=X["FORM"],
        LOG=LOG,
        NAME=MODEL.NAME,
        LAG=MODEL.LAG,
        TS=TS,
        MINDATE=pd.to_datetime(MINDATE),
        SUM=MODEL.SUM,
        vcov=MODEL.vcov,
        AIC=MODEL.AIC,
        BIC=MODEL.BIC,
        df=MODEL.df,
        RES=MODEL.RES,
        PRED=MODEL.PRED,
        p_trend=MODEL.p_trend,
        COD=CCF["COD"],
        DATA=DATA,
    )


# --------------------------------------------------------------------------------------
# Main script port
# --------------------------------------------------------------------------------------


def run_water_level_trend_analysis(
    wl_rs: pd.DataFrame,
    MAXLAG: int = 90,
    LOG: str = "NA",
    TS: Any = None,
    MINDATE: Union[str, pd.Timestamp] = pd.Timestamp("1994-01-01"),
    N: int = 8,
    PND: float = 0.0,
    r_script_path: Optional[str] = None,
) -> Dict[str, Optional[Union[TrendSummary, Dict[str, TrendSummary]]]]:

    wllist = {name: grp.copy() for name, grp in wl_rs.groupby("NAME", sort=False)}
    wllag: Dict[str, Optional[Union[TrendSummary, Dict[str, TrendSummary]]]] = {}

    with tqdm(
        wllist.items(),
        total=len(wllist),
        desc="Water level trend analysis",
        unit="well",
    ) as pbar:
        for well_curr, (name, X) in enumerate(pbar, start=1):
            pbar.set_postfix(current=name, done=f"{well_curr}/{len(wllist)}")

            X_0 = X.loc[~pd.isna(X["WLE"])].copy()

            if pd.to_datetime(X_0["EVENT"]).dt.year.nunique() > 1:
                DAT = parse_regression(
                    X, LHS="WLE", RHS=["INTERP", "EVENT"], LOG=LOG, TS=TS
                )
                LM = do_ols(
                    DAT,
                    MAXLAG=MAXLAG,
                    LOG=LOG,
                    MINDATE=MINDATE,
                    N=N,
                    PND=PND,
                    r_script_path=r_script_path,
                )
                wllag[name] = LM
            else:
                wllag[name] = None

    return wllag


def flatten_water_level_trends(
    res: Dict[str, Optional[Union[TrendSummary, Dict[str, TrendSummary]]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def _scalar_or_nan(x):
        arr = np.atleast_1d(x)
        if arr.size == 0:
            return np.nan
        val = arr[0]
        return val if np.isscalar(val) else np.nan

    def _sum_value(ts: TrendSummary, row_idx: int, col_idx: int):
        if isinstance(ts.SUM, np.ndarray) and ts.SUM.ndim == 2:
            if ts.SUM.shape[0] > row_idx and ts.SUM.shape[1] > col_idx:
                return ts.SUM[row_idx, col_idx]
        return np.nan

    def _flatten_dt_value(x):
        if isinstance(x, pd.DataFrame):
            if x.empty:
                return np.nan
            return x.to_dict(orient="records")
        return x

    for well_name, obj in res.items():
        if obj is None:
            continue

        if isinstance(obj, dict):
            items = obj.items()
        else:
            items = [(np.nan, obj)]

        for iter_name, ts in items:
            rows.append(
                {
                    "KEY": (
                        well_name if pd.isna(iter_name) else f"{well_name}_{iter_name}"
                    ),
                    "NAME": well_name,
                    "ITER": iter_name,
                    "MODEL": ts.MODEL,
                    "FORM": ts.FORM,
                    "LOG": ts.LOG,
                    "TS": ts.TS,
                    "MINDATE": ts.MINDATE,
                    "p_trend": _scalar_or_nan(ts.p_trend),
                    "df": _scalar_or_nan(ts.df),
                    "LAG": _scalar_or_nan(ts.LAG),
                    "SUM_rows": (
                        ts.SUM.shape[0]
                        if isinstance(ts.SUM, np.ndarray) and ts.SUM.ndim == 2
                        else np.nan
                    ),
                    "SUM_cols": (
                        ts.SUM.shape[1]
                        if isinstance(ts.SUM, np.ndarray) and ts.SUM.ndim == 2
                        else np.nan
                    ),
                    "coef1": _sum_value(ts, 0, 0),
                    "coef2": _sum_value(ts, 1, 0),
                    "coef3": _sum_value(ts, 2, 0),
                    "CLASS": "Trend_Summary",
                    "COD": _flatten_dt_value(ts.COD),
                    "AIC": _flatten_dt_value(ts.AIC),
                    "BIC": _flatten_dt_value(ts.BIC),
                }
            )

    return pd.DataFrame(rows)
