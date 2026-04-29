import os
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
import tempfile
import subprocess
import scipy
from scipy.stats import norm
import re


def match_arg(value: Optional[str], choices: List[str], arg_name: str) -> str:
    """
    Minimal R-like match.arg for a scalar character argument.
    - exact match first
    - otherwise unique prefix match
    - otherwise error
    """
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


def create_formula_rstyle(LHS: str, RHS, LOG: str = "log") -> str:
    """
    Exact port of R createFormula(), except returned as a Python string
    instead of an R formula object.
    """
    LOG = match_arg(LOG, ["log", "log10", "none"], "LOG")

    if isinstance(RHS, str):
        RHS = [RHS]

    if LOG == "log":
        form = f"log({LHS})~"
    elif LOG == "log10":
        form = f"log10({LHS})~"
    else:
        form = f"{LHS}~"

    form = form + "+".join(RHS)
    return form


def parse_regression_rstyle(
    X: pd.DataFrame,
    LHS: str,
    RHS,
    LOG: str = "log",
    TS: str = "days",
):
    """
    Exact structural port of sspaTrendAnalysis::parseRegression.
    """
    LOG = match_arg(LOG, ["log", "log10", "none"], "LOG")
    TS = TS  # do not constrain until exact R TS choices are provided

    FORM = create_formula_rstyle(LHS=LHS, RHS=RHS, LOG=LOG)

    if "TERM" not in X.columns:
        return {
            "TS": TS,
            "LOG": LOG,
            "FORM": FORM,
            "DATA": X,
        }
    else:
        TERMS = sorted(pd.unique(X["TERM"]).tolist())
        return {
            "TS": TS,
            "LOG": LOG,
            "FORM": FORM,
            "TERMS": TERMS,
            "DATA": X,
        }


def _to_event_numeric_rstyle(series):
    """
    Match R's as.numeric(Date): days since 1970-01-01.
    """
    dt = pd.to_datetime(series, errors="coerce").dt.floor("D")
    return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)).to_numpy()


def do_lag_r_exact(x, y, DEP, INDEP, MAXLAG, N, PND, r_script_path):
    X_0 = x.loc[~pd.isna(x[DEP])].copy()
    n = len(X_0)
    PNDS = X_0["NDS"].sum() / n if n > 0 else np.nan

    if len(X_0) >= N and PNDS <= PND:
        x1 = _to_event_numeric_rstyle(X_0["EVENT"])
        x2 = _to_event_numeric_rstyle(y["EVENT"])
        y1 = X_0[DEP].to_numpy()
        y2 = y[INDEP].to_numpy()

        parts = []
        for lag in range(0, int(MAXLAG) + 1):
            part = crosscor_r_bridge(x1, y1, x2, y2, lag, r_script_path)
            parts.append(part)

        ccf = pd.concat(parts, ignore_index=True)
        max_abs = np.nanmax(np.abs(ccf["acf"].to_numpy()))
        lag = ccf.loc[np.abs(ccf["acf"]) == max_abs, "lag"].to_numpy()
    else:
        ccf = pd.DataFrame(columns=["acf", "lag"])
        lag = np.array([], dtype=float)

    return {"COD": ccf, "LAG": lag}


def _tricube(u):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    m = np.abs(u) < 1
    out[m] = (1 - np.abs(u[m]) ** 3) ** 3
    return out


def loess_fit_debug_1d(x, y, span=0.75, degree=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    q = max(int(np.ceil(span * n)), degree + 1)

    fitted = np.full(n, np.nan, dtype=float)

    for i in range(n):
        d = np.abs(x - x[i])
        h = np.partition(d, q - 1)[q - 1]

        if h == 0:
            w = (d == 0).astype(float)
        else:
            u = d / h
            w = np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)

        z = x - x[i]
        X = np.column_stack([np.ones(n), z, z**2])

        keep = w > 0
        Xw = X[keep] * np.sqrt(w[keep])[:, None]
        yw = y[keep] * np.sqrt(w[keep])

        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        fitted[i] = beta[0]

    return fitted


# def crosscor_rstyle(x1, y1, x2, y2, lag=0):
#     """
#     Port of R crosscor().

#     Returns a DataFrame with columns ['acf', 'lag'] to match:
#       data.table(acf = y, lag = lag)
#     """
#     lag = int(round(lag))

#     x1 = np.asarray(x1)
#     y1 = np.asarray(y1, dtype=float)
#     x2 = np.asarray(x2)
#     y2 = np.asarray(y2, dtype=float)

#     x_lag = x1 - lag
#     x = np.intersect1d(x2, x_lag)

#     # R:
#     # i1 <- sapply(x + lag, function(u) which.max(u == x1))
#     # i2 <- sapply(x,       function(u) which.max(u == x2))
#     #
#     # which.max(logical) returns first TRUE position, or 1 if all FALSE.
#     # Here x comes from intersect(x2, x_lag), so matches should exist.
#     i1 = np.array([np.argmax((x1 == u).astype(int)) for u in (x + lag)], dtype=int)
#     i2 = np.array([np.argmax((x2 == u).astype(int)) for u in x], dtype=int)

#     x1m = x1[i1]
#     y1m = y1[i1]
#     x2m = x2[i2]
#     y2m = y2[i2]

#     r1 = _loess_residuals_rstyle(x1m, y1m)
#     r2 = _loess_residuals_rstyle(x2m, y2m)

#     # R:
#     # Xts <- ts.intersect(as.ts(r1), as.ts(r2))
#     # acf(... lag.max = 0, type = "correlation")
#     #
#     # For lag.max=0 on 2 aligned series, this reduces to contemporaneous correlation.
#     ok = np.isfinite(r1) & np.isfinite(r2)
#     r1 = r1[ok]
#     r2 = r2[ok]

#     if len(r1) == 0 or len(r2) == 0:
#         acf_val = np.nan
#     elif len(r1) == 1 or len(r2) == 1:
#         acf_val = np.nan
#     else:
#         try:
#             acf_val = float(np.corrcoef(r1, r2)[0, 1])
#         except Exception:
#             acf_val = np.nan

#     return pd.DataFrame({"acf": [acf_val], "lag": [lag]})


def test_crosscor_for_well(chem_rs, well, term=1, lag=0):
    sub = chem_rs[(chem_rs["NAME"] == well) & (chem_rs["TERM"] == term)].copy()
    full = chem_rs[chem_rs["NAME"] == well].copy()

    x1 = _to_event_numeric_rstyle(sub.loc[sub["VAL"].notna(), "EVENT"])
    x2 = _to_event_numeric_rstyle(full["EVENT"])
    y1 = sub.loc[sub["VAL"].notna(), "VAL"].to_numpy()
    y2 = full["INTERP"].to_numpy()

    out = crosscor_debug_rstyle(x1, y1, x2, y2, lag=lag)
    return out


# def crosscor_debug_rstyle(x1, y1, x2, y2, lag=0):
#     lag = int(round(lag))

#     x1 = np.asarray(x1)
#     y1 = np.asarray(y1, dtype=float)
#     x2 = np.asarray(x2)
#     y2 = np.asarray(y2, dtype=float)

#     x_lag = x1 - lag
#     x = np.intersect1d(x2, x_lag)

#     i1 = np.array([np.argmax((x1 == u).astype(int)) for u in (x + lag)], dtype=int)
#     i2 = np.array([np.argmax((x2 == u).astype(int)) for u in x], dtype=int)

#     x1m = x1[i1]
#     y1m = y1[i1]
#     x2m = x2[i2]
#     y2m = y2[i2]

#     r1 = _loess_residuals_rstyle(x1m, y1m)
#     r2 = _loess_residuals_rstyle(x2m, y2m)

#     ok = np.isfinite(r1) & np.isfinite(r2)

#     if ok.sum() < 2:
#         acf_val = np.nan
#     else:
#         acf_val = float(np.corrcoef(r1[ok], r2[ok])[0, 1])

#     return {
#         "x1m": x1m,
#         "y1m": y1m,
#         "x2m": x2m,
#         "y2m": y2m,
#         "r1": r1,
#         "r2": r2,
#         "acf": acf_val,
#         "lag": lag,
#     }


def crosscor_r_bridge(x1, y1, x2, y2, lag, r_script_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, "in.csv")
        outfile = os.path.join(tmpdir, "out.csv")

        n1 = len(x1)
        n2 = len(x2)
        n = max(n1, n2)

        df = pd.DataFrame(
            {
                "id": [1] * n,
                "x1": list(x1) + [np.nan] * (n - n1),
                "y1": list(y1) + [np.nan] * (n - n1),
                "x2": list(x2) + [np.nan] * (n - n2),
                "y2": list(y2) + [np.nan] * (n - n2),
                "lag": [lag] * n,
            }
        )
        df.to_csv(infile, index=False)

        cmd = ["Rscript", r_script_path, infile, outfile]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if res.returncode != 0:
            raise RuntimeError(
                f"Rscript failed\n"
                f"CMD: {cmd}\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}"
            )

        out = pd.read_csv(outfile)
        return out


def censreg_ll_test(beta, X, y, left, right=None):
    beta = np.asarray(beta, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    left = np.asarray(left, dtype=bool)

    if right is None:
        right = np.zeros(len(y), dtype=bool)
    else:
        right = np.asarray(right, dtype=bool)

    sigma = np.exp(beta[-1])
    yhat = X @ beta[:-1]
    r = (y - yhat) / sigma

    ll = np.empty(len(y), dtype=float)
    between = ~(left | right)

    ll[left] = norm.logcdf(r[left])
    ll[between] = norm.logpdf(r[between]) - beta[-1]
    ll[right] = norm.logsf(r[right])

    return ll


def standardize_rstyle(x):
    z = np.asarray(x, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    mu = np.zeros(z.shape[1], dtype=float)
    sigma = np.ones(z.shape[1], dtype=float)
    Z = np.empty_like(z, dtype=float)

    for j in range(z.shape[1]):
        col = z[:, j]
        s = np.std(col, ddof=1)
        m = np.mean(col)
        if np.isnan(s) or s == 0:
            mu[j] = 0.0
            sigma[j] = 1.0
        else:
            mu[j] = m
            sigma[j] = s
        Z[:, j] = (col - mu[j]) / sigma[j]

    return {"Z": Z, "mu": mu, "sigma": sigma}


def loglik_attr_rstyle(
    theta,
    fnOrig,
    gradOrig=None,
    hessOrig=None,
    fixed=None,
    sumObs=False,
    returnHessian=True,
    **kwargs,
):

    theta = np.asarray(theta, dtype=float)
    nParam = len(theta)

    f = fnOrig(theta, **kwargs)
    f = np.asarray(f, dtype=float)

    if fixed is None:
        fixed = np.zeros(nParam, dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    if np.any(np.isnan(f)):
        gr = np.full((len(f), nParam), np.nan)
        h = np.nan
        return {"value": np.sum(f) if sumObs else f, "gradient": gr, "hessian": h}

    gr = None
    if gradOrig is not None:
        gr = gradOrig(theta, **kwargs)
    else:
        gr = numeric_gradient_rstyle(fnOrig, theta, fixed=fixed, **kwargs)

    gr = np.asarray(gr, dtype=float)

    activeGr = gr[:, ~fixed] if gr.ndim == 2 else gr[~fixed]
    if np.any(np.isnan(activeGr)):
        return {"value": np.sum(f) if sumObs else f, "gradient": gr, "hessian": np.nan}

    if gr.ndim == 1:
        gr = gr.copy()
        gr[fixed] = np.nan
    else:
        gr = gr.copy()
        gr[:, fixed] = np.nan

    if returnHessian is True:
        if hessOrig is not None:
            h = np.asarray(hessOrig(theta, **kwargs), dtype=float)
        else:

            def ll_func(th, **kw):
                return np.array([np.sum(fnOrig(th, **kw))], dtype=float)

            def grad_func(th, **kw):
                return sum_gradients_rstyle(
                    numeric_gradient_rstyle(fnOrig, th, fixed=fixed, **kw), nParam
                )

            h = numeric_hessian_rstyle(
                f=ll_func, grad=grad_func, t0=theta, fixed=fixed, **kwargs
            )

        h = np.asarray(h, dtype=float)
        h[fixed, :] = np.nan
        h[:, fixed] = np.nan
    else:
        h = None

    if sumObs:
        f = sum_keep_attr_rstyle(f)
        gr = sum_gradients_rstyle(gr, nParam)

    return {"value": f, "gradient": gr, "hessian": h}


def numeric_gradient_rstyle(f, t0, eps=1e-6, fixed=None, **kwargs):
    t0 = np.asarray(t0, dtype=float)
    n = len(t0)
    f0 = f(t0, **kwargs)
    nVal = len(f0)

    grad = np.full((nVal, n), np.nan)

    if fixed is None:
        fixed = np.zeros(n, dtype=bool)

    for i in range(n):
        if fixed[i]:
            continue
        t1 = t0.copy()
        t2 = t0.copy()
        t1[i] -= eps / 2
        t2[i] += eps / 2

        ft1 = f(t1, **kwargs)
        ft2 = f(t2, **kwargs)

        grad[:, i] = (ft2 - ft1) / eps

    return grad


def prepare_fixed_rstyle(start, activePar=None, fixed=None):
    start = np.asarray(start, dtype=float)
    nParam = len(start)

    if fixed is not None:
        if activePar is not None:
            if not np.all(activePar):
                # R warns; Python can ignore or warn
                pass

        if isinstance(fixed, (list, tuple, np.ndarray)):
            fixed_arr = np.asarray(fixed)

            if fixed_arr.dtype == bool:
                if fixed_arr.ndim != 1 or len(fixed_arr) != len(start):
                    raise ValueError(
                        "if fixed parameters are specified using logical values, "
                        "argument 'fixed' must be a logical vector with one element for each parameter"
                    )
                activePar = ~fixed_arr

            elif np.issubdtype(fixed_arr.dtype, np.number):
                if fixed_arr.ndim != 1 or len(fixed_arr) >= len(start):
                    raise ValueError(
                        "if fixed parameters are specified using their positions, "
                        "argument 'fixed' must be a numerical vector with less elements than the number of parameters"
                    )
                if fixed_arr.min() < 1 or fixed_arr.max() > len(start):
                    raise ValueError(
                        "if fixed parameters are specified using their positions, "
                        "argument 'fixed' must have values between 1 and the total number of parameters"
                    )
                activePar = ~np.isin(np.arange(1, len(start) + 1), fixed_arr)

            else:
                raise ValueError(
                    "argument 'fixed' must be either a logical vector, "
                    "a numeric vector, or a vector of character strings"
                )
        else:
            raise ValueError(
                "argument 'fixed' must be either a logical vector, "
                "a numeric vector, or a vector of character strings"
            )
    else:
        if activePar is None:
            activePar = np.repeat(True, len(start))
        else:
            activePar = np.asarray(activePar)
            if np.issubdtype(activePar.dtype, np.number) and activePar.dtype != bool:
                a = np.repeat(False, nParam)
                a[activePar - 1] = True
                activePar = a.astype(bool)

    activePar = np.asarray(activePar, dtype=bool)

    if np.all(~activePar):
        raise ValueError(
            "At least one parameter must not be fixed using argument 'fixed'"
        )

    return ~activePar


def sum_gradients_rstyle(gr, nParam):
    gr = np.asarray(gr)

    if gr.ndim > 1:
        gr = np.sum(gr, axis=0)
    else:
        if nParam == 1 and len(gr) > 1:
            gr = np.sum(gr)

    return gr


def sum_keep_attr_rstyle(x, keepNames=False, na_rm=False):
    x_arr = np.asarray(x)

    if na_rm:
        value = np.nansum(x_arr)
    else:
        value = np.sum(x_arr)

    return value


def observation_gradient_rstyle(g, nParam):
    g = np.asarray(g)

    if g.ndim == 1:
        if nParam == 1 and len(g) > 1:
            return True
        return False

    if g.shape[0] == 1:
        return False

    return True


def maxim_message_rstyle(code):
    messages = {
        1: "gradient close to zero",
        2: "successive function values within tolerance limit",
        3: "Last step could not find a value above the current.\nBoundary of parameter space? \nConsider switching to a more robust optimisation method temporarily.",
        4: "Iteration limit exceeded.",
        5: "Infinite value",
        6: "Infinite gradient",
        7: "Infinite Hessian",
        8: "Relative change of the function within relative tolerance",
        9: "Gradient did not change,\ncannot improve BFGS approximation for the Hessian.\nUse different optimizer and/or analytic gradient.",
        100: "Initial value out of range.",
    }
    return messages.get(code, f"Code {code}")


import inspect


def check_func_args_rstyle(func, checkArgs, argName, funcName):
    if not callable(func):
        raise ValueError(
            f"argument '{argName}' of function '{funcName}' is not a function"
        )

    sig = inspect.signature(func)
    funcArgs = list(sig.parameters.keys())

    if len(funcArgs) > 1:
        test_args = funcArgs[1:]

        matches = []
        for a in test_args:
            matched = [c for c in checkArgs if c.startswith(a)]
            matches.append((a, matched))

        hit = [a for a, m in matches if len(m) > 0]

        if len(hit) == 1:
            raise ValueError(
                f"argument '{hit[0]}' of the function specified in argument "
                f"'{argName}' of function '{funcName}' (partially) matches the "
                f"argument names of function '{funcName}'. Please change the "
                f"name of this argument"
            )
        elif len(hit) > 1:
            joined = "', '".join(hit)
            raise ValueError(
                f"arguments '{joined}' of the function specified in argument "
                f"'{argName}' of function '{funcName}' (partially) match the "
                f"argument names of function '{funcName}'. Please change the "
                f"names of these arguments"
            )

    return None


def maxnr_init_rstyle(fn, start, fixed=None, bhhhHessian=False, **kwargs):
    start = np.asarray(start, dtype=float)
    nParam = len(start)

    if fixed is None:
        fixed = np.zeros(nParam, dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    returnHessian = "BHHH" if bhhhHessian else True

    f1 = fn(start, fixed=fixed, sumObs=True, returnHessian=returnHessian, **kwargs)

    f1_value = f1["value"]
    G1 = np.asarray(f1["gradient"], dtype=float)
    H1 = f1["hessian"]

    if np.any(np.isnan(np.atleast_1d(f1_value))):
        return {
            "code": 100,
            "message": maxim_message_rstyle(100),
            "iterations": 0,
            "type": "Newton-Raphson maximisation",
        }

    if (
        np.any(np.isinf(np.atleast_1d(f1_value)))
        and np.sum(np.atleast_1d(f1_value)) > 0
    ):
        return {
            "code": 5,
            "message": maxim_message_rstyle(5),
            "iterations": 0,
            "type": "Newton-Raphson maximisation",
        }

    if np.any(np.isnan(G1[~fixed])):
        raise ValueError("NA in the initial gradient")

    if np.any(np.isinf(G1[~fixed])):
        raise ValueError("Infinite initial gradient")

    if len(G1) != nParam:
        raise ValueError(
            f"length of gradient ({len(G1)}) not equal to the no. of parameters ({nParam})"
        )

    if H1 is not None:
        H1 = np.asarray(H1, dtype=float)

        if H1.size == 1:
            if np.any(np.isnan(H1)):
                raise ValueError("NA in the initial Hessian")

        if np.any(np.isnan(H1[~fixed][:, ~fixed])):
            raise ValueError("NA in the initial Hessian")

        if np.any(np.isinf(H1)):
            raise ValueError("Infinite initial Hessian")

    return {
        "start1": start.copy(),
        "f1": f1_value,
        "G1": G1,
        "H1": H1,
        "fixed": fixed,
        "nParam": nParam,
        "returnHessian": returnHessian,
        "type": "Newton-Raphson maximisation",
    }


def numeric_hessian_rstyle(f, grad=None, t0=None, eps=1e-6, fixed=None, **kwargs):
    t0 = np.asarray(t0, dtype=float)

    if fixed is None:
        fixed = np.zeros(len(t0), dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    n = len(t0)

    if grad is None:
        raise NotImplementedError("numericNHessian path not implemented yet")
    else:
        H = numeric_gradient_rstyle(f=grad, t0=t0, eps=eps, fixed=fixed, **kwargs)

    return np.asarray(H, dtype=float)


def maxnr_one_step_rstyle(start0, f0, G0, H0, fixed=None, lambdatol=1e-6, qrtol=1e-10):
    start0 = np.asarray(start0, dtype=float)
    G0 = np.asarray(G0, dtype=float)
    H0 = np.asarray(H0, dtype=float)

    nParam = len(start0)
    I = np.eye(nParam)

    if fixed is None:
        fixed = np.zeros(nParam, dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    H = H0.copy()
    lambda_ = 0.0

    def max_eigen(M):
        vals = np.linalg.eigvalsh(M)
        return vals[-1]

    active = ~fixed
    Haa = H[np.ix_(active, active)]

    while True:
        me = max_eigen(Haa)
        qRank = np.linalg.matrix_rank(Haa, tol=qrtol)

        if not (me >= -lambdatol or qRank < np.sum(active)):
            break

        lambda_ = abs(me) + lambdatol + np.min(np.abs(np.diag(H)[active])) / 1e7
        H = H - lambda_ * I
        Haa = H[np.ix_(active, active)]

    amount = np.zeros(nParam, dtype=float)
    amount[active] = np.linalg.solve(Haa, G0[active])

    step = 1.0
    start1 = start0 - step * amount

    return {
        "lambda": lambda_,
        "step": step,
        "amount": amount,
        "start1": start1,
        "H": H,
    }


def maxnr_backtrack_rstyle(
    fn, start0, f0, amount, fixed=None, returnHessian=True, steptol=1e-10, **kwargs
):
    start0 = np.asarray(start0, dtype=float)
    amount = np.asarray(amount, dtype=float)

    if fixed is None:
        fixed = np.zeros(len(start0), dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    step = 1.0
    start1 = start0 - step * amount
    f1 = fn(start1, fixed=fixed, sumObs=True, returnHessian=returnHessian, **kwargs)

    while (
        np.any(np.isnan(np.atleast_1d(f1["value"])))
        or (np.sum(np.atleast_1d(f1["value"])) < np.sum(np.atleast_1d(f0)))
    ) and (step >= steptol):
        step = step / 2.0
        start1 = start0 - step * amount
        f1 = fn(start1, fixed=fixed, sumObs=True, returnHessian=returnHessian, **kwargs)

    if step < steptol:
        start1 = start0.copy()
        f1 = f0
        samm = {"theta0": start0, "f0": f0, "climb": amount}
    else:
        samm = None

    return {
        "step": step,
        "start1": start1,
        "f1": f1,
        "last_step": samm,
    }


def maxnr_termination_code_rstyle(
    f0,
    f1,
    G1,
    step,
    fixed=None,
    gradtol=1e-6,
    tol=1e-8,
    reltol=np.sqrt(np.finfo(float).eps),
):
    G1 = np.asarray(G1, dtype=float)

    if fixed is None:
        fixed = np.zeros(len(G1), dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool)

    if step < 1e-10:
        return 3

    if np.sqrt(np.dot(G1[~fixed], G1[~fixed])) < gradtol:
        return 1

    if (np.sum(np.atleast_1d(f1)) - np.sum(np.atleast_1d(f0))) < tol:
        return 2

    if (np.sum(np.atleast_1d(f1)) - np.sum(np.atleast_1d(f0))) < reltol * (
        np.sum(np.atleast_1d(f1)) + reltol
    ):
        return 2

    if np.any(np.isinf(np.atleast_1d(f1))) and np.sum(np.atleast_1d(f1)) > 0:
        return 5

    return None


def maxnr_compute_rstyle(
    fn,
    start,
    print_level=0,
    tol=1e-8,
    reltol=np.sqrt(np.finfo(float).eps),
    gradtol=1e-6,
    steptol=1e-10,
    lambdatol=1e-6,
    qrtol=1e-10,
    iterlim=150,
    finalHessian=True,
    bhhhHessian=False,
    fixed=None,
    **kwargs,
):

    init = maxnr_init_rstyle(
        fn=fn, start=start, fixed=fixed, bhhhHessian=bhhhHessian, **kwargs
    )

    if "code" in init:
        return init

    start1 = init["start1"]
    f1 = init["f1"]
    G1 = init["G1"]
    H1 = init["H1"]
    fixed = init["fixed"]
    nParam = init["nParam"]
    returnHessian = init["returnHessian"]
    maxim_type = init["type"]

    iter_count = 0
    samm = None
    code = None

    while True:
        if iter_count >= iterlim:
            code = 4
            break

        iter_count += 1
        start0 = start1.copy()
        f0 = f1
        G0 = G1.copy()
        H0 = H1.copy()

        step_out = maxnr_one_step_rstyle(
            start0=start0,
            f0=f0,
            G0=G0,
            H0=H0,
            fixed=fixed,
            lambdatol=lambdatol,
            qrtol=qrtol,
        )

        bt = maxnr_backtrack_rstyle(
            fn=fn,
            start0=start0,
            f0=f0,
            amount=step_out["amount"],
            fixed=fixed,
            returnHessian=returnHessian,
            steptol=steptol,
            **kwargs,
        )

        start1 = bt["start1"]
        f1_obj = bt["f1"]
        samm = bt["last_step"]

        f1 = f1_obj["value"]
        G1 = np.asarray(f1_obj["gradient"], dtype=float)
        H1 = np.asarray(f1_obj["hessian"], dtype=float)

        code = maxnr_termination_code_rstyle(
            f0=f0,
            f1=f1,
            G1=G1,
            step=bt["step"],
            fixed=fixed,
            gradtol=gradtol,
            tol=tol,
            reltol=reltol,
        )

        if code is not None:
            break

    return {
        "maximum": float(np.asarray(f1).reshape(-1)[0]),
        "estimate": start1,
        "gradient": G1,
        "hessian": H1 if finalHessian else None,
        "code": code,
        "message": maxim_message_rstyle(code),
        "last_step": samm,
        "fixed": fixed,
        "iterations": iter_count,
        "type": maxim_type,
    }


def all_vars_rstyle(formula_text: str):
    """
    Approximate R all.vars() for simple formulas like:
      log(VAL)~INTERP+EVENT
      VAL~EVENT
    Returns variable names in order of appearance.
    """
    s = str(formula_text).replace(" ", "")
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)
    # drop known function names
    drop = {"log", "log10"}
    return [t for t in tokens if t not in drop]


def censreg_fit_rstyle(
    formula_text, data, left, right=None, start=None, logLikOnly=False
):
    if right is None:
        right = np.zeros(len(data), dtype=bool)
    else:
        right = np.asarray(right, dtype=bool)

    if start is not None:
        raise NotImplementedError("start != NULL path not implemented yet")

    # parse formula
    vars_ = all_vars_rstyle(formula_text)
    dep = vars_[0]
    indep = vars_[1:]

    df = data.copy()

    # response
    if formula_text.startswith("log("):
        yVec = np.log(df[dep].to_numpy(dtype=float))
    elif formula_text.startswith("log10("):
        yVec = np.log10(df[dep].to_numpy(dtype=float))
    else:
        yVec = df[dep].to_numpy(dtype=float)

    # design matrix
    X_cols = [np.ones(len(df), dtype=float)]
    for v in indep:
        if v == "EVENT":
            ev = (
                (pd.to_datetime(df["EVENT"]) - pd.Timestamp("1970-01-01"))
                / pd.Timedelta(days=1)
            ).to_numpy(dtype=float)
            X_cols.append(ev)
        else:
            X_cols.append(df[v].to_numpy(dtype=float))

    xMat = np.column_stack(X_cols)

    validObs = (
        np.sum(
            np.isnan(np.column_stack([yVec, xMat]))
            | np.isinf(np.column_stack([yVec, xMat])),
            axis=1,
        )
        == 0
    )

    yVec = yVec[validObs]
    xMat = xMat[validObs, :]
    left = np.asarray(left, dtype=bool)[validObs]
    right = np.asarray(right, dtype=bool)[validObs]

    y_std = standardize_rstyle(yVec)
    yVec0 = y_std["Z"].ravel()

    x_std = standardize_rstyle(xMat)
    xMat0 = x_std["Z"]

    p = xMat0.shape[1] + 1

    ols_coef, *_ = np.linalg.lstsq(xMat0, yVec0, rcond=None)
    resid = yVec0 - xMat0 @ ols_coef
    msr = np.mean(resid**2)
    msr_log = -300.0 if msr <= 0 else np.log(msr)
    start0 = np.concatenate([ols_coef, [msr_log]])

    if logLikOnly:
        return censreg_ll_test(start0, xMat0, yVec0, left, right)

    fit = maxnr_compute_rstyle(
        fn=lambda theta, **kw: loglik_attr_rstyle(
            theta,
            fnOrig=lambda b, **kw2: censreg_ll_test(b, xMat0, yVec0, left, right),
            fixed=kw.get("fixed"),
            sumObs=kw.get("sumObs", False),
            returnHessian=kw.get("returnHessian", True),
        ),
        start=start0,
        fixed=np.zeros(len(start0), dtype=bool),
    )

    fit["estimate.0"] = fit["estimate"].copy()
    fit["y.std"] = y_std
    fit["x.std"] = x_std

    fit["estimate"] = unstandardize_censreg_estimate_rstyle(
        fit["estimate.0"], fit["x.std"], fit["y.std"]
    )
    fit["coefficients"] = fit["estimate"][:-1]

    # parameter names
    param_names = ["(Intercept)"] + indep + ["logSigma"]

    # variance-covariance on unstandardized scale
    fit["varcovar"] = compute_varcovar_rstyle(
        fit["hessian"],
        fit["x.std"],
        fit["y.std"],
    )

    fit["coef_table"] = coefficient_pvalues_rstyle(
        fit["estimate"],
        fit["varcovar"],
        param_names,
    )
    fit["fitted.values"] = xMat @ fit["coefficients"]
    fit["residuals"] = yVec - fit["fitted.values"]
    fit["df.residual"] = len(yVec) - p

    fit["nObs"] = {
        "Total": len(yVec),
        "Left-censored": int(np.sum(left)),
        "Uncensored": int(np.sum(~(left | right))),
        "Right-censored": int(np.sum(right)),
    }
    fit["model"] = df.loc[validObs].copy()
    fit["terms"] = formula_text
    fit["left"] = left
    fit["right"] = right

    return fit


def unstandardize_censreg_estimate_rstyle(estimate0, x_std, y_std):
    estimate0 = np.asarray(estimate0, dtype=float)
    p = len(estimate0)

    e = estimate0[:-1].copy()

    tau = y_std["sigma"][0] / x_std["sigma"]

    # intercept column is the constant column with sigma == 1 after standardize
    i_constant = np.where(np.asarray(x_std["sigma"], dtype=float) == 1.0)[0]

    if len(i_constant) > 0:
        ic = i_constant[0]
        e[ic] = (
            y_std["mu"][0]
            + (e[ic] - np.sum(e * x_std["mu"] / x_std["sigma"])) * y_std["sigma"][0]
        )

    nonconst = np.setdiff1d(np.arange(len(e)), i_constant)
    e[nonconst] = e[nonconst] * tau[nonconst]

    estimate = np.concatenate([e, [estimate0[-1] + np.log(y_std["sigma"][0])]])
    return estimate


def run_tobit_rstyle(x, DEP, FORM, LOG, N, PND):
    X_0 = x.loc[~pd.isna(x[DEP])].copy()
    TERMS = all_vars_rstyle(FORM)

    if len(TERMS) == 3:
        FORM2 = create_formula_rstyle(
            LHS=TERMS[0],
            RHS=[TERMS[len(TERMS) - 2]],
            LOG=LOG,
        )
        FORM3 = create_formula_rstyle(
            LHS=TERMS[0],
            RHS=[TERMS[len(TERMS) - 1]],
            LOG=LOG,
        )

    if "NDS" not in X_0.columns:
        X_0 = X_0.copy()
        X_0["NDS"] = False

    n = len(X_0)
    PNDS = X_0["NDS"].sum() / n if n > 0 else np.nan

    if len(X_0) >= N and PNDS <= PND:
        CEN = censreg_fit_rstyle(
            formula_text=FORM,
            data=X_0,
            left=X_0["NDS"].to_numpy(dtype=bool),
        )

        # direct intercept-only refit instead of R update(CEN, . ~ 1)
        dep = TERMS[0]
        FORM0 = create_formula_rstyle(LHS=dep, RHS=["1"], LOG=LOG).replace("~1", "~1")
        # create intercept-only formula explicitly
        if LOG == "log":
            FORM0 = f"log({dep})~1"
        elif LOG == "log10":
            FORM0 = f"log10({dep})~1"
        else:
            FORM0 = f"{dep}~1"

        CEN_0 = censreg_fit_rstyle(
            formula_text=FORM0,
            data=X_0,
            left=X_0["NDS"].to_numpy(dtype=bool),
        )

        if len(TERMS) == 3:
            CEN_2 = censreg_fit_rstyle(
                formula_text=FORM2,
                data=X_0,
                left=X_0["NDS"].to_numpy(dtype=bool),
            )
            CEN_3 = censreg_fit_rstyle(
                formula_text=FORM3,
                data=X_0,
                left=X_0["NDS"].to_numpy(dtype=bool),
            )
        else:
            CEN_2 = np.nan
            CEN_3 = np.nan
    else:
        CEN = np.nan
        CEN_0 = np.nan
        CEN_2 = np.nan
        CEN_3 = np.nan

    return {
        "CEN": CEN,
        "CEN_0": CEN_0,
        "CEN_2": CEN_2,
        "CEN_3": CEN_3,
    }


def extract_model_rstyle(x, y, DEP, INDEP, LAG, MODEL="Tobit", ITER=None):
    NM = x["NAME"].iloc[0] if len(x) else None
    x_nonmiss = x.loc[~pd.isna(x[DEP])].copy()

    if len(INDEP) == 2:
        if y["CEN"] is None or isinstance(y["CEN"], float):
            return {
                "KEY": f"{NM}__ITER{ITER}" if ITER is not None else NM,
                "WELL": NM,
                "ITER": ITER,
                "CLASS": "Trend_Summary",
                "MODEL": MODEL,
                "FORM_raw": create_formula_rstyle(DEP, INDEP, "log"),
                "FORM_label": "INTERP + EVENT",
                "LOG": "log",
                "LAG": LAG,
                "p_trend": np.nan,
                "df": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "RL": np.nan,
                "AIC_EVENT": np.nan,
                "AIC_INTERP": np.nan,
                "AIC_NULL": np.nan,
                "BIC_EVENT": np.nan,
                "BIC_INTERP": np.nan,
                "BIC_NULL": np.nan,
                "logLik": np.nan,
                "n_obs": np.nan,
                "n_cens": np.nan,
                "beta_intercept": np.nan,
                "beta_interp": np.nan,
                "beta_event": np.nan,
                "beta_logSigma": np.nan,
                "p_interp": np.nan,
                "p_event": np.nan,
                "SUM_rows": 1,
                "SUM_cols": 1,
                "model_type": "SUM_ROWS_1",
                "fit_ok": False,
            }

        ll_full = y["CEN"]["maximum"]
        ll_null = y["CEN_0"]["maximum"]
        D = 2 * (ll_full - ll_null)
        df_lr = 3 - 1
        p_trend = 1 - scipy.stats.chi2.cdf(D, df_lr)

        def AIC(k, ll):
            return 2 * k - 2 * ll

        def BIC(k, ll, n):
            return np.log(n) * k - 2 * ll

        n = len(x_nonmiss)

        aic_full = AIC(len(y["CEN"]["coefficients"]) - 1, ll_full)
        aic_null = AIC(len(y["CEN_0"]["coefficients"]), ll_null)
        aic_interp = AIC(len(y["CEN_2"]["coefficients"]) - 1, y["CEN_2"]["maximum"])
        aic_event = AIC(len(y["CEN_3"]["coefficients"]) - 1, y["CEN_3"]["maximum"])

        bic_full = BIC(len(y["CEN"]["coefficients"]) - 1, ll_full, n)
        bic_null = BIC(len(y["CEN_0"]["coefficients"]), ll_null, n)
        bic_interp = BIC(len(y["CEN_2"]["coefficients"]) - 1, y["CEN_2"]["maximum"], n)
        bic_event = BIC(len(y["CEN_3"]["coefficients"]) - 1, y["CEN_3"]["maximum"], n)

        rl = np.exp((min([aic_full, aic_null, aic_interp, aic_event]) - aic_full) / 2)

        coef = np.asarray(y["CEN"]["estimate"], dtype=float)
        ct = y["CEN"].get("coef_table", {})

        return {
            "KEY": f"{NM}__ITER{ITER}" if ITER is not None else NM,
            "WELL": NM,
            "ITER": ITER,
            "CLASS": "Trend_Summary",
            "MODEL": MODEL,
            "FORM_raw": create_formula_rstyle(DEP, INDEP, "log"),
            "FORM_label": "INTERP + EVENT",
            "LOG": "log",
            "LAG": LAG,
            "p_trend": p_trend,
            "df": y["CEN"]["df.residual"],
            "AIC": aic_full,
            "BIC": bic_full,
            "RL": rl,
            "AIC_EVENT": aic_event,
            "AIC_INTERP": aic_interp,
            "AIC_NULL": aic_null,
            "BIC_EVENT": bic_event,
            "BIC_INTERP": bic_interp,
            "BIC_NULL": bic_null,
            "logLik": ll_full,
            "n_obs": len(y["CEN"]["fitted.values"]),
            "n_cens": (
                y["CEN"]["nObs"]["Left-censored"] if "nObs" in y["CEN"] else np.nan
            ),
            "beta_intercept": coef[0] if len(coef) > 0 else np.nan,
            "beta_interp": coef[1] if len(coef) > 1 else np.nan,
            "beta_event": coef[2] if len(coef) > 2 else np.nan,
            "beta_logSigma": coef[-1] if len(coef) > 0 else np.nan,
            "p_interp": ct.get("INTERP", {}).get("p", np.nan),
            "p_event": ct.get("EVENT", {}).get("p", np.nan),
            "SUM_rows": 4,
            "SUM_cols": 4,
            "model_type": "INTERP+EVENT",
            "fit_ok": True,
        }
    elif len(INDEP) == 1:
        if y["CEN"] is None or isinstance(y["CEN"], float):
            return {
                "KEY": f"{NM}__ITER{ITER}" if ITER is not None else NM,
                "WELL": NM,
                "ITER": ITER,
                "CLASS": "Trend_Summary",
                "MODEL": MODEL,
                "FORM_raw": create_formula_rstyle(DEP, INDEP, "log"),
                "FORM_label": INDEP[0],
                "LOG": "log",
                "LAG": 0,
                "p_trend": np.nan,
                "df": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "RL": np.nan,
                "AIC_EVENT": np.nan,
                "AIC_INTERP": np.nan,
                "AIC_NULL": np.nan,
                "BIC_EVENT": np.nan,
                "BIC_INTERP": np.nan,
                "BIC_NULL": np.nan,
                "logLik": np.nan,
                "n_obs": np.nan,
                "n_cens": np.nan,
                "beta_intercept": np.nan,
                "beta_interp": np.nan,
                "beta_event": np.nan,
                "beta_logSigma": np.nan,
                "p_interp": np.nan,
                "p_event": np.nan,
                "SUM_rows": 1,
                "SUM_cols": 1,
                "model_type": "SUM_ROWS_1",
                "fit_ok": False,
            }

        ll_full = y["CEN"]["maximum"]
        ll_null = y["CEN_0"]["maximum"]
        D = 2 * (ll_full - ll_null)
        df_lr = 2 - 1
        p_trend = 1 - scipy.stats.chi2.cdf(D, df_lr)

        def AIC(k, ll):
            return 2 * k - 2 * ll

        def BIC(k, ll, n):
            return np.log(n) * k - 2 * ll

        n = len(x_nonmiss)

        aic_full = AIC(len(y["CEN"]["coefficients"]) - 1, ll_full)
        aic_null = AIC(len(y["CEN_0"]["coefficients"]), ll_null)
        bic_full = BIC(len(y["CEN"]["coefficients"]) - 1, ll_full, n)
        bic_null = BIC(len(y["CEN_0"]["coefficients"]), ll_null, n)

        rl = np.exp((min([aic_full, aic_null]) - aic_full) / 2)

        coef = np.asarray(y["CEN"]["estimate"], dtype=float)
        ct = y["CEN"].get("coef_table", {})

        return {
            "KEY": f"{NM}__ITER{ITER}" if ITER is not None else NM,
            "WELL": NM,
            "ITER": ITER,
            "CLASS": "Trend_Summary",
            "MODEL": MODEL,
            "FORM_raw": create_formula_rstyle(DEP, INDEP, "log"),
            "FORM_label": INDEP[0],
            "LOG": "log",
            "LAG": 0,
            "p_trend": p_trend,
            "df": y["CEN"]["df.residual"],
            "AIC": aic_full,
            "BIC": bic_full,
            "RL": rl,
            "AIC_EVENT": aic_full if INDEP[0] == "EVENT" else np.nan,
            "AIC_INTERP": aic_full if INDEP[0] == "INTERP" else np.nan,
            "AIC_NULL": aic_null,
            "BIC_EVENT": bic_full if INDEP[0] == "EVENT" else np.nan,
            "BIC_INTERP": bic_full if INDEP[0] == "INTERP" else np.nan,
            "BIC_NULL": bic_null,
            "logLik": ll_full,
            "n_obs": len(y["CEN"]["fitted.values"]),
            "n_cens": (
                y["CEN"]["nObs"]["Left-censored"] if "nObs" in y["CEN"] else np.nan
            ),
            "beta_intercept": coef[0] if len(coef) > 0 else np.nan,
            "beta_interp": (
                coef[1] if INDEP[0] == "INTERP" and len(coef) > 1 else np.nan
            ),
            "beta_event": coef[1] if INDEP[0] == "EVENT" and len(coef) > 1 else np.nan,
            "beta_logSigma": coef[-1] if len(coef) > 0 else np.nan,
            "p_interp": ct.get("INTERP", {}).get("p", np.nan),
            "p_event": ct.get("EVENT", {}).get("p", np.nan),
            "SUM_rows": 3,
            "SUM_cols": 4,
            "model_type": INDEP[0],
            "fit_ok": True,
        }


def lag_col_rstyle(X, LAG):
    X = np.asarray(X, dtype=float)

    if LAG == 0 or np.isnan(LAG):
        return X.copy()

    LAG = int(LAG)

    if LAG < 0:
        k = -LAG
        return np.concatenate([np.full(k, np.nan), X[: len(X) - k]])
    else:
        return np.concatenate([X[LAG:], np.full(LAG, np.nan)])


def do_tobit_rstyle(
    x, DEP, INDEP, LOG, MAXLAG, N, PND, r_script_path, ulags=None, newrs_names=None
):
    results = []

    well_count = x["NAME"].dropna().nunique()

    if x.empty:
        return results

    names = x["NAME"].dropna().unique()

    with tqdm(names, total=well_count, desc="Tobit analysis", unit="well") as pbar:
        for well_curr, name in enumerate(pbar, start=1):
            df_full = x[x["NAME"] == name].copy()

            ulag_applied = ulags is not None and name in ulags and pd.notna(ulags[name])
            is_newrs = newrs_names is not None and name in newrs_names

            pbar.set_postfix(
                current=name,
                done=f"{well_curr}/{well_count}",
                ULAG=ulag_applied,
            )

            for term in sorted(pd.Series(df_full["TERM"]).dropna().unique()):
                df_term_raw = df_full[df_full["TERM"] == term].copy()

                if is_newrs:
                    lag_scalar = 0
                    df_term = df_term_raw.copy()
                    indep_term = ["EVENT"]
                else:
                    if ulag_applied:
                        lag = ulags[name]
                    else:
                        lag_out = do_lag_r_exact(
                            df_term_raw,
                            df_full,
                            DEP=DEP,
                            INDEP=INDEP[0],
                            MAXLAG=MAXLAG,
                            N=N,
                            PND=PND,
                            r_script_path=r_script_path,
                        )
                        lag = lag_out["LAG"]

                    if isinstance(lag, (list, tuple, np.ndarray, pd.Series)):
                        lag_scalar = lag[0] if len(lag) > 0 else np.nan
                    else:
                        lag_scalar = lag

                    df_lag = df_full.copy()
                    if pd.notna(lag_scalar) and lag_scalar > 0:
                        df_lag["INTERP"] = lag_col_rstyle(
                            df_lag["INTERP"].to_numpy(), -lag_scalar
                        )

                    df_term = df_lag[df_lag["TERM"] == term].copy()

                    if pd.notna(lag_scalar) and lag_scalar > 0:
                        df_term = df_term.loc[~pd.isna(df_term["INTERP"])].copy()

                    indep_term = ["INTERP", "EVENT"]

                FORM = create_formula_rstyle(DEP, indep_term, LOG)

                tobit_out = run_tobit_rstyle(
                    x=df_term,
                    DEP=DEP,
                    FORM=FORM,
                    LOG=LOG,
                    N=N,
                    PND=PND,
                )

                model = extract_model_rstyle(
                    x=df_term,
                    y=tobit_out,
                    DEP=DEP,
                    INDEP=indep_term,
                    LAG=lag_scalar,
                    ITER=int(term),
                )

                results.append(model)

    return results


def export_res_to_csv(res, csv_path):
    df = pd.DataFrame(res)
    df.to_csv(csv_path, index=False)
    return df


def compute_varcovar_rstyle(hessian0, x_std, y_std):
    """
    Port of the varcovar unstandardization logic in censReg().
    Returns covariance matrix for all parameters including logSigma.
    """
    H0 = np.asarray(hessian0, dtype=float)
    p = H0.shape[0]

    tau = y_std["sigma"][0] / np.asarray(x_std["sigma"], dtype=float)

    zero = np.zeros((p - 1, p), dtype=float)
    top = np.concatenate([np.asarray(x_std["mu"], dtype=float), [0.0]])
    u_inv = np.diag(np.concatenate([tau, [1.0]])) - np.vstack([top, zero])

    eigvals, eigvecs = np.linalg.eigh(H0)

    e_inv = np.full(p, np.inf, dtype=float)
    bad = np.round(eigvals, 14) >= 0
    e_inv[~bad] = 1.0 / eigvals[~bad]

    v = eigvecs @ np.diag(-e_inv) @ eigvecs.T
    varcovar = u_inv @ v @ u_inv.T
    return varcovar


def coefficient_pvalues_rstyle(estimate, varcovar, param_names):
    """
    Wald z-test p-values from estimate and covariance matrix.
    """
    estimate = np.asarray(estimate, dtype=float)
    se = np.sqrt(np.diag(varcovar))
    z = estimate / se
    p = 2 * norm.sf(np.abs(z))

    out = {}
    for i, nm in enumerate(param_names):
        out[nm] = {
            "estimate": estimate[i],
            "se": se[i],
            "z": z[i],
            "p": p[i],
        }
    return out


def merge_python_r(py_df, r_csv_path):
    r_df = pd.read_csv(r_csv_path)

    keep_r = [
        "KEY",
        "WELL",
        "ITER",
        "LAG",
        "p_trend",
        "p_interp",
        "p_event",
        "AIC",
        "BIC",
        "AIC_EVENT",
        "AIC_INTERP",
        "AIC_NULL",
        "BIC_EVENT",
        "BIC_INTERP",
        "BIC_NULL",
        "n_obs",
        "beta_intercept",
        "beta_interp",
        "beta_event",
        "model_type",
        "fit_ok",
    ]
    keep_r = [c for c in keep_r if c in r_df.columns]
    r_df = r_df[keep_r].copy()

    merged = py_df.merge(
        r_df, on=["KEY", "WELL", "ITER"], how="outer", suffixes=("_py", "_r")
    )

    compare_cols = [
        "LAG",
        "p_trend",
        "p_interp",
        "p_event",
        "AIC",
        "BIC",
        "AIC_EVENT",
        "AIC_INTERP",
        "AIC_NULL",
        "BIC_EVENT",
        "BIC_INTERP",
        "BIC_NULL",
        "n_obs",
        "beta_intercept",
        "beta_interp",
        "beta_event",
    ]

    for c in compare_cols:
        py = f"{c}_py"
        rr = f"{c}_r"
        if py in merged.columns and rr in merged.columns:
            merged[f"{c}_diff"] = pd.to_numeric(
                merged[py], errors="coerce"
            ) - pd.to_numeric(merged[rr], errors="coerce")
            merged[f"{c}_absdiff"] = merged[f"{c}_diff"].abs()

    return merged


import plotly.express as px


def plot_py_vs_r(merged, col):
    xcol = f"{col}_r"
    ycol = f"{col}_py"

    df = merged[[xcol, ycol, "KEY", "WELL", "ITER"]].copy()
    df = df.dropna()

    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        hover_data=["KEY", "WELL", "ITER"],
        title=f"Python vs R: {col}",
    )

    if not df.empty:
        xmin = min(df[xcol].min(), df[ycol].min())
        xmax = max(df[xcol].max(), df[ycol].max())
        fig.add_shape(
            type="line", x0=xmin, y0=xmin, x1=xmax, y1=xmax, line=dict(dash="dash")
        )

    fig.show()
