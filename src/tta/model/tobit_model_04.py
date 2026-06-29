import os
import re
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from tqdm import tqdm


def match_arg(value: Optional[str], choices: list[str], arg_name: str) -> str:
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
    """
    Estimate the optimal cross-correlation lag between chemistry and river stage.

    Calls crosscor_r_bridge once with all lag values 0..MAXLAG and returns the
    lag with the highest absolute CCF. Returns an empty LAG array if the well
    has fewer than N observations or proportion of non-detects exceeds PND.

    Parameters
    ----------
    x : pd.DataFrame
        Chemistry data for the current well/TERM.
    y : pd.DataFrame
        Full water-level / stage interpolation data for the well.
    DEP : str
        Dependent variable column name (e.g. ``"VAL"``).
    INDEP : str
        Independent (stage) column name (e.g. ``"INTERP"``).
    MAXLAG : int
        Maximum lag (days) to test.
    N : int
        Minimum number of observations required to compute the lag.
    PND : float
        Maximum allowed proportion of non-detects.
    r_script_path : str | Path
        Path to ``crosscor_r_bridge.R``.

    Returns
    -------
    dict with keys:
        ``"COD"`` – DataFrame of (acf, lag) pairs across all tested lags.
        ``"LAG"`` – Array of lag value(s) with maximum |acf|; empty if skipped.
    """
    X_0 = x.loc[~pd.isna(x[DEP])].copy()
    n = len(X_0)
    PNDS = X_0["NDS"].sum() / n if n > 0 else np.nan

    if len(X_0) >= N and PNDS <= PND:
        x1 = _to_event_numeric_rstyle(X_0["EVENT"])
        x2 = _to_event_numeric_rstyle(y["EVENT"])
        y1 = X_0[DEP].to_numpy()
        y2 = y[INDEP].to_numpy()
        ccf = crosscor_r_bridge(x1, y1, x2, y2, range(0, int(MAXLAG) + 1), r_script_path)
        max_abs = np.nanmax(np.abs(ccf["acf"].to_numpy()))
        lag = ccf.loc[np.abs(ccf["acf"]) == max_abs, "lag"].to_numpy()
    else:
        ccf = pd.DataFrame(columns=["acf", "lag"])
        lag = np.array([], dtype=float)

    return {"COD": ccf, "LAG": lag}


def crosscor_r_bridge(x1, y1, x2, y2, lags, r_script_path):
    """
    Call the R cross-correlation script for all lag values in a single subprocess.

    The series (x1/y1, x2/y2) and the lag vector are written as columns in one
    CSV; they may differ in length and are NaN-padded to match. The R script
    loops over the lags internally and returns one row per lag.

    Parameters
    ----------
    x1, y1 : array-like
        Numeric day-indices and values for series 1 (chemistry).
    x2, y2 : array-like
        Numeric day-indices and values for series 2 (river stage).
    lags : iterable of int
        Lag values (days) to test, e.g. ``range(0, MAXLAG + 1)``.
    r_script_path : str | Path
        Path to ``crosscor_r_bridge.R``.

    Returns
    -------
    pd.DataFrame
        Columns ``acf`` and ``lag``, one row per tested lag.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, "in.csv")
        outfile = os.path.join(tmpdir, "out.csv")

        n1 = len(x1)
        n2 = len(x2)
        lags_list = list(lags)
        n_lags = len(lags_list)
        n = max(n1, n2, n_lags)

        df = pd.DataFrame(
            {
                "x1":  list(x1)  + [np.nan] * (n - n1),
                "y1":  list(y1)  + [np.nan] * (n - n1),
                "x2":  list(x2)  + [np.nan] * (n - n2),
                "y2":  list(y2)  + [np.nan] * (n - n2),
                "lag": lags_list + [np.nan] * (n - n_lags),
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

        return pd.read_csv(outfile)


def censreg_ll_test(beta, X, y, left, right=None):
    """
    Per-observation log-likelihood for the censored normal model.

    Left-censored observations contribute ``log Φ((y − Xβ) / σ)``,
    uncensored observations contribute the normal log-density minus ``log σ``,
    and right-censored observations contribute ``log(1 − Φ(...))``.

    Parameters
    ----------
    beta : array-like, shape (p+1,)
        Coefficients followed by ``log(σ)`` as the last element.
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response vector (possibly log-transformed).
    left : array-like of bool, shape (n,)
        True for left-censored (non-detect) observations.
    right : array-like of bool or None, shape (n,)
        True for right-censored observations. Defaults to all False.

    Returns
    -------
    np.ndarray, shape (n,)
        Per-observation log-likelihood contributions.
    """
    beta = np.asarray(beta, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    left = np.asarray(left, dtype=bool)

    if right is None:
        right = np.zeros(len(y), dtype=bool)
    else:
        right = np.asarray(right, dtype=bool)

    # Suppress transient floating-point warnings that arise when the optimizer
    # explores extreme log(sigma) values (sigma → 0 → divide-by-zero → huge
    # residuals → overflow in norm.logpdf).  The optimiser recovers; the final
    # result is unaffected.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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
    """
    Column-wise standardization matching R's ``scale()``.

    Columns with zero or NaN standard deviation are left at their original
    values (mu=0, sigma=1) to avoid division by zero.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p)
        Data to standardize.

    Returns
    -------
    dict with keys:
        ``"Z"`` – Standardized array, shape (n, p).
        ``"mu"`` – Column means, shape (p,).
        ``"sigma"`` – Column standard deviations (ddof=1), shape (p,).
    """
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
    """
    Port of R ``maxLik::logLik``-attribute wrapper used by ``maxNR``.

    Evaluates ``fnOrig`` at ``theta`` and computes the gradient and Hessian
    (numerically if analytic versions are not supplied). Fixed parameters are
    masked to NaN in the gradient and Hessian.

    Parameters
    ----------
    theta : array-like
        Parameter vector.
    fnOrig : callable
        Per-observation log-likelihood function.
    gradOrig : callable or None
        Analytic gradient; numeric approximation used if None.
    hessOrig : callable or None
        Analytic Hessian; numeric approximation used if None.
    fixed : array-like of bool or None
        Mask of fixed (non-optimised) parameters.
    sumObs : bool
        If True, return scalar log-likelihood instead of per-observation vector.
    returnHessian : bool
        If False, skip Hessian computation.

    Returns
    -------
    dict with keys ``"value"``, ``"gradient"``, ``"hessian"``.
    """
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
    """
    Central-difference numerical gradient, shape (nVal, nParam).

    Parameters
    ----------
    f : callable
        Function returning a 1-D array of length nVal.
    t0 : array-like
        Parameter vector at which to evaluate the gradient.
    eps : float
        Step size for finite differences.
    fixed : array-like of bool or None
        Fixed parameters are skipped and left as NaN.

    Returns
    -------
    np.ndarray, shape (nVal, nParam)
    """
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

        with np.errstate(invalid="ignore"):
            grad[:, i] = (ft2 - ft1) / eps

    return grad


def prepare_fixed_rstyle(start, activePar=None, fixed=None):
    """
    Port of R ``maxLik::prepareFixed``; returns a boolean mask of fixed params.

    Parameters
    ----------
    start : array-like
        Initial parameter vector (used to determine length).
    activePar : array-like or None
        Boolean or integer indices of active (non-fixed) parameters.
    fixed : array-like or None
        Boolean mask or integer indices of fixed parameters.

    Returns
    -------
    np.ndarray of bool, shape (nParam,)
        True where a parameter is fixed.
    """
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
    """
    Sum per-observation gradients to a parameter-length vector.

    Parameters
    ----------
    gr : array-like
        Gradient array, either shape (nObs, nParam) or (nParam,).
    nParam : int
        Number of parameters.

    Returns
    -------
    np.ndarray, shape (nParam,)
    """
    gr = np.asarray(gr)

    if gr.ndim > 1:
        gr = np.sum(gr, axis=0)
    else:
        if nParam == 1 and len(gr) > 1:
            gr = np.sum(gr)

    return gr


def sum_keep_attr_rstyle(x, keepNames=False, na_rm=False):
    """
    Scalar sum of an array, optionally ignoring NaN; port of R ``sum()``.

    Parameters
    ----------
    x : array-like
        Values to sum.
    keepNames : bool
        Unused; present for API compatibility with the R port.
    na_rm : bool
        If True, NaN values are ignored (maps to R ``na.rm=TRUE``).

    Returns
    -------
    float
    """
    x_arr = np.asarray(x)

    if na_rm:
        value = np.nansum(x_arr)
    else:
        value = np.sum(x_arr)

    return value


def observation_gradient_rstyle(g, nParam):
    """
    Return True if ``g`` is a per-observation gradient matrix (nObs × nParam).

    Port of R ``maxLik::observationGradient``.

    Parameters
    ----------
    g : array-like
        Gradient array to inspect.
    nParam : int
        Number of parameters.

    Returns
    -------
    bool
    """
    g = np.asarray(g)

    if g.ndim == 1:
        if nParam == 1 and len(g) > 1:
            return True
        return False

    if g.shape[0] == 1:
        return False

    return True


def maxim_message_rstyle(code):
    """
    Human-readable convergence message for a ``maxNR`` termination code.

    Parameters
    ----------
    code : int
        Termination code returned by ``maxnr_compute_rstyle``.

    Returns
    -------
    str
    """
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


def maxnr_init_rstyle(fn, start, fixed=None, bhhhHessian=False, **kwargs):
    """
    Evaluate the log-likelihood, gradient, and Hessian at the starting point.

    Returns an early-exit dict (with ``"code"`` key) if the initial value is
    NaN or +Inf, otherwise returns the initialisation state for the main loop.

    Parameters
    ----------
    fn : callable
        Log-likelihood wrapper (``loglik_attr_rstyle``).
    start : array-like
        Initial parameter vector.
    fixed : array-like of bool or None
        Fixed parameter mask.
    bhhhHessian : bool
        If True, use BHHH outer-product Hessian (not currently used).

    Returns
    -------
    dict
        Either ``{"code": int, ...}`` for early exit, or the init state dict.
    """
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
    """
    Numerical Hessian via central differences on the gradient.

    Parameters
    ----------
    f : callable
        Scalar log-likelihood function (used only if ``grad`` is None).
    grad : callable or None
        Gradient function; required — direct Hessian path is not implemented.
    t0 : array-like
        Parameter vector at which to evaluate.
    eps : float
        Step size for finite differences.
    fixed : array-like of bool or None
        Fixed parameter mask.

    Returns
    -------
    np.ndarray, shape (nParam, nParam)
    """
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
    """
    Compute one Newton-Raphson step with ridge regularisation if needed.

    If the active Hessian is not negative-definite, a diagonal ridge
    (``lambda * I``) is added until it is, matching R's ``maxNR`` behaviour.

    Parameters
    ----------
    start0 : array-like
        Current parameter vector.
    f0 : float
        Current log-likelihood.
    G0 : array-like
        Current gradient.
    H0 : array-like
        Current Hessian.
    fixed : array-like of bool or None
        Fixed parameter mask.
    lambdatol, qrtol : float
        Ridge and rank tolerances.

    Returns
    -------
    dict with keys ``"lambda"``, ``"step"``, ``"amount"``, ``"start1"``, ``"H"``.
    """
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
    """
    Backtracking line search for the Newton-Raphson step.

    Halves the step size until the log-likelihood improves or the step falls
    below ``steptol``. If no improvement is found, returns the current point.

    Parameters
    ----------
    fn : callable
        Log-likelihood wrapper.
    start0 : array-like
        Current parameter vector.
    f0 : float or array-like
        Current log-likelihood value(s).
    amount : array-like
        Newton step direction.
    fixed : array-like of bool or None
        Fixed parameter mask.
    returnHessian : bool
        Whether to request the Hessian from ``fn``.
    steptol : float
        Minimum acceptable step size.

    Returns
    -------
    dict with keys ``"step"``, ``"start1"``, ``"f1"``, ``"last_step"``.
    """
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
    """
    Check Newton-Raphson termination criteria; return a code or None.

    Returns None if none of the stopping criteria are met (continue iterating).
    Codes: 1=gradient close to zero, 2=function value tolerance, 3=step too
    small, 5=infinite log-likelihood.

    Parameters
    ----------
    f0, f1 : float or array-like
        Log-likelihood at previous and current iteration.
    G1 : array-like
        Gradient at current iteration.
    step : float
        Accepted step size from backtracking.
    fixed : array-like of bool or None
        Fixed parameter mask.
    gradtol, tol, reltol : float
        Convergence tolerances.

    Returns
    -------
    int or None
    """
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
    """
    Newton-Raphson maximisation; port of R ``maxLik::maxNR``.

    Iterates Newton steps with backtracking until a termination criterion is
    met or ``iterlim`` iterations are exhausted.

    Parameters
    ----------
    fn : callable
        Log-likelihood wrapper returning ``{"value", "gradient", "hessian"}``.
    start : array-like
        Initial parameter vector.
    print_level : int
        Unused; retained for API compatibility with the R port.
    tol, reltol, gradtol, steptol : float
        Convergence tolerances.
    lambdatol, qrtol : float
        Ridge and rank tolerances for the one-step solver.
    iterlim : int
        Maximum number of iterations.
    finalHessian : bool
        If True, include the Hessian in the return dict.
    bhhhHessian : bool
        Not implemented; present for API compatibility.
    fixed : array-like of bool or None
        Fixed parameter mask.

    Returns
    -------
    dict with keys:
        ``"maximum"``, ``"estimate"``, ``"gradient"``, ``"hessian"``,
        ``"code"``, ``"message"``, ``"last_step"``, ``"fixed"``,
        ``"iterations"``, ``"type"``.
    """
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
    """
    Fit a censored-normal (Tobit) regression; port of R ``censReg::censReg``.

    Standardises response and predictors before MLE, then back-transforms
    the estimates to the original scale. The dependent variable is
    log-transformed when ``formula_text`` starts with ``"log("`` or
    ``"log10("``.

    Parameters
    ----------
    formula_text : str
        Formula string, e.g. ``"log(VAL)~INTERP+EVENT"``.
    data : pd.DataFrame
        Data containing all variables referenced in the formula.
        ``"EVENT"`` columns are converted to numeric days since 1970-01-01.
    left : array-like of bool
        True for left-censored (non-detect) observations.
    right : array-like of bool or None
        True for right-censored observations. Defaults to all False.
    start : None
        Custom start values; not yet implemented.
    logLikOnly : bool
        If True, return only the per-observation log-likelihood at the OLS
        starting point (used for testing).

    Returns
    -------
    dict
        Fitted model containing ``"estimate"``, ``"coefficients"``,
        ``"varcovar"``, ``"coef_table"``, ``"maximum"``, ``"nObs"``,
        ``"fitted.values"``, ``"residuals"``, ``"df.residual"``, and others.
    """
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
    """
    Back-transform standardized Tobit estimates to the original scale.

    Port of the unstandardization step in R ``censReg::censReg``.
    The intercept is adjusted to absorb the effect of mean-centering all
    predictors; slopes are rescaled by ``y_sigma / x_sigma_j``; and
    ``logSigma`` is shifted by ``log(y_sigma)``.

    Parameters
    ----------
    estimate0 : array-like, shape (p+1,)
        Standardized estimates, last element is ``logSigma``.
    x_std : dict
        ``{"mu": ..., "sigma": ...}`` from ``standardize_rstyle`` on X.
    y_std : dict
        ``{"mu": ..., "sigma": ...}`` from ``standardize_rstyle`` on y.

    Returns
    -------
    np.ndarray, shape (p+1,)
        Estimates on the original (unstandardized) scale.
    """
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
    """
    Fit all Tobit model variants for a single well/TERM.

    Fits the full model (``FORM``), the intercept-only null, and (when FORM
    has two predictors) the INTERP-only and EVENT-only single-covariate
    models. Returns ``np.nan`` for all four if the minimum observations or
    maximum non-detect proportion criteria are not met.

    Parameters
    ----------
    x : pd.DataFrame
        Chemistry data for one well/TERM, with ``DEP`` and ``"NDS"`` columns.
    DEP : str
        Dependent variable column name.
    FORM : str
        Full model formula string, e.g. ``"log(VAL)~INTERP+EVENT"``.
    LOG : str
        Log transformation: ``"log"``, ``"log10"``, or ``"none"``.
    N : int
        Minimum number of non-missing observations.
    PND : float
        Maximum allowed proportion of non-detects.

    Returns
    -------
    dict with keys ``"CEN"``, ``"CEN_0"``, ``"CEN_2"``, ``"CEN_3"``.
        Each value is either a fitted model dict or ``np.nan``.
    """
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
    """
    Summarise a fitted Tobit result into a flat results dict.

    Computes the likelihood-ratio test (``p_trend``), AIC/BIC for all model
    variants, and extracts coefficients and standard errors.

    Parameters
    ----------
    x : pd.DataFrame
        Chemistry data for the well/TERM (used only for observation counts).
    y : dict
        Output of ``run_tobit_rstyle``: keys ``"CEN"``, ``"CEN_0"``, etc.
    DEP : str
        Dependent variable column name.
    INDEP : list of str
        Independent variable names (length 1 or 2).
    LAG : float
        Applied lag (days).
    MODEL : str
        Model label written to the ``"MODEL"`` output column.
    ITER : int or None
        TERM index written to the ``"ITER"`` output column.

    Returns
    -------
    dict
        Flat results row containing all output columns (p_trend, AIC, BIC,
        beta_*, se_*, p_*, fit_ok, etc.).
    """
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
                "se_intercept": np.nan,
                "se_interp": np.nan,
                "se_event": np.nan,
                "beta_logSigma": np.nan,
                "p_interp": np.nan,
                "p_event": np.nan,
                "SUM_rows": 1,
                "SUM_cols": 1,
                "model_type": "SUM_ROWS_1",
                "fit_ok": False,
                "vcov": None,
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
            "se_intercept": ct.get("(Intercept)", {}).get("se", np.nan),
            "se_interp": ct.get("INTERP", {}).get("se", np.nan),
            "se_event": ct.get("EVENT", {}).get("se", np.nan),
            "beta_logSigma": coef[-1] if len(coef) > 0 else np.nan,
            "p_interp": ct.get("INTERP", {}).get("p", np.nan),
            "p_event": ct.get("EVENT", {}).get("p", np.nan),
            "SUM_rows": 4,
            "SUM_cols": 4,
            "model_type": "INTERP+EVENT",
            "fit_ok": True,
            "vcov": np.asarray(y["CEN"]["varcovar"], dtype=float)[:-1, :-1],
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
                "se_intercept": np.nan,
                "se_interp": np.nan,
                "se_event": np.nan,
                "beta_logSigma": np.nan,
                "p_interp": np.nan,
                "p_event": np.nan,
                "SUM_rows": 1,
                "SUM_cols": 1,
                "model_type": "SUM_ROWS_1",
                "fit_ok": False,
                "vcov": None,
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
            "se_intercept": ct.get("(Intercept)", {}).get("se", np.nan),
            "se_interp": ct.get("INTERP", {}).get("se", np.nan),
            "se_event": ct.get("EVENT", {}).get("se", np.nan),
            "beta_logSigma": coef[-1] if len(coef) > 0 else np.nan,
            "p_interp": ct.get("INTERP", {}).get("p", np.nan),
            "p_event": ct.get("EVENT", {}).get("p", np.nan),
            "SUM_rows": 3,
            "SUM_cols": 4,
            "model_type": INDEP[0],
            "fit_ok": True,
            "vcov": np.asarray(y["CEN"]["varcovar"], dtype=float)[:-1, :-1],
        }


def lag_col_rstyle(X, LAG):
    """
    Shift a 1-D array by LAG positions, filling introduced positions with NaN.

    A positive LAG shifts the series forward (head filled with NaN), matching
    R's ``dplyr::lag(x, n=LAG)``. A negative LAG shifts backward (tail filled
    with NaN).

    Parameters
    ----------
    X : array-like
        Input series.
    LAG : int or float
        Number of positions to shift. 0 or NaN returns a copy unchanged.

    Returns
    -------
    np.ndarray
    """
    X = np.asarray(X, dtype=float)

    if LAG == 0 or np.isnan(LAG):
        return X.copy()

    LAG = int(LAG)

    if LAG < 0:
        k = -LAG
        return np.concatenate([np.full(k, np.nan), X[: len(X) - k]])
    else:
        return np.concatenate([X[LAG:], np.full(LAG, np.nan)])


def _process_well_tobit(args: tuple) -> list:
    """
    Worker for parallel Tobit trend analysis.

    Processes one well across all its TERMs. Must be a module-level function
    so that the Windows ``spawn`` multiprocessing context can pickle it.

    Parameters
    ----------
    args : tuple
        ``(name, df_full, DEP, INDEP, LOG, MAXLAG, N, PND, r_script_path,
        ulags, newrs_names)``

    Returns
    -------
    list of dict
        One result dict per TERM (as returned by ``extract_model_rstyle``).
    """
    name, df_full, DEP, INDEP, LOG, MAXLAG, N, PND, r_script_path, ulags, newrs_names = args

    ulag_applied = ulags is not None and name in ulags and pd.notna(ulags[name])
    is_newrs = newrs_names is not None and name in newrs_names

    if is_newrs:
        lag_origin = "no river stage"
    elif ulag_applied:
        lag_origin = "ULAG"
    else:
        lag_origin = "calculated"

    well_results = []
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
                df_lag["INTERP"] = lag_col_rstyle(df_lag["INTERP"].to_numpy(), -lag_scalar)

            df_term = df_lag[df_lag["TERM"] == term].copy()

            if pd.notna(lag_scalar) and lag_scalar > 0:
                df_term = df_term.loc[~pd.isna(df_term["INTERP"])].copy()

            indep_term = ["INTERP", "EVENT"]

        FORM = create_formula_rstyle(DEP, indep_term, LOG)

        tobit_out = run_tobit_rstyle(x=df_term, DEP=DEP, FORM=FORM, LOG=LOG, N=N, PND=PND)

        model = extract_model_rstyle(
            x=df_term,
            y=tobit_out,
            DEP=DEP,
            INDEP=indep_term,
            LAG=lag_scalar,
            ITER=int(term),
        )
        model["lag_origin"] = lag_origin
        well_results.append(model)

    return well_results


def do_tobit_rstyle(
    x,
    DEP,
    INDEP,
    LOG,
    MAXLAG,
    N,
    PND,
    r_script_path,
    ulags=None,
    newrs_names=None,
    n_workers: Optional[int] = None,
):
    """
    Run Tobit trend analysis for every well/TERM combination in ``x`` in parallel.

    Each well (and all its TERMs) is processed by an independent
    ``ProcessPoolExecutor`` worker. Results are extended in the same order as
    the well name iteration.

    Parameters
    ----------
    x : pd.DataFrame
        Chemistry data with columns ``NAME``, ``TERM``, ``DEP``, ``NDS``,
        ``INTERP``, ``EVENT``.
    DEP : str
        Dependent variable column name (e.g. ``"VAL"``).
    INDEP : list of str
        Independent variable names, e.g. ``["INTERP", "EVENT"]``.
    LOG : str
        Log transformation: ``"log"``, ``"log10"``, or ``"none"``.
    MAXLAG : int
        Maximum lag (days) to test when no pre-computed lag is available.
    N : int
        Minimum number of observations required for model fitting.
    PND : float
        Maximum allowed proportion of non-detects.
    r_script_path : str | Path
        Path to ``crosscor_r_bridge.R``.
    ulags : dict or None
        Mapping of well name → pre-computed lag (days). Wells present in this
        dict skip the cross-correlation step.
    newrs_names : set or None
        Well names fitted with EVENT-only model and lag=0.
    n_workers : int or None
        Number of parallel worker processes. Defaults to ``os.cpu_count()``.

    Returns
    -------
    list of dict
        One result dict per well/TERM (as returned by ``extract_model_rstyle``).
    """
    if x.empty:
        return []

    names = x["NAME"].dropna().unique()
    n = min(len(names), os.cpu_count() or 1) if n_workers is None else n_workers

    args_iter = [
        (
            name,
            x[x["NAME"] == name].copy(),
            DEP, INDEP, LOG, MAXLAG, N, PND, r_script_path,
            ulags, newrs_names,
        )
        for name in names
    ]

    results: list = []
    with ProcessPoolExecutor(max_workers=n) as executor:
        for well_results in tqdm(
            executor.map(_process_well_tobit, args_iter),
            total=len(names),
            desc="Tobit analysis",
            unit="well",
        ):
            results.extend(well_results)

    return results


def compute_varcovar_rstyle(hessian0, x_std, y_std):
    """
    Port of the varcovar unstandardization logic in censReg().
    Returns covariance matrix for all parameters including logSigma.
    """
    H0 = np.asarray(hessian0, dtype=float)
    p = H0.shape[0]

    x_mu = np.asarray(x_std["mu"], dtype=float)
    x_sigma = np.asarray(x_std["sigma"], dtype=float)
    y_sigma = float(y_std["sigma"][0])

    tau = y_sigma / x_sigma

    # Jacobian of original parameters with respect to standardized parameters.
    #
    # intercept = y_mu + y_sigma * beta0_std
    #             - sum(beta_j_std * y_sigma * x_mu_j / x_sigma_j)
    #
    # slopes    = beta_j_std * y_sigma / x_sigma_j
    #
    # logSigma  = logSigma_std + log(y_sigma)
    jac = np.diag(np.concatenate([tau, [1.0]]))

    # Correct intercept row.
    # The old code subtracted x_mu only; it must subtract tau * x_mu.
    jac[0, : len(x_mu)] -= tau * x_mu

    eigvals, eigvecs = np.linalg.eigh(H0)

    e_inv = np.full(p, np.inf, dtype=float)
    bad = np.round(eigvals, 14) >= 0
    e_inv[~bad] = 1.0 / eigvals[~bad]

    v = eigvecs @ np.diag(-e_inv) @ eigvecs.T
    varcovar = jac @ v @ jac.T

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


