from __future__ import annotations
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------------
# removeDups (Waterlevel + River only — exact logic subset)
# --------------------------------------------------------------------------------------
def remove_dups(X: pd.DataFrame, TYPE="River", MAN="MAN", TRANS="XD", WLDUPE="MAN"):
    X = X.copy()

    if TYPE == "River":
        return X.groupby(["NAME", "EVENT", "UNITS"], as_index=False)["STAGE"].mean()

    if TYPE == "Waterlevel":
        # Mean by TYPE
        X = X.groupby(["NAME", "EVENT", "TYPE", "UNITS"], as_index=False)["WLE"].mean()

        WL = X[["NAME", "EVENT"]].drop_duplicates()

        MAN_DATA = X[X["TYPE"] == MAN].copy()
        TRANS_DATA = X[X["TYPE"] == TRANS].copy()

        WLS = WL.merge(
            MAN_DATA, on=["NAME", "EVENT"], how="left", suffixes=("", "_MAN")
        )

        if WLDUPE == "MAN":
            if not TRANS_DATA.empty:
                WLS = WL.merge(
                    TRANS_DATA,
                    on=["NAME", "EVENT"],
                    how="left",
                    suffixes=("", "_TRANS"),
                )
                WLS["WLE"] = np.where(WLS["WLE"].isna(), WLS["WLE_TRANS"], WLS["WLE"])
                WLS["TYPE"] = np.where(WLS["WLE"].isna(), "TRANS", "MAN")
            return WLS[["NAME", "EVENT", "WLE", "TYPE", "UNITS"]]

    return X


# --------------------------------------------------------------------------------------
# interpRS
# --------------------------------------------------------------------------------------
def interp_rs(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["EVENT"] = pd.to_datetime(X["EVENT"])

    full = pd.DataFrame({"EVENT": pd.date_range(X["EVENT"].min(), X["EVENT"].max())})
    full = full.merge(X, on="EVENT", how="left")

    full["INTERP"] = full["STAGE"].interpolate()
    full["NAME"] = X["NAME"].iloc[0]

    return full


# --------------------------------------------------------------------------------------
# interpWL
# --------------------------------------------------------------------------------------
def interp_wl(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["EVENT"] = pd.to_datetime(X["EVENT"])

    out = []
    for name, Y in X.groupby("NAME"):
        if len(Y) == 1:
            Y["INTERP"] = Y["WLE"]
            out.append(Y)
            continue

        full = pd.DataFrame(
            {"EVENT": pd.date_range(Y["EVENT"].min(), Y["EVENT"].max())}
        )
        full = full.merge(Y, on="EVENT", how="left")

        full["INTERP"] = full["WLE"].interpolate()
        full["NAME"] = name
        out.append(full)

    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------
# combineData (WL + RS branch only)
# --------------------------------------------------------------------------------------
def combine_data(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    Y = Y.copy()

    # rename
    X = X.rename(columns={"NAME": "RS_NAME", "UNITS": "RS_UNITS"})
    Y = Y.rename(columns={"NAME": "WL_NAME", "UNITS": "WL_UNITS"})

    out = []

    for name, x in Y.groupby("WL_NAME"):
        tmp = pd.merge(x, X, on="EVENT", how="left")

        tmp["WL_NAME"] = name
        tmp["RS_UNITS"] = X["RS_UNITS"].dropna().unique()[0]

        tmp = tmp[
            [
                "WL_NAME",
                "EVENT",
                "WLE",
                "WL_UNITS",
                "TYPE",
                "RS_NAME",
                "INTERP",
                "RS_UNITS",
            ]
        ]

        out.append(tmp)

    out = pd.concat(out, ignore_index=True)
    out = out.rename(columns={"WL_NAME": "NAME"})

    return out


# --------------------------------------------------------------------------------------
# MAIN: Script 02
# --------------------------------------------------------------------------------------
def run_water_level_import(WL: pd.DataFrame, RS: pd.DataFrame):
    # ---- WL ----
    WL = WL[["NAME", "EVENT", "WLE", "TYPE", "UNITS"]].copy()
    WL["EVENT"] = pd.to_datetime(WL["EVENT"])
    WL = WL.dropna(subset=["WLE"])

    WL = remove_dups(WL, TYPE="Waterlevel")
    WL = interp_wl(WL)
    WL = WL.sort_values("EVENT")

    # ---- RS ----
    RS = RS[["NAME", "EVENT", "STAGE", "UNITS"]].copy()
    RS["EVENT"] = pd.to_datetime(RS["EVENT"])
    RS = RS.dropna(subset=["STAGE"])

    RS = remove_dups(RS, TYPE="River")
    RS = interp_rs(RS)
    RS = RS.sort_values("EVENT")

    # ---- COMBINE ----
    WL_RS = combine_data(RS, WL)

    return WL_RS
