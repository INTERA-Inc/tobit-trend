from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _match_arg(value: str, choices: List[str], arg_name: str) -> str:
    if value in choices:
        return value
    prefix = [c for c in choices if c.startswith(value)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) == 0:
        raise ValueError(f"{arg_name} must be one of {choices}; got {value!r}")
    raise ValueError(f"{arg_name} is ambiguous: {value!r} matches {prefix}")


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _year(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.year


def _rbind_nonnull(frames: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
    keep = [x for x in frames if x is not None and len(x) > 0]
    if not keep:
        return pd.DataFrame()
    return pd.concat(keep, ignore_index=True)


# --------------------------------------------------------------------------------------
# removeDups() direct port for River and Waterlevel branches used by Script 02
# --------------------------------------------------------------------------------------


def remove_dups(
    X: pd.DataFrame,
    TYPE: str = "River",
    FIL: bool = False,
    MAN: str = "MAN",
    TRANS: str = "XD",
    WLDUPE: str = "MAN",
) -> pd.DataFrame:
    WLDUPE = _match_arg(WLDUPE, ["MAN", "TRANS"], "WLDUPE")
    X = X.copy()

    if TYPE == "River":
        out = X.groupby(["NAME", "EVENT", "UNITS"], dropna=False, as_index=False)[
            "STAGE"
        ].mean()
        return out

    if TYPE == "Waterlevel":
        X = X.groupby(["NAME", "EVENT", "TYPE", "UNITS"], dropna=False, as_index=False)[
            "WLE"
        ].mean()

        WL = X.loc[:, ["NAME", "EVENT"]].copy()
        WL["MATCH"] = WL["NAME"].astype(str) + "-" + WL["EVENT"].astype(str)
        WL = WL.loc[~WL["MATCH"].duplicated(), ["NAME", "EVENT"]].copy()

        MAN_DATA = X.loc[X["TYPE"] == MAN].copy()
        TRANS_DATA = X.loc[X["TYPE"] == TRANS].copy()

        MAN_DATA = MAN_DATA.rename(
            columns={
                "EVENT": "EVENT_MAN",
                "WLE": "WLE_MAN",
                "TYPE": "TYPE_MAN",
                "UNITS": "UNITS_MAN",
            }
        )
        TRANS_DATA = TRANS_DATA.rename(
            columns={
                "EVENT": "EVENT_TRANS",
                "WLE": "WLE_TRANS",
                "TYPE": "TYPE_TRANS",
                "UNITS": "UNITS_TRANS",
            }
        )

        WLS = WL.merge(
            MAN_DATA,
            left_on=["NAME", "EVENT"],
            right_on=["NAME", "EVENT_MAN"],
            how="left",
            sort=False,
        )

        if WLDUPE == "MAN":
            if len(TRANS_DATA) > 0:
                WLS = TRANS_DATA.merge(
                    WLS,
                    left_on=["NAME", "EVENT_TRANS"],
                    right_on=["NAME", "EVENT"],
                    how="right",
                    sort=False,
                )

                # preserve the event date for MAN-only rows
                WLS["EVENT_OUT"] = WLS["EVENT_TRANS"].where(
                    WLS["EVENT_TRANS"].notna(), WLS["EVENT"]
                )

                WLS["WLE"] = np.where(
                    WLS["WLE_MAN"].isna(), WLS["WLE_TRANS"], WLS["WLE_MAN"]
                )
                WLS["TYPE"] = np.where(WLS["TYPE_MAN"].isna(), "TRANS", "MAN")
                WLS["UNITS"] = np.where(
                    WLS["UNITS_MAN"].isna(), WLS["UNITS_TRANS"], WLS["UNITS_MAN"]
                )

                WLS = WLS.loc[:, ["NAME", "EVENT_OUT", "WLE", "TYPE", "UNITS"]].copy()
                WLS = WLS.rename(columns={"EVENT_OUT": "EVENT"})
            else:
                WLS = WLS.loc[
                    :, ["NAME", "EVENT_MAN", "WLE_MAN", "TYPE_MAN", "UNITS_MAN"]
                ].copy()
                WLS = WLS.rename(
                    columns={
                        "EVENT_MAN": "EVENT",
                        "WLE_MAN": "WLE",
                        "TYPE_MAN": "TYPE",
                        "UNITS_MAN": "UNITS",
                    }
                )

        elif WLDUPE == "TRANS":
            if len(TRANS_DATA) > 0:
                WLS = TRANS_DATA.merge(
                    WLS,
                    left_on=["NAME", "EVENT_TRANS"],
                    right_on=["NAME", "EVENT"],
                    how="right",
                    sort=False,
                )

                # preserve the event date for MAN-only rows
                WLS["EVENT_OUT"] = WLS["EVENT_TRANS"].where(
                    WLS["EVENT_TRANS"].notna(), WLS["EVENT"]
                )

                WLS["WLE"] = np.where(
                    WLS["WLE_TRANS"].isna(), WLS["WLE_MAN"], WLS["WLE_TRANS"]
                )
                WLS["TYPE"] = np.where(WLS["WLE_TRANS"].isna(), "MAN", "TRANS")
                WLS["UNITS"] = np.where(
                    WLS["UNITS_TRANS"].isna(), WLS["UNITS_MAN"], WLS["UNITS_TRANS"]
                )

                WLS = WLS.loc[:, ["NAME", "EVENT_OUT", "WLE", "TYPE", "UNITS"]].copy()
                WLS = WLS.rename(columns={"EVENT_OUT": "EVENT"})
            else:
                WLS = WLS.loc[
                    :, ["NAME", "EVENT_MAN", "WLE_MAN", "TYPE_MAN", "UNITS_MAN"]
                ].copy()
                WLS = WLS.rename(
                    columns={
                        "EVENT_MAN": "EVENT",
                        "WLE_MAN": "WLE",
                        "TYPE_MAN": "TYPE",
                        "UNITS_MAN": "UNITS",
                    }
                )

        return WLS

    return X


# --------------------------------------------------------------------------------------
# interpRS() direct port
# --------------------------------------------------------------------------------------


def interp_rs(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["EVENT"] = _to_date(X["EVENT"])

    min_event = X["EVENT"].min()
    max_event = X["EVENT"].max()

    DT = pd.DataFrame({"EVENT": pd.date_range(min_event, max_event, freq="D")})
    DT = DT.merge(X, on="EVENT", how="left", sort=False)

    DT["INTERP"] = DT["STAGE"].interpolate(method="linear", limit_direction="both")
    DT["NAME"] = X["NAME"].dropna().iloc[0]

    return DT


# --------------------------------------------------------------------------------------
# interpWL() direct port
# Included for completeness because you uploaded it, but Script 02 does not use it
# since prepDataTrends() default WLINTERP=FALSE and Script 02 does not override that.
# --------------------------------------------------------------------------------------


def interp_wl(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["EVENT"] = _to_date(X["EVENT"])

    out = []
    for _, Y in X.groupby("NAME", sort=False):
        min_event = Y["EVENT"].min()
        max_event = Y["EVENT"].max()

        if len(Y) == 1:
            DT = Y.copy()
            DT["INTERP"] = DT["WLE"]
        else:
            DT = pd.DataFrame({"EVENT": pd.date_range(min_event, max_event, freq="D")})
            DT = DT.merge(Y, on="EVENT", how="left", sort=False)
            DT["INTERP"] = DT["WLE"].interpolate(
                method="linear", limit_direction="both"
            )
            DT["NAME"] = Y["NAME"].dropna().iloc[0]

        out.append(DT)

    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------
# prepDataTrends() direct port for RS + WL path used by Script 02
# --------------------------------------------------------------------------------------


def prep_data_trends(
    CHEM: Optional[pd.DataFrame] = None,
    RS: Optional[pd.DataFrame] = None,
    WL: Optional[pd.DataFrame] = None,
    CHEMCOL: List[str] = None,
    RSCOL: List[str] = None,
    WLCOL: List[str] = None,
    MINDATE: pd.Timestamp = pd.Timestamp("1994-01-01"),
    MAXDATE: pd.Timestamp = pd.Timestamp("2014-12-31"),
    FIL: bool = False,
    MDLSUB: float = 1,
    MAN: str = "MAN",
    TRANS: str = "XD",
    WLDUPE: str = "MAN",
    WLINTERP: bool = False,
    DAILY: bool = True,
    POSITIVE: bool = True,
):
    if CHEMCOL is None:
        CHEMCOL = ["NAME", "EVENT", "ANALYTE", "DT", "RESULT", "LABQ", "UNITS", "RL"]
    if RSCOL is None:
        RSCOL = ["NAME", "EVENT", "STAGE", "UNITS"]
    if WLCOL is None:
        WLCOL = ["NAME", "EVENT", "WLE", "TYPE", "UNITS"]

    if CHEM is not None and len(CHEMCOL) != 8:
        raise ValueError(
            f"Missing required columns. Chemistry dataset requires 8 columns. Dataset only has {len(CHEMCOL)} columns."
        )
    if RS is not None and len(RSCOL) != 4:
        raise ValueError(
            f"Missing required columns. River stage dataset requires 4 columns. Dataset only has {len(RSCOL)} columns."
        )
    if WL is not None and len(WLCOL) != 5:
        raise ValueError(
            f"Missing required columns. Water Level dataset requires 5 columns. Dataset only has {len(WLCOL)} columns."
        )

    RS_out = None
    WL_out = None

    if RS is not None:
        RS_out = RS.copy()
        rs_names = pd.unique(
            RS_out["GAUGE"] if "GAUGE" in RS_out.columns else RS_out[RSCOL[0]]
        )
        if len(rs_names) > 1:
            raise ValueError("Multiple River Stage Measurement Locations.")

        RS_out = RS_out.loc[:, RSCOL].copy()
        RS_out.columns = ["NAME", "EVENT", "STAGE", "UNITS"]

        if DAILY:
            RS_out["EVENT"] = _to_date(RS_out["EVENT"])
        else:
            RS_out["EVENT"] = pd.to_datetime(RS_out["EVENT"], errors="coerce")

        RS_out = RS_out.loc[
            (_to_date(RS_out["EVENT"]) >= pd.to_datetime(MINDATE))
            & (_to_date(RS_out["EVENT"]) <= pd.to_datetime(MAXDATE))
        ].copy()

        RS_out = RS_out.loc[~RS_out["STAGE"].isna()].copy()
        RS_out = remove_dups(RS_out, TYPE="River")
        RS_out = interp_rs(RS_out)
        RS_out = RS_out.sort_values("EVENT").reset_index(drop=True)

    if WL is not None:
        WLDUPE = _match_arg(WLDUPE, ["MAN", "TRANS"], "WLDUPE")

        WL_out = WL.copy()
        WL_out = WL_out.loc[:, WLCOL].copy()
        WL_out.columns = ["NAME", "EVENT", "WLE", "TYPE", "UNITS"]

        if DAILY:
            WL_out["EVENT"] = _to_date(WL_out["EVENT"])
        else:
            WL_out["EVENT"] = pd.to_datetime(WL_out["EVENT"], errors="coerce")

        WL_out = WL_out.loc[
            (_to_date(WL_out["EVENT"]) >= pd.to_datetime(MINDATE))
            & (_to_date(WL_out["EVENT"]) <= pd.to_datetime(MAXDATE))
        ].copy()

        WL_out = WL_out.loc[~WL_out["WLE"].isna()].copy()
        WL_out = remove_dups(
            WL_out, TYPE="Waterlevel", MAN=MAN, TRANS=TRANS, WLDUPE=WLDUPE
        )

        if WLINTERP:
            WL_out = interp_wl(WL_out)

        WL_out = WL_out.sort_values("EVENT").reset_index(drop=True)

    if CHEM is None and RS_out is not None and WL_out is None:
        return RS_out
    if CHEM is None and RS_out is None and WL_out is not None:
        return WL_out
    if CHEM is None and RS_out is not None and WL_out is not None:
        return {"RS": RS_out, "WL": WL_out}

    raise NotImplementedError(
        "This direct port only implements the RS + WL path used by Script 02."
    )


# --------------------------------------------------------------------------------------
# combineData() direct port for WL + RS branch
# --------------------------------------------------------------------------------------


def combine_data(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    Y = Y.copy()

    if not ("WLE" in Y.columns and "STAGE" in X.columns):
        raise ValueError(
            "This direct port only implements the WL + RS combineData branch."
        )

    X = X.rename(columns={"NAME": "RS_NAME", "UNITS": "RS_UNITS"})
    Y = Y.rename(columns={"NAME": "WL_NAME", "UNITS": "WL_UNITS"})

    out = []
    for _, x in Y.groupby("WL_NAME", sort=False):
        # data.table: x[X]  => keep X rows, match x on EVENT
        TMP = X.merge(x, on="EVENT", how="left", sort=False)

        wl_name_vals = x["WL_NAME"].dropna().unique()
        TMP["WL_NAME"] = wl_name_vals[0] if len(wl_name_vals) > 0 else np.nan

        rs_units_vals = X.loc[~X["RS_UNITS"].isna(), "RS_UNITS"].unique()
        TMP["RS_UNITS"] = rs_units_vals[0] if len(rs_units_vals) > 0 else np.nan

        TMP = TMP.loc[
            :,
            [
                "WL_NAME",
                "EVENT",
                "WLE",
                "WL_UNITS",
                "TYPE",
                "RS_NAME",
                "INTERP",
                "RS_UNITS",
            ],
        ].copy()

        out.append(TMP)

    Y_COMB = pd.concat(out, ignore_index=True)
    Y_COMB = Y_COMB.rename(columns={"WL_NAME": "NAME"})
    return Y_COMB


# --------------------------------------------------------------------------------------
# Script 02 direct port
# --------------------------------------------------------------------------------------


def run_water_level_import(
    wl: pd.DataFrame,
    river_stage: pd.DataFrame,
    dist: pd.DataFrame,
    stagedist: pd.DataFrame,
    well: pd.DataFrame,
    screen: pd.DataFrame,
    yr: int,
) -> pd.DataFrame:
    # --Import River Stage Data--#
    STAGE = river_stage.copy()
    STAGE["EVENT"] = _to_date(STAGE["EVENT"])
    STAGE = STAGE.loc[_year(STAGE["EVENT"]) <= yr].copy()

    # --Load the new WL file--#
    WL = wl.loc[:, ["NAME", "EVENT", "VAL", "TYPE"]].copy()
    WL["EVENT"] = _to_date(WL["EVENT"])
    WL.loc[WL["TYPE"] == "MAN_HEIS", "TYPE"] = "MAN"

    # --Subset Data for OUS of Interest--#
    WELL = well.copy()
    SCREEN = screen.copy()
    DIST = dist.copy()
    STAGEDIST = stagedist.copy()

    WELLS = WELL.loc[WELL["OU"].isin(["100-KR-4", "100-HR-3-D", "100-HR-3-H"])].copy()
    WELLS = WELLS.loc[
        ~WELLS["STATUS"].isin(["DECOMMISSIONED-V", "DRILLING CANCELLED"]), "NAME"
    ].copy()

    WL["UNITS"] = "m"
    WL = WL.loc[WL["NAME"].isin(WELLS)].copy()
    WL = WL.loc[_year(WL["EVENT"]) <= yr].copy()

    # --Combine Water-Level and River Gauge Data--#
    WLRS = []
    for _, X in WL.groupby("NAME", sort=False):
        # --Limit Trend Analysis to Data After 2008--#
        if len(X) > 2 and X["EVENT"].dt.year.max() >= 2008:
            WELL_NAME = X["NAME"].iloc[0]

            gauge_vals = STAGEDIST.loc[STAGEDIST["NAME"] == WELL_NAME, "STAGE"]
            GAUGE = gauge_vals.iloc[0] if len(gauge_vals) > 0 else None

            # --Extract Gauge Data--#
            if GAUGE is None or pd.isna(GAUGE) or len(gauge_vals) == 0:
                RS = pd.DataFrame(
                    {
                        "GAUGE": "PRD",
                        "EVENT": STAGE["EVENT"],
                        "RS": STAGE["PRD"],
                        "UNITS": "m",
                    }
                )
            else:
                if GAUGE not in STAGE.columns:
                    raise KeyError(
                        f"Gauge column {GAUGE!r} not found in river_stage input."
                    )
                RS = pd.DataFrame(
                    {
                        "GAUGE": GAUGE,
                        "EVENT": STAGE["EVENT"],
                        "RS": STAGE[GAUGE],
                        "UNITS": "m",
                    }
                )

            WLCOL = ["NAME", "EVENT", "VAL", "TYPE", "UNITS"]
            RSCOL = ["GAUGE", "EVENT", "RS", "UNITS"]

            DATA = prep_data_trends(
                RS=RS,
                RSCOL=RSCOL,
                WL=X,
                WLCOL=WLCOL,
                MAXDATE=pd.Timestamp(year=yr, month=12, day=31),
            )

            WL_RS_one = combine_data(X=DATA["RS"], Y=DATA["WL"])
            WLRS.append(WL_RS_one)

    WL_RS = _rbind_nonnull(WLRS)

    # --Combine Water-level Data with Distance to River--#
    WL_RS = WL_RS.merge(DIST, on="NAME", how="left", sort=False)

    # --Subset Datasets by OU--#
    COORDS = WELL.loc[:, ["NAME", "XCOORDS", "YCOORDS", "ZCOORDS"]].copy()
    WL_RS = WL_RS.merge(COORDS, on="NAME", how="left", sort=False)

    # --Add Well Screen Data--#
    WL_RS = WL_RS.merge(SCREEN, on="NAME", how="left", sort=False)

    # --Add Well OU--#
    AREA = WELL.loc[:, ["NAME", "OU"]].copy()
    WL_RS = WL_RS.merge(AREA, on="NAME", how="left", sort=False)

    # --Subset Final Dataset--#
    WL_RS = WL_RS.loc[
        :,
        [
            "NAME",
            "XCOORDS",
            "YCOORDS",
            "ZCOORDS",
            "TOP",
            "BOT",
            "OU",
            "DIST",
            "EVENT",
            "WLE",
            "WL_UNITS",
            "RS_NAME",
            "INTERP",
            "RS_UNITS",
        ],
    ].copy()

    return WL_RS
