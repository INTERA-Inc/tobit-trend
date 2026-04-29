from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def _validate_required_columns(
    df: pd.DataFrame, required: list[str], df_name: str
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def _load_well(well: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(well, pd.DataFrame):
        df = well.copy()
    else:
        df = pd.read_csv(well)

    _validate_required_columns(
        df,
        ["NAME", "OU", "STATUS", "XCOORDS", "YCOORDS", "ZCOORDS"],
        "well",
    )
    return df


def _load_gauge(gauge: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(gauge, pd.DataFrame):
        df = gauge.copy()
    else:
        df = pd.read_csv(gauge)

    _validate_required_columns(df, ["WELL_NAME", "EASTING", "NORTHING"], "gauge")
    return df


def _load_river(river_shapefile: str | Path | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(river_shapefile, gpd.GeoDataFrame):
        rk = river_shapefile.copy()
    else:
        rk = gpd.read_file(river_shapefile)
    if len(rk) != 1:
        raise ValueError(
            f"Expected rivKrigeHigh to contain exactly 1 feature, found {len(rk)}."
        )
    if rk.crs is None:
        raise ValueError("River shapefile/GeoDataFrame has no CRS.")
    return rk


def run_calculate_distance(
    well: str | Path | pd.DataFrame,
    gauge: str | Path | pd.DataFrame,
    river_shapefile: str | Path | gpd.GeoDataFrame,
) -> dict[str, Any]:
    """
    Direct port of 00_CalculateDistance_clean.R / .txt.

    Parameters
    ----------
    well
        WELL.csv path or DataFrame. Must contain:
        NAME, OU, STATUS, XCOORDS, YCOORDS, ZCOORDS
    gauge
        River gauge CSV path or DataFrame. Must contain:
        WELL_NAME, EASTING, NORTHING
    river_shapefile
        rivKrigeHigh shapefile path or GeoDataFrame.
        Must contain exactly 1 feature.
    Returns
    -------
    "DIST": pandas.DataFrame,
    "STAGEDIST": pandas.DataFrame,
    """
    well_df = _load_well(well)
    gauge_df = _load_gauge(gauge)
    rk = _load_river(river_shapefile)

    # -------------------------------------------------------------------------
    # Subset Data for OUs of Interest
    # R:
    # WELLS <- WELL[OU %in% c('100-KR-4','100-HR-3-D','100-HR-3-H')]
    # WELLS <- WELLS[!STATUS %in% c('DECOMMISSIONED-V','DRILLING CANCELLED')]$NAME
    # -------------------------------------------------------------------------
    wells_filtered = well_df.loc[
        well_df["OU"].isin(["100-KR-4", "100-HR-3-D", "100-HR-3-H"])
    ]
    wells_filtered = wells_filtered.loc[
        ~wells_filtered["STATUS"].isin(["DECOMMISSIONED-V", "DRILLING CANCELLED"]),
        "NAME",
    ]

    # -------------------------------------------------------------------------
    # Calculate Distance to River
    # R:
    # COORDS <- subset(WELL, select=c(NAME,XCOORDS, YCOORDS, ZCOORDS))
    # DIST <- subset(COORDS, COORDS$NAME %in% WELLS)
    # coordinates(DIST) <- ~XCOORDS+YCOORDS
    # projection(DIST)  <- projection(RK)
    # D <- as.numeric(gDistance(DIST, RK, byid = TRUE))
    # DIST <- data.table(NAME = DIST$NAME, DIST = D)
    # -------------------------------------------------------------------------
    coords = well_df.loc[:, ["NAME", "XCOORDS", "YCOORDS", "ZCOORDS"]].copy()
    dist_source = coords.loc[coords["NAME"].isin(wells_filtered)].copy()

    dist_points = gpd.GeoDataFrame(
        dist_source.copy(),
        geometry=gpd.points_from_xy(dist_source["XCOORDS"], dist_source["YCOORDS"]),
        crs=rk.crs,
    )

    river_geom = rk.geometry.iloc[0]
    d = dist_points.geometry.distance(river_geom).astype(float).to_numpy()

    dist = pd.DataFrame(
        {
            "NAME": dist_source["NAME"].to_numpy(),
            "DIST": d,
        }
    )

    # -------------------------------------------------------------------------
    # Calculate Distance from Each Gauge
    # R:
    # STAGEDIST <- subset(COORDS,COORDS$NAME %in% WELLS)
    # for(i in 1:nrow(GAUGE)) {
    #   G <- sub("-.*", "",GAUGE$WELL_NAME[i])
    #   x <- GAUGE$EASTING[i]
    #   y <- GAUGE$NORTHING[i]
    #   nm <- paste0(G,'_GAUGE')
    #   STAGEDIST[[nm]] <- sqrt(((STAGEDIST$XCOORDS - x)^2) + ((STAGEDIST$YCOORDS - y)^2))
    # }
    # -------------------------------------------------------------------------
    stagedist = coords.loc[coords["NAME"].isin(wells_filtered)].copy()

    for i in range(len(gauge_df)):
        g = str(gauge_df.iloc[i]["WELL_NAME"]).split("-", 1)[0]
        x = gauge_df.iloc[i]["EASTING"]
        y = gauge_df.iloc[i]["NORTHING"]
        nm = f"{g}_GAUGE"
        stagedist[nm] = np.sqrt(
            ((stagedist["XCOORDS"] - x) ** 2) + ((stagedist["YCOORDS"] - y) ** 2)
        )

    # -------------------------------------------------------------------------
    # Select Closest Gauge
    # R:
    # STAGEDIST$STAGE <- NA
    # for(i in 1:nrow(STAGEDIST)){ # warning: hard coded col from 5 to 11,
    #   STAGEDIST$STAGE[i] <- names(which.min(apply(STAGEDIST[i,5:11,with=FALSE],MARGIN=2,min)))
    # }
    #
    # Exact direct port of hard-coded R slice 5:11 (1-based)
    # -> Python slice 4:11 (0-based, end-exclusive).
    # -------------------------------------------------------------------------
    stagedist["STAGE"] = pd.NA

    if stagedist.shape[1] < 11:
        raise ValueError(
            "STAGEDIST has fewer than 11 columns before closest-gauge selection; "
            "cannot directly port the hard-coded R slice 5:11."
        )

    stage_cols = list(stagedist.columns[4:11])

    for i in range(len(stagedist)):
        row_values = stagedist.iloc[i, 4:11].to_numpy(dtype=float)
        min_idx = int(np.argmin(row_values))
        stagedist.iat[i, stagedist.columns.get_loc("STAGE")] = stage_cols[min_idx]

    dist_path: Path | None = None
    stagedist_path: Path | None = None

    return dist, stagedist
