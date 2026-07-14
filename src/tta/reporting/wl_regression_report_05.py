import logging
import math

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from pathlib import Path
from typing import Tuple

from tqdm import tqdm

logger = logging.getLogger("tta")

FIGURE_SIZE = (8.5, 11)
GRID_ROWS, GRID_COLS = 12, 4
GRID_WSPACE, GRID_HSPACE = 0.5, 4

COLOR_LIGHT_GRAY = "#E5E5E5"
COLOR_BLACK = "#000000"
COLOR_GREEN = "#006400"
COLOR_LIGHT_GREEN = "#00640080"
COLOR_DARK_GRAY = "#B3B3B3"
COLOR_GRAY_EDGE = "#7F7F7F"
COLOR_RED = "#FF0000"
COLOR_LIGHT_BLUE = "#97C4EF"
COLOR_BISQUE = "bisque"
COLOR_WHITE = "#FFFFFF"

# GIS map bounds — site-specific; match generate_report_05.py
GIS_X_MIN, GIS_X_MAX = 561500, 586000
GIS_Y_MIN, GIS_Y_MAX = 145000, 155000

FONT_SIZE_AXIS_LABELS = 9
FONT_SIZE_ANNOTATION = 6


def plt_header(
    page_fig: Figure,
    well_name: str,
    p_interp: float,
    p_trend: float,
    distance_to_river: float,
) -> None:
    """Render well name and river-stage significance header."""
    page_fig.text(x=0.5, y=0.95, s=well_name, ha="center", va="top", fontsize=20)

    if np.isnan(p_interp):
        header_text = "No Correlation with River Stage"
    else:
        if p_trend < 0.05 and p_interp < 0.05 and distance_to_river < 1500:
            sig_text = "Correlation with River Stage significant"
        else:
            sig_text = "Correlation with River Stage not significant"
        header_text = (
            f"Significance of River Stage (p-value): {p_interp:.2g}\n"
            f"Distance to River: {round(distance_to_river)} m\n"
            f"{sig_text}"
        )

    page_fig.text(
        x=0.5,
        y=0.925,
        s=header_text,
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )


def plt_gis(
    gis_wells: gpd.GeoDataFrame,
    ou: str,
    gis_well: gpd.GeoDataFrame,
    page_fig: Figure,
    grid_spec: GridSpec,
    gis_highriv_df: gpd.GeoDataFrame,
    gis_roads_df: gpd.GeoDataFrame,
    gis_ous_df: gpd.GeoDataFrame,
) -> None:
    """Render site map showing well location within its operational unit."""
    if gis_highriv_df.crs is None:
        raise ValueError("River GeoDataFrame has no CRS.")

    gis_roads = gis_roads_df.to_crs(gis_highriv_df.crs)
    ifile_gis_ous = gis_ous_df.to_crs(gis_highriv_df.crs)
    gis_ou = ifile_gis_ous[ifile_gis_ous["Name"] == ou]

    gis_axis = page_fig.add_subplot(grid_spec[1:4, 0:3])
    gis_roads.plot(
        ax=gis_axis, color=COLOR_LIGHT_GRAY, linewidth=1, markersize=50, zorder=1
    )
    gis_highriv_df.plot(
        ax=gis_axis,
        color=COLOR_LIGHT_GRAY,
        edgecolor=COLOR_BLACK,
        linewidth=1,
        zorder=2,
    )
    gis_ou.plot(
        ax=gis_axis, color="none", edgecolor=COLOR_GREEN, linewidth=1.5, zorder=3
    )
    gis_wells.plot(
        ax=gis_axis,
        color=COLOR_DARK_GRAY,
        edgecolor=COLOR_GRAY_EDGE,
        markersize=4,
        linewidth=0.5,
        zorder=4,
    )
    gis_well.plot(
        ax=gis_axis,
        color=COLOR_RED,
        marker="o",
        edgecolor=COLOR_BLACK,
        markersize=12,
        linewidth=0.75,
        zorder=5,
    )
    gis_axis.set_xticks([])
    gis_axis.set_yticks([])
    gis_axis.set_xlim((GIS_X_MIN, GIS_X_MAX))
    gis_axis.set_ylim((GIS_Y_MIN, GIS_Y_MAX))


def plt_cross_cor_lag(
    page_fig: Figure, grid_spec: GridSpec, wl_trends_well: pd.DataFrame
) -> int:
    """Render cross-correlation lag plot."""
    cross_cor_lag_plot = page_fig.add_subplot(grid_spec[4:7, 0:2])
    cod: pd.Series = wl_trends_well["COD"]
    cod_val = cod.iloc[0]
    if not isinstance(cod_val, list) or len(cod_val) == 0:
        cross_cor_lag_plot.text(
            0.5, 0.5, "No cross-correlation data",
            ha="center", va="center", transform=cross_cor_lag_plot.transAxes,
            fontsize=FONT_SIZE_AXIS_LABELS, color=COLOR_DARK_GRAY,
        )
        cross_cor_lag_plot.set_facecolor(COLOR_LIGHT_GRAY)
        cross_cor_lag_plot.set_xlabel("Lag Time (days)")
        cross_cor_lag_plot.set_ylabel("Cross Correlation Coefficient")
        return 0
    cod_df = pd.DataFrame(cod_val)
    lags: pd.Series = cod_df["lag"]
    acfs: pd.Series = cod_df["acf"]
    max_acf_idx = np.argmax(acfs)
    lag_max_acf = lags[max_acf_idx]
    max_acf = acfs[max_acf_idx]
    cross_cor_lag_plot.plot(lags, acfs, color=COLOR_GREEN)
    cross_cor_lag_plot.plot(
        [lag_max_acf, lag_max_acf], [0, max_acf], color=COLOR_BLACK, linestyle="-"
    )
    cross_cor_lag_plot.plot(
        lag_max_acf, max_acf, color=COLOR_BLACK, marker="o", markersize=2
    )
    cross_cor_lag_plot.annotate(
        text=f"({lag_max_acf}, {round(max_acf, 2)})",
        xy=(lag_max_acf, max_acf),
        xytext=(min(80, lag_max_acf), min(0.95, max_acf + 0.1)),
        fontsize=FONT_SIZE_ANNOTATION,
        va="center",
    )
    cross_cor_lag_plot.axhline(0, color=COLOR_BLACK, linewidth=1, linestyle="--")
    cross_cor_lag_plot.fill_between(lags, 0, acfs, color=COLOR_GREEN, alpha=0.3)
    cross_cor_lag_plot.set_xticks(np.arange(0, 91, 10))
    cross_cor_lag_plot.set_xticks(np.arange(0, 91, 5), minor=True)
    cross_cor_lag_plot.set_ylim(-1, 1)
    cross_cor_lag_plot.set_yticks(np.arange(-1, 1.01, 0.2))
    cross_cor_lag_plot.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cross_cor_lag_plot.set_xlabel("Lag Time (days)")
    cross_cor_lag_plot.set_ylabel("Cross Correlation Coefficient")
    cross_cor_lag_plot.set_facecolor(COLOR_LIGHT_GRAY)
    cross_cor_lag_plot.grid(True, linewidth=0.5, color=COLOR_WHITE)
    cross_cor_lag_plot.set_axisbelow(True)

    lag_handle = Line2D([], [], linestyle="-", color=COLOR_BLACK, marker="o")
    cross_cor_lag_plot.legend(
        handles=[lag_handle],
        labels=["Optimized Lag Time (days)"],
        fontsize=FONT_SIZE_ANNOTATION,
    )

    return lag_max_acf


def plt_wl_rs(
    page_fig: Figure, grid_spec: GridSpec, wl_rs_well: pd.DataFrame, lag_time: int
) -> None:
    """Render water-level vs river-stage scatter with OLS trend line."""
    wl_rs_axis = page_fig.add_subplot(grid_spec[4:7, 2:4])
    wl_rs_well_clean = wl_rs_well[wl_rs_well["WLE"].notna()]
    wl_elevations_clean: pd.Series = wl_rs_well_clean["WLE"]
    river_stages_clean: pd.Series = wl_rs_well_clean["INTERP"]
    slope, intercept = np.polyfit(river_stages_clean, wl_elevations_clean, deg=1)
    wl_rs_trend = slope * river_stages_clean + intercept
    sort_indexes = np.argsort(river_stages_clean.to_numpy())
    river_stages_clean_sorted = river_stages_clean.to_numpy()[sort_indexes]
    wl_rs_trend_sorted = wl_rs_trend.to_numpy()[sort_indexes]
    wl_rs_axis.plot(
        river_stages_clean, 
        wl_elevations_clean, 
        marker="o",
        markersize=4,
        linewidth=1,
        linestyle="None",
        markerfacecolor=COLOR_LIGHT_GREEN,
        markeredgewidth=0.75,
        markeredgecolor=COLOR_GREEN,
        zorder=2,
    )
    wl_rs_axis.plot(
        river_stages_clean_sorted,
        wl_rs_trend_sorted,
        color=COLOR_BLACK,
        linestyle="--",
    )
    wl_rs_axis.set_facecolor(COLOR_LIGHT_GRAY)
    wl_rs_axis.grid(True, linewidth=0.5, color=COLOR_WHITE)
    wl_rs_axis.set_axisbelow(True)
    wl_rs_axis.set_xlabel("River Stage (m amsl)")
    wl_rs_axis.set_ylabel("Water-Level (m amsl)")
    wl_rs_axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    wl_rs_axis.text(
        0.02,
        0.97,
        f"Lag Time = {lag_time} days",
        fontsize=FONT_SIZE_ANNOTATION,
        transform=wl_rs_axis.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor=COLOR_WHITE,
            edgecolor=COLOR_BLACK,
            linewidth=0.5
        ),
    )


def plt_wl_rs_timeseries(
    page_fig: Figure,
    grid_spec: GridSpec,
    wl_rs_well: pd.DataFrame,
    lag_time: int,
) -> Tuple[Axes, Axes]:
    """Render dual-axis water-level and river-stage time series."""
    wl_elevations: pd.Series = pd.to_numeric(wl_rs_well["WLE"], errors="coerce")
    wl_river_stages: pd.Series = pd.to_numeric(wl_rs_well["INTERP"], errors="coerce")
    wl_trends_dates = pd.to_datetime(wl_rs_well["EVENT"], errors="coerce")
    wl_elevations_clean = wl_elevations[~np.isnan(wl_elevations)]
    wl_trends_dates_clean = wl_trends_dates[~np.isnan(wl_elevations)]

    wl_elevation_axis = page_fig.add_subplot(grid_spec[7:9, :])
    wl_elevation_axis.set_facecolor(COLOR_LIGHT_GRAY)
    wl_elevation_axis.grid(True, linewidth=0.5, color=COLOR_WHITE)
    wl_elevation_axis.set_axisbelow(True)
    wl_elevation_axis.xaxis.set_major_locator(YearLocator())
    wl_elevation_axis.xaxis.set_major_formatter(DateFormatter("%Y"))
    wl_elevation_axis.set_ylabel("Water-Level (m amsl)")
    wl_elevation_axis.tick_params(axis="x", labelrotation=90)
    wl_elevation_axis.tick_params(axis="y", labelsize=FONT_SIZE_AXIS_LABELS)

    wl_ymin = math.floor(wl_elevations_clean.min())
    wl_ymax = math.ceil(wl_elevations_clean.max())
    wl_ticks = np.linspace(wl_ymin, wl_ymax, 6)
    wl_elevation_axis.set_yticks(wl_ticks)

    min_stage = 2 * np.floor((wl_river_stages.min(skipna=True) - 1) / 2) + 1
    max_stage = 2 * np.ceil((wl_river_stages.max(skipna=True) - 1) / 2) + 1
    stage_ymin = min_stage - 0.25
    stage_ymax = max_stage + 0.25
    ticks = np.arange(min_stage, max_stage + 2, 2)

    # Right axis is display-only. River stage is scaled and drawn on the
    # water-level axis so it stays behind water-level observations.
    wl_river_stage_axis = wl_elevation_axis.twinx()
    wl_river_stage_axis.set_ylim(stage_ymin, stage_ymax)
    wl_river_stage_axis.set_yticks(ticks)
    wl_river_stage_axis.set_ylabel("River Stage (m amsl)")
    wl_river_stage_axis.patch.set_visible(False)
    wl_river_stage_axis.grid(False)
    wl_river_stage_axis.tick_params(axis="y", labelsize=FONT_SIZE_AXIS_LABELS)

    stage_frac = (wl_river_stages - stage_ymin) / (stage_ymax - stage_ymin)
    wl_river_stages_scaled = wl_ymin + stage_frac * (wl_ymax - wl_ymin)

    wl_elevation_axis.plot(
        wl_trends_dates_clean,
        wl_elevations_clean,
        marker="o",
        linewidth=1,
        linestyle="-",
        color=COLOR_GREEN,
        markersize=4,
        markerfacecolor=COLOR_LIGHT_GREEN,
        markeredgewidth=0.75,
        markeredgecolor=COLOR_GREEN,
        label="Water-Level",
        zorder=4,
    )

    wl_elevation_axis.plot(
        wl_trends_dates,
        wl_river_stages_scaled,
        linewidth=0.75,
        color=COLOR_LIGHT_BLUE,
        label="River Stage",
        zorder=1,
    )

    wl_river_stage_axis.text(
        0.0075,
        0.94,
        f"Lag Time = {lag_time} days",
        fontsize=FONT_SIZE_ANNOTATION,
        transform=wl_elevation_axis.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor=COLOR_WHITE,
            edgecolor=COLOR_BLACK,
            linewidth=0.5
        ),
    )

    return wl_elevation_axis, wl_river_stage_axis


def plt_std_res_pred_wl(
    page_fig: Figure, grid_spec: GridSpec, wl_trends_well: pd.DataFrame
) -> None:
    """Render standardized residuals vs predicted water-level diagnostic plot."""
    std_res_pred_wl_plot = page_fig.add_subplot(grid_spec[9:12, :])
    residuals: np.ndarray = np.array(wl_trends_well["RES"].iloc[0])
    pred_raw: np.ndarray = np.array(wl_trends_well["PRED"].iloc[0], dtype=float)

    if len(residuals) < 2:
        std_res_pred_wl_plot.text(
            0.5, 0.5, "Insufficient residual data",
            ha="center", va="center", transform=std_res_pred_wl_plot.transAxes,
            fontsize=FONT_SIZE_AXIS_LABELS, color=COLOR_DARK_GRAY,
        )
        std_res_pred_wl_plot.set_facecolor(COLOR_LIGHT_GRAY)
        std_res_pred_wl_plot.set_xlabel("Predicted Water-Level (m amsl)")
        std_res_pred_wl_plot.set_ylabel("Standardized Residuals")
        return

    log_mode: str = wl_trends_well["LOG"].iloc[0]
    if log_mode == "log":
        predicted_wl = np.exp(pred_raw)
    elif log_mode == "log10":
        predicted_wl = 10.0**pred_raw
    else:
        predicted_wl = pred_raw
    resid_std = np.std(residuals)
    std_residuals = residuals / resid_std if resid_std > 0 else residuals
    std_res_pred_wl_plot.plot(
        predicted_wl, 
        std_residuals, 
        marker="o",
        markersize=4,
        linewidth=1,
        linestyle="None",
        markerfacecolor=COLOR_LIGHT_GREEN,
        markeredgewidth=0.75,
        markeredgecolor=COLOR_GREEN,
    )
    std_res_pred_wl_plot.yaxis.set_major_locator(MultipleLocator(0.5))
    std_res_pred_wl_plot.set_facecolor(COLOR_LIGHT_GRAY)
    std_res_pred_wl_plot.grid(True, linewidth=0.5, color=COLOR_WHITE)
    std_res_pred_wl_plot.set_axisbelow(True)
    std_res_pred_wl_plot.set_xlabel("Predicted Water-Level (m amsl)")
    std_res_pred_wl_plot.set_ylabel("Standardized Residuals")
    std_res_ymin = math.floor(std_residuals.min())
    std_res_ymax = math.ceil(std_residuals.max())
    std_res_max = round(max(abs(std_res_ymin), abs(std_res_ymax)))
    std_res_ticks = np.arange(-std_res_max, std_res_max + 0.5, 0.5)
    std_res_pred_wl_plot.set_ylim(-std_res_max, std_res_max)
    std_res_pred_wl_plot.set_yticks(std_res_ticks)
    std_res_pred_wl_plot.margins(y=0.1)
    std_res_pred_wl_plot.axhline(-2, color=COLOR_BLACK, linestyle=(0, (5, 5)), linewidth=1.25, dash_capstyle="round")
    std_res_pred_wl_plot.axhline(2, color=COLOR_BLACK, linestyle=(0, (5, 5)), linewidth=1.25, dash_capstyle="round")
    std_res_pred_wl_plot.axhline(0, color=COLOR_BLACK, linestyle=(0, (2, 3.5)), linewidth=1, dash_capstyle="round")


def plt_legend(
    page_fig: Figure,
    grid_spec: GridSpec,
    wl_elevation_axis: Axes,
    wl_river_stage_axis: Axes,
) -> None:
    """Render combined legend for water-level and river-stage series."""
    legend_plot = page_fig.add_subplot(grid_spec[1:4, 3])
    wl_elev_ax_handles, wl_elev_ax_labels = (
        wl_elevation_axis.get_legend_handles_labels()
    )
    wl_rs_ax_handles, wl_rs_ax_labels = wl_river_stage_axis.get_legend_handles_labels()
    handles = wl_elev_ax_handles + wl_rs_ax_handles
    labels = wl_elev_ax_labels + wl_rs_ax_labels
    legend_plot.legend(handles, labels, loc="center")
    legend_plot.axis("off")


def wl_regression_report(
    run_id: str,
    output_dir: Path,
    dist: pd.DataFrame,
    wl_rs: pd.DataFrame,
    wl_trends: pd.DataFrame,
    wells: pd.DataFrame,
    gis_river_shapefile: Path,
    gis_roads_shapefile: Path,
    gis_ou_shapefile: Path,
    map_crs: str,
) -> None:
    """
    Generate a PDF of water-level regression diagnostic plots, one page per well.

    Parameters
    ----------
    run_id : str
        Run identifier appended to the output filename.
    output_dir : Path
        Directory where the PDF is written.
    dist : pd.DataFrame
        Distance table with NAME and DIST columns.
    wl_rs : pd.DataFrame
        Water-level / river-stage combined table from step 02.
    wl_trends : pd.DataFrame
        Flattened water-level trend results from step 03.
    wells : pd.DataFrame
        Well information table with NAME, XCOORDS, YCOORDS, OU columns.
    gis_river_shapefile : Path
        Path to river polyline shapefile.
    gis_roads_shapefile : Path
        Path to roads shapefile.
    gis_ou_shapefile : Path
        Path to operational-unit polygon shapefile.
    map_crs : str
        CRS string (e.g. 'EPSG:2926') for the well point layer.
    """
    gis_wells = gpd.GeoDataFrame(
        wells,
        geometry=gpd.points_from_xy(wells["XCOORDS"], wells["YCOORDS"]),
        crs=map_crs,
    )

    gis_highriv_df = gpd.read_file(gis_river_shapefile)
    gis_roads_df = gpd.read_file(gis_roads_shapefile)
    gis_ous_df = gpd.read_file(gis_ou_shapefile)

    out_path = output_dir / f"OLSRegression_Plots_Water_Levels_Python_{run_id}.pdf"

    plt.rcParams["font.family"] = "Arial"

    n_wells = len(wells)
    with PdfPages(out_path) as pdf:
        for i in tqdm(range(n_wells), desc="WL regression report", unit="well", leave=True):
            well: pd.Series = wells.iloc[i]
            well_name = well["NAME"]
            logger.debug(
                "WL regression report: well %d/%d — %s", i + 1, n_wells, well_name
            )

            wl_rs_well = wl_rs.loc[wl_rs["NAME"] == well_name]
            wl_trends_well = wl_trends.loc[wl_trends["NAME"] == well_name]

            if wl_rs_well.empty or wl_trends_well.empty:
                logger.warning(
                    "WL regression report: skipping %s — no data in wl_rs or wl_trends",
                    well_name,
                )
                continue

            page_fig = plt.figure(figsize=FIGURE_SIZE)
            grid_spec = GridSpec(
                GRID_ROWS,
                GRID_COLS,
                wspace=GRID_WSPACE,
                hspace=GRID_HSPACE,
                figure=page_fig,
            )

            p_interp: float = wl_trends_well["p_interp"].iloc[0]
            p_trend: float = wl_trends_well["p_trend"].iloc[0]
            distance_to_river: float = dist.loc[dist["NAME"] == well_name, "DIST"].iloc[
                0
            ]
            ou: str = wl_rs_well["OU"].iloc[0]
            gis_well = gis_wells[gis_wells["NAME"] == well_name]

            plt_header(page_fig, well_name, p_interp, p_trend, distance_to_river)
            plt_gis(
                gis_wells=gis_wells,
                ou=ou,
                gis_well=gis_well,
                page_fig=page_fig,
                grid_spec=grid_spec,
                gis_highriv_df=gis_highriv_df,
                gis_roads_df=gis_roads_df,
                gis_ous_df=gis_ous_df,
            )
            lag_time = plt_cross_cor_lag(page_fig, grid_spec, wl_trends_well)
            plt_wl_rs(page_fig, grid_spec, wl_rs_well, lag_time)
            wl_elevation_axis, wl_river_stage_axis = plt_wl_rs_timeseries(
                page_fig, grid_spec, wl_rs_well, lag_time
            )
            plt_std_res_pred_wl(page_fig, grid_spec, wl_trends_well)
            plt_legend(page_fig, grid_spec, wl_elevation_axis, wl_river_stage_axis)

            pdf.savefig(page_fig)
            plt.close(page_fig)

    logger.info("WL regression report written to %s", out_path)
