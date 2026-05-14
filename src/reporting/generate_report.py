import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from typing import Tuple
from pathlib import Path
from tqdm import tqdm

# Figure constants
FIGURE_SIZE = (8.5, 11)
GRID_ROWS, GRID_COLS = 6, 4
GRID_WSPACE, GRID_HSPACE = 0.1, 0.15
FIGURE_LEFT_MARGIN = 1.25 / 8.5
FIGURE_RIGHT_MARGIN = 1 - 1.5 / 11

# GIS map bounds
GIS_X_MIN, GIS_X_MAX = 561500, 586000
GIS_Y_MIN, GIS_Y_MAX = 145000, 155000

# Font sizes
FONT_SIZE_TITLE = 20
FONT_SIZE_HEADER = 10
FONT_SIZE_TEXT = 8
FONT_SIZE_TABLE = 10

# Colors
COLOR_LIGHT_GRAY = "#E5E5E5"
COLOR_DARK_GRAY = "#B3B3B3"
COLOR_GRAY_EDGE = "#7F7F7F"
COLOR_GREEN = "#006400"
COLOR_RED = "#FF0000"
COLOR_BLACK = "#000000"
COLOR_LIGHT_BLUE = "#97C4EF"
COLOR_BISQUE = "bisque"
COLOR_ND = "#D19999"
COLOR_ND_EDGE = "#B12224"
COLOR_CALC = "#E9B831"

# Marker colors
MARKER_FACE_COLORS = ["#ADD2BD", "#C9C5E0", "#B9B9B9", "#FFDC9B", "#9B9BEB"]
MARKER_EDGE_COLORS = ["#2E8B57", "#756BB1", "#4D4D4D", "#FFA500", "#0000CD"]


def _event_numeric_rstyle(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce").dt.floor("D")
    return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)).to_numpy(
        dtype=float
    )


def calculate_tobit_prediction(
    chem_term: pd.DataFrame,
    model_row: pd.Series,
    log_mode: str = "log",
) -> pd.Series:
    event_num = _event_numeric_rstyle(chem_term["EVENT"])

    linear = model_row["beta_intercept"] + model_row["beta_event"] * event_num

    if pd.notna(model_row.get("beta_interp", np.nan)):
        linear = linear + model_row["beta_interp"] * pd.to_numeric(
            chem_term["INTERP"], errors="coerce"
        ).to_numpy(dtype=float)

    if log_mode == "log":
        pred = np.exp(linear)
    elif log_mode == "log10":
        pred = 10**linear
    else:
        pred = linear

    return pd.Series(pred, index=chem_term.index)


def plt_report(
    OUs: list[str],
    wells: pd.DataFrame,
    ifile_wl_trends: pd.DataFrame,
    ifile_chem_trends: pd.DataFrame,
    gis_wells: pd.DataFrame,
    ifile_wl_rs: pd.DataFrame,
    ifile_chem_rs: pd.DataFrame,
    ifile_no_rs: pd.DataFrame,
    wl_wells_set: set[str],
    river_shapefile: Path,
    roads_shapefile: Path,
    ou_shapefile: Path,
    output_dir: Path,
    run_ver: str,
):
    for ou in OUs:
        wells_ou = wells[wells["OU"] == ou]
        output_file = output_dir / f"TobitRegression_WLlag_{ou}_CY2024_{run_ver}.pdf"

        with PdfPages(output_file) as pdf:
            well: str
            for well in tqdm(
                wells_ou["NAME"],
                desc=f"Reporting {ou}",
                unit="well",
                leave=True,
            ):
                wl_trends_well = ifile_wl_trends[ifile_wl_trends["KEY"] == well]
                chem_trends_well = ifile_chem_trends.loc[
                    ifile_chem_trends["WELL"] == well, "ITER"
                ]
                gis_well = gis_wells[gis_wells["NAME"] == well]
                wl_rs_well = ifile_wl_rs[ifile_wl_rs["NAME"] == well]
                chem_rs_well = ifile_chem_rs[ifile_chem_rs["NAME"] == well]

                page_fig = plt.figure(figsize=FIGURE_SIZE)
                grid_spec = GridSpec(
                    nrows=GRID_ROWS,
                    ncols=GRID_COLS,
                    wspace=GRID_WSPACE,
                    hspace=GRID_HSPACE,
                )

                plt_header(
                    ifile_no_rs,
                    ifile_wl_trends,
                    ifile_chem_trends,
                    well,
                    wl_trends_well,
                    chem_trends_well,
                    gis_well,
                    page_fig,
                    grid_spec,
                )

                plt_gis(
                    gis_wells,
                    ou,
                    gis_well,
                    page_fig,
                    grid_spec,
                    river_shapefile,
                    roads_shapefile,
                    ou_shapefile,
                )

                num_trend_equations = plt_regression(
                    chem_trends_well, ifile_chem_trends, well, page_fig
                )

                (
                    chem_dates,
                    chem_concentrations_axis,
                    chem_river_stage_axis,
                    chem_concentrations_axis2,
                ) = plt_chem(
                    num_trend_equations,
                    wl_wells_set,
                    well,
                    chem_trends_well,
                    chem_rs_well,
                    ifile_chem_trends,
                    page_fig,
                    grid_spec,
                )

                wl_elevation_axis = None
                wl_river_stage_axis = None
                if well in wl_wells_set:
                    wl_elevation_axis, wl_river_stage_axis = plt_wl_rs(
                        wl_rs_well,
                        page_fig,
                        grid_spec,
                        chem_dates,
                        chem_river_stage_axis,
                    )

                plt_legend(
                    page_fig,
                    grid_spec,
                    chem_concentrations_axis,
                    chem_concentrations_axis2,
                    wl_elevation_axis,
                    wl_river_stage_axis,
                )

                page_fig.subplots_adjust(
                    left=FIGURE_LEFT_MARGIN, right=FIGURE_RIGHT_MARGIN
                )
                pdf.savefig(page_fig)
                plt.close(page_fig)


def plt_header(
    ifile_no_rs: pd.DataFrame,
    ifile_wl_trends: pd.DataFrame,
    ifile_chem_trends: pd.DataFrame,
    well: str,
    wl_trends_well: pd.DataFrame,
    chem_trends_well: pd.DataFrame,
    gis_well: pd.DataFrame,
    page_fig: Figure,
    grid_spec: GridSpec,
):
    page_fig.text(
        0.5,
        0.95,
        f"{well}",
        ha="center",
        va="top",
        fontsize=FONT_SIZE_TITLE,
        fontstyle="italic",
        fontweight="light",
    )
    page_fig.text(
        0.5,
        0.925,
        f"""
Distance to River: {round(gis_well.at[gis_well.index[0], "DIST"])} m
Number of Trends Calculated: {len(chem_trends_well)}
""",
        ha="center",
        va="top",
        fontsize=FONT_SIZE_HEADER,
        fontstyle="italic",
    )

    trends_axis = page_fig.add_subplot(grid_spec[0, 2])
    trends_axis.axis("off")
    trends_rows = [
        "Est. Lag Time (days):",
        "Significance of Trend (p-value):",
        "Significance of Date (p-value):",
        "Number of Observations:",
        "Percent NDs:",
    ]
    trends_columns: list[str] = []
    trends_values: list[str] = []

    no_river_stage = well in ifile_no_rs["NAME"].values
    if not no_river_stage:
        trends_rows.insert(2, "Significance of River Stage (p-value):")

    if len(wl_trends_well) == 1:
        trends_columns.append("WL")
    if len(chem_trends_well) == 1:
        trends_columns.append("Conc")
    else:
        trends_columns += [f"Trend{n}" for n in range(1, len(chem_trends_well) + 1)]

    for _, row_label in enumerate(trends_rows):
        row_values: list[str] = []

        # WL column
        if len(wl_trends_well) == 1:
            wl_lag = ifile_wl_trends.loc[ifile_wl_trends["KEY"] == well, "LAG"].item()
            wl_p_trend = ifile_wl_trends.loc[
                ifile_wl_trends["KEY"] == well, "p_trend"
            ].item()
            wl_p_interp = (
                ifile_wl_trends.get(
                    "p_interp", pd.Series(0, index=ifile_wl_trends.index)
                )
                .loc[ifile_wl_trends["KEY"] == well]
                .item()
            )
            wl_p_event = (
                ifile_wl_trends.get(
                    "p_event", pd.Series(0, index=ifile_wl_trends.index)
                )
                .loc[ifile_wl_trends["KEY"] == well]
                .item()
            )
            wl_n_obs = (
                ifile_wl_trends.get("n_obs", pd.Series(0, index=ifile_wl_trends.index))
                .loc[ifile_wl_trends["KEY"] == well]
                .item()
            )

            if row_label == "Est. Lag Time (days):":
                row_values.append(f"{round(wl_lag)}" if ~np.isnan(wl_lag) else "NA")
            elif row_label == "Significance of Trend (p-value):":
                row_values.append(
                    f"{wl_p_trend:.2g}" if ~np.isnan(wl_p_trend) else "NA"
                )
            elif row_label == "Significance of River Stage (p-value):":
                row_values.append(
                    f"{wl_p_interp:.2g}" if ~np.isnan(wl_p_interp) else "NA"
                )
            elif row_label == "Significance of Date (p-value):":
                row_values.append(
                    f"{wl_p_event:.2g}" if ~np.isnan(wl_p_event) else "NA"
                )
            elif row_label == "Number of Observations:":
                row_values.append(f"{round(wl_n_obs)}" if ~np.isnan(wl_n_obs) else "NA")
            else:
                row_values.append("")

        # Chem columns
        for trend_idx in range(len(chem_trends_well)):
            chem_trend_row = ifile_chem_trends[ifile_chem_trends["WELL"] == well].iloc[
                trend_idx
            ]
            chem_lag = "NA"

            if no_river_stage and trend_idx == 0:
                chem_lag = "No Covariate"
            elif no_river_stage and trend_idx != 0:
                chem_lag = "No RS"
            elif ~np.isnan(chem_trend_row["LAG"]):
                chem_lag = f"{round(chem_trend_row['LAG'])}"

            if row_label == "Est. Lag Time (days):":
                row_values.append(chem_lag)
            elif row_label == "Significance of Trend (p-value):":
                row_values.append(
                    f"{chem_trend_row['p_trend']:.2g}"
                    if ~np.isnan(chem_trend_row["p_trend"])
                    else "NA"
                )
            elif row_label == "Significance of River Stage (p-value):":
                row_values.append(
                    f"{chem_trend_row['p_interp']:.2g}"
                    if ~np.isnan(chem_trend_row["p_interp"])
                    else "NA"
                )
            elif row_label == "Significance of Date (p-value):":
                row_values.append(
                    f"{chem_trend_row['p_event']:.2g}"
                    if ~np.isnan(chem_trend_row["p_event"])
                    else "NA"
                )
            elif row_label == "Number of Observations:":
                row_values.append(
                    f"{round(chem_trend_row['n_obs'])}"
                    if ~np.isnan(chem_trend_row["n_obs"])
                    else "NA"
                )
            elif row_label == "Percent NDs:":
                row_values.append(
                    f"  {round(chem_trend_row['n_cens'] / chem_trend_row['n_obs'] * 100)}%"
                    if ~np.isnan(chem_trend_row["n_cens"])
                    else "NA"
                )

        trends_values.append(row_values)

    trends_table = trends_axis.table(
        cellText=trends_values,
        rowLabels=trends_rows,
        colLabels=trends_columns,
        loc="center",
        rowLoc="right",
        cellLoc="center",
        edges="open",
    )
    trends_table.auto_set_column_width(col=list(range(len(trends_columns))))
    trends_table.auto_set_font_size(False)
    trends_table.set_fontsize(FONT_SIZE_TABLE)

    for (row, col), cell in trends_table.get_celld().items():
        if col == -1:
            cell.set_text_props(fontproperties=FontProperties(style="italic"))
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))


def plt_gis(
    gis_wells: gpd.GeoDataFrame,
    ou: str,
    gis_well: gpd.GeoDataFrame,
    page_fig: Figure,
    grid_spec: GridSpec,
    river_shapefile: Path,
    roads_shapefile: Path,
    ou_shapefile: Path,
):
    ifile_gis_highriv = gpd.read_file(river_shapefile)
    ifile_gis_roads = gpd.read_file(roads_shapefile)
    ifile_gis_ous = gpd.read_file(ou_shapefile)
    gis_roads = ifile_gis_roads.to_crs(ifile_gis_highriv.crs)
    ifile_gis_ous = ifile_gis_ous.to_crs(ifile_gis_highriv.crs)
    gis_ou = ifile_gis_ous[ifile_gis_ous["Name"] == ou]
    gis_ou = gis_ou.to_crs(ifile_gis_highriv.crs)

    gis_axis = page_fig.add_subplot(grid_spec[1, 0:2])
    gis_roads.plot(
        ax=gis_axis, color=COLOR_LIGHT_GRAY, linewidth=1, markersize=50, zorder=1
    )
    ifile_gis_highriv.plot(
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
    gis_axis.set_xlim([GIS_X_MIN, GIS_X_MAX])
    gis_axis.set_ylim([GIS_Y_MIN, GIS_Y_MAX])


def plt_regression(
    chem_trends_well: pd.DataFrame,
    ifile_chem_trends: pd.DataFrame,
    well: str,
    page_fig: Figure,
) -> int:
    model_equation_text: list[str] = []
    num_trend_equations = 0

    for trend_idx in range(len(chem_trends_well)):
        row = ifile_chem_trends.loc[
            (ifile_chem_trends["WELL"] == well)
            & (ifile_chem_trends["ITER"] == trend_idx + 1)
        ].iloc[0]

        beta_interp: float = row["beta_interp"]
        beta_event: float = row["beta_event"]
        beta_intercept: float = row["beta_intercept"]
        se_interp: float = row["se_interp"]
        se_event: float = row["se_event"]
        se_intercept: float = row["se_intercept"]

        if row["p_trend"] > 0.05:
            model_equation_text.append(
                f"Trend{trend_idx + 1}:\nTrend Not Significant\n"
            )
            num_trend_equations += 1
        elif (
            ~np.isnan(beta_interp)
            and ~np.isnan(beta_event)
            and ~np.isnan(beta_intercept)
        ):
            model_equation_text.append(f"""Trend{trend_idx + 1}:
In Conc. = {beta_interp:.2f} (+/- {se_interp:.3f})*River Stage + {beta_event:.5f} (+/- {se_event:.6f})*Date + {beta_intercept:.0f} (+/- {se_intercept:.1f})
""")
            num_trend_equations += 1
        else:
            model_equation_text.append(
                f"Trend{trend_idx + 1}:\nNo Trend Analysis Performed"
            )

    if num_trend_equations == 0:
        page_fig.text(
            x=0.5,
            y=0.3,
            s="No Regression Analysis Performed",
            fontsize=FONT_SIZE_TEXT,
            ha="center",
            va="center",
        )
    else:
        page_fig.text(
            x=0.5,
            y=0.17,
            s="Censored Regression (Tobit) Model",
            fontsize=FONT_SIZE_TEXT,
            ha="center",
            va="center",
        )
        for i in range(len(model_equation_text)):
            page_fig.text(
                x=0.5,
                y=0.15 - 0.025 * i,
                s=model_equation_text[i],
                fontsize=FONT_SIZE_TEXT,
                ha="center",
                va="top",
            )

    return num_trend_equations


def plt_chem(
    num_trend_equations: int,
    wl_wells_set: set[str],
    well: str,
    chem_trends_well: pd.Series,
    chem_rs_well: pd.DataFrame,
    ifile_chem_trends: pd.DataFrame,
    page_fig: Figure,
    grid_spec: GridSpec,
) -> Tuple[pd.DatetimeIndex, Axes, Axes, Axes]:

    # R equivalent: DT <- X@DATA; DT <- DT[order(EVENT), ]
    chem_rs_well = chem_rs_well.copy()
    chem_rs_well["EVENT"] = pd.to_datetime(chem_rs_well["EVENT"], errors="coerce")
    chem_rs_well["TERM_NUM"] = pd.to_numeric(chem_rs_well["TERM"], errors="coerce")
    chem_rs_well["VAL_NUM"] = pd.to_numeric(chem_rs_well["VAL"], errors="coerce")
    chem_rs_well["INTERP_NUM"] = pd.to_numeric(chem_rs_well["INTERP"], errors="coerce")
    chem_rs_well = chem_rs_well.sort_values("EVENT").reset_index(drop=True)

    chem_obs = chem_rs_well.loc[
        chem_rs_well["VAL_NUM"].notna() & chem_rs_well["TERM_NUM"].notna()
    ].copy()

    chem_nd = chem_obs.loc[chem_obs["NDS"].fillna(False).astype(bool)].copy()

    chem_dates = chem_rs_well["EVENT"]
    chem_river_stages = chem_rs_well["INTERP_NUM"]

    if chem_obs.empty:
        chem_min, chem_max = 1.0, 10.0
    else:
        chem_min = chem_obs["VAL_NUM"].min(skipna=True)
        chem_max = chem_obs["VAL_NUM"].max(skipna=True)
        if not np.isfinite(chem_min) or chem_min <= 0:
            chem_min = chem_obs.loc[chem_obs["VAL_NUM"] > 0, "VAL_NUM"].min(skipna=True)
        if not np.isfinite(chem_min) or chem_min <= 0:
            chem_min = 1.0
        if not np.isfinite(chem_max) or chem_max <= chem_min:
            chem_max = chem_min * 10.0

    chem_concentrations_axis = page_fig.add_subplot(
        grid_spec[3, :] if well in wl_wells_set else grid_spec[2, :]
    )
    chem_concentrations_axis.set_facecolor(COLOR_LIGHT_GRAY)
    chem_concentrations_axis.xaxis.set_major_locator(YearLocator())
    chem_concentrations_axis.xaxis.set_major_formatter(DateFormatter("%Y"))
    chem_concentrations_axis.set_yscale("log")
    chem_concentrations_axis.yaxis.set_major_locator(LogLocator(base=10))
    chem_concentrations_axis.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    chem_concentrations_axis.yaxis.set_minor_formatter(NullFormatter())
    ymin = 10 ** np.floor(np.log10(chem_min))
    ymax = 10 ** np.ceil(np.log10(chem_max))
    chem_concentrations_axis.set_ylim(ymin - 0.1, ymax + 1)
    chem_concentrations_axis.set_ylabel("Hex. & Filt. Cr (µg/L)")
    chem_concentrations_axis.grid(
        True, which="major", axis="x", linewidth=0.5, color="#FFFFFF"
    )
    chem_concentrations_axis.grid(
        True, which="major", axis="y", linewidth=0.5, color=COLOR_BLACK
    )
    chem_concentrations_axis.grid(
        True, which="minor", axis="y", linewidth=0.5, color="#FFFFFF"
    )
    chem_concentrations_axis.set_axisbelow(True)

    # Right axis is display-only. River stage is plotted on the chemistry axis
    # after scaling, which avoids twinx layering/grid artefacts.
    chem_river_stage_axis = chem_concentrations_axis.twinx()
    chem_river_stage_axis.set_ylabel("River Stage (m amsl)")
    min_stage = 2 * np.floor((chem_river_stages.min(skipna=True) - 1) / 2) + 1
    max_stage = 2 * np.ceil((chem_river_stages.max(skipna=True) - 1) / 2) + 1
    stage_ymin = min_stage - 0.25
    stage_ymax = max_stage + 0.25
    ticks = np.arange(min_stage, max_stage + 2, 2)
    chem_river_stage_axis.set_yticks(ticks)
    chem_river_stage_axis.set_ylim(stage_ymin, stage_ymax)
    chem_river_stage_axis.patch.set_visible(False)
    chem_river_stage_axis.grid(False)

    # Log-space scaling because the concentration axis is log-scaled.
    chem_ymin, chem_ymax = chem_concentrations_axis.get_ylim()
    stage_frac = (chem_river_stages - stage_ymin) / (stage_ymax - stage_ymin)
    chem_river_stages_scaled = np.exp(
        np.log(chem_ymin) + stage_frac * (np.log(chem_ymax) - np.log(chem_ymin))
    )
    chem_concentrations_axis.plot(
        chem_dates,
        chem_river_stages_scaled,
        linewidth=0.75,
        color=COLOR_LIGHT_BLUE,
        label="River Stage",
        zorder=1,
    )

    chem_concentrations_axis2 = page_fig.add_subplot(
        grid_spec[4, :] if well in wl_wells_set else grid_spec[3, :]
    )
    chem_concentrations_axis2.set_facecolor(COLOR_LIGHT_GRAY)
    chem_concentrations_axis2.xaxis.set_major_locator(YearLocator())
    chem_concentrations_axis2.xaxis.set_major_formatter(DateFormatter("%Y"))
    chem_concentrations_axis2.set_yscale("log")
    chem_concentrations_axis2.yaxis.set_major_locator(LogLocator(base=10))
    chem_concentrations_axis2.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    chem_concentrations_axis2.yaxis.set_minor_formatter(NullFormatter())
    chem_concentrations_axis2.set_ylim(ymin - 0.1, ymax + 1)
    chem_concentrations_axis2.set_ylabel("Hex. & Filt. Cr (µg/L)")
    chem_concentrations_axis2.grid(
        True, which="major", axis="x", linewidth=0.5, color="#FFFFFF"
    )
    chem_concentrations_axis2.grid(
        True, which="major", axis="y", linewidth=0.5, color=COLOR_BLACK
    )
    chem_concentrations_axis2.grid(
        True, which="minor", axis="y", linewidth=0.5, color="#FFFFFF"
    )
    chem_concentrations_axis2.set_axisbelow(True)

    if num_trend_equations == 0:
        chem_concentrations_axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    else:
        chem_concentrations_axis.set_xticklabels([])
        chem_concentrations_axis2.tick_params(axis="x", labelrotation=90, labelsize=7)

    marker_face_colors = MARKER_FACE_COLORS
    marker_edge_colors = MARKER_EDGE_COLORS

    for trend_idx in range(len(chem_trends_well)):
        term_no = trend_idx + 1

        obs_term = chem_obs.loc[chem_obs["TERM_NUM"] == term_no].copy()
        obs_term = obs_term.sort_values("EVENT").reset_index(drop=True)

        detects_term = obs_term.loc[~obs_term["NDS"].fillna(False).astype(bool)].copy()

        chem_concentrations_axis.plot(
            obs_term["EVENT"],
            obs_term["VAL_NUM"],
            linewidth=1,
            color=marker_edge_colors[trend_idx],
            zorder=3,
        )

        if not detects_term.empty:
            chem_concentrations_axis.plot(
                detects_term["EVENT"],
                detects_term["VAL_NUM"],
                linewidth=0,
                marker="o",
                markersize=4,
                markerfacecolor=marker_face_colors[trend_idx],
                markeredgewidth=0.75,
                markeredgecolor=marker_edge_colors[trend_idx],
                label=(
                    "Observed Concentration"
                    if len(chem_trends_well) == 1
                    else f"Observed Conc. (Trend{term_no})"
                ),
                zorder=4,
            )

        if num_trend_equations > 0:
            chem_concentrations_axis2.plot(
                obs_term["EVENT"],
                obs_term["VAL_NUM"],
                linewidth=1,
                color=marker_edge_colors[trend_idx],
                zorder=3,
            )

            if not detects_term.empty:
                chem_concentrations_axis2.plot(
                    detects_term["EVENT"],
                    detects_term["VAL_NUM"],
                    linewidth=0,
                    marker="o",
                    markersize=4,
                    markerfacecolor=marker_face_colors[trend_idx],
                    markeredgewidth=0.75,
                    markeredgecolor=marker_edge_colors[trend_idx],
                    zorder=4,
                )

            model_rows = ifile_chem_trends.loc[
                (ifile_chem_trends["WELL"] == well)
                & (ifile_chem_trends["ITER"] == term_no)
            ].copy()

            if "fit_ok" in model_rows.columns:
                model_rows = model_rows.loc[model_rows["fit_ok"] == True]

            if not model_rows.empty:
                model_row = model_rows.iloc[0]

                if pd.notna(model_row["beta_intercept"]):
                    pred_term = chem_rs_well.loc[
                        chem_rs_well["TERM_NUM"] == term_no
                    ].copy()

                    pred_term = (
                        pred_term.loc[
                            pred_term["EVENT"].notna() & pred_term["INTERP_NUM"].notna()
                        ]
                        .sort_values("EVENT")
                        .reset_index(drop=True)
                    )

                    if not pred_term.empty:
                        pred = calculate_tobit_prediction(pred_term, model_row)

                        ok = pred.notna() & np.isfinite(pred) & (pred > 0)

                        chem_concentrations_axis2.plot(
                            pred_term.loc[ok, "EVENT"],
                            pred.loc[ok],
                            linestyle="-",
                            color=COLOR_CALC,
                            linewidth=1,
                            label="Calculated Conc." if trend_idx == 0 else None,
                            zorder=1,
                        )

    if not chem_nd.empty:
        chem_concentrations_axis.plot(
            chem_nd["EVENT"],
            chem_nd["VAL_NUM"],
            marker="v",
            linewidth=0,
            markersize=4,
            markerfacecolor=COLOR_ND,
            markeredgewidth=0.75,
            markeredgecolor=COLOR_ND_EDGE,
            label="Non-Detect",
            zorder=4,
        )

        if num_trend_equations > 0:
            chem_concentrations_axis2.plot(
                chem_nd["EVENT"],
                chem_nd["VAL_NUM"],
                marker="v",
                linewidth=0,
                markersize=4,
                markerfacecolor=COLOR_ND,
                markeredgewidth=0.75,
                markeredgecolor=COLOR_ND_EDGE,
                zorder=4,
            )

    if num_trend_equations == 0:
        chem_concentrations_axis2.set_axis_off()

    return (
        chem_dates,
        chem_concentrations_axis,
        chem_river_stage_axis,
        chem_concentrations_axis2,
    )


def plt_wl_rs(
    wl_rs_well: pd.DataFrame,
    page_fig: Figure,
    grid_spec: GridSpec,
    chem_dates: pd.DatetimeIndex,
    chem_river_stage_axis: Axes,
) -> Tuple[Axes, Axes]:
    wl_elevations: pd.Series[float] = pd.to_numeric(wl_rs_well["WLE"], errors="coerce")
    wl_river_stages: pd.Series[float] = pd.to_numeric(
        wl_rs_well["INTERP"], errors="coerce"
    )
    wl_trends_dates = pd.to_datetime(wl_rs_well["EVENT"], errors="coerce")
    wl_elevations_clean = wl_elevations[~np.isnan(wl_elevations)]
    wl_trends_dates_clean = wl_trends_dates[~np.isnan(wl_elevations)]

    wl_elevation_axis = page_fig.add_subplot(grid_spec[2, :])
    wl_elevation_axis.set_facecolor(COLOR_LIGHT_GRAY)
    wl_elevation_axis.grid(True, linewidth=0.5, color="#FFFFFF")
    wl_elevation_axis.set_axisbelow(True)
    wl_elevation_axis.xaxis.set_major_locator(YearLocator())
    wl_elevation_axis.xaxis.set_major_formatter(DateFormatter("%Y"))
    wl_elevation_axis.set_xticklabels([])
    wl_elevation_axis.set_ylabel("Water-Level (m amsl)")

    wl_trends_dates_filtered = wl_trends_dates[wl_trends_dates >= chem_dates.min()]
    wl_river_stages_filtered = wl_river_stages[-1 * wl_trends_dates_filtered.count() :]

    min_stage = 2 * np.floor((wl_river_stages.min(skipna=True) - 1) / 2) + 1
    max_stage = 2 * np.ceil((wl_river_stages.max(skipna=True) - 1) / 2) + 1
    stage_ymin = min_stage - 0.25
    stage_ymax = max_stage + 0.25
    ticks = np.arange(min_stage, max_stage + 2, 2)

    wl_ymin = np.nanmin(
        [
            wl_rs_well["BOT"].iloc[0],
            wl_rs_well["TOP"].iloc[0],
            wl_elevations_clean.min(),
        ]
    )
    wl_ymax = np.nanmax(
        [
            wl_rs_well["BOT"].iloc[0],
            wl_rs_well["TOP"].iloc[0],
            wl_elevations_clean.max(),
        ]
    )
    wl_elevation_axis.set_ylim(wl_ymin, wl_ymax)

    # Right axis is display-only. River stage is scaled and drawn on the
    # water-level axis so it stays behind water-level observations.
    wl_river_stage_axis = wl_elevation_axis.twinx()
    wl_river_stage_axis.set_ylim(stage_ymin, stage_ymax)
    wl_river_stage_axis.set_yticks(ticks)
    wl_river_stage_axis.set_ylabel("River Stage (m amsl)")
    wl_river_stage_axis.patch.set_visible(False)
    wl_river_stage_axis.grid(False)

    stage_frac = (wl_river_stages_filtered - stage_ymin) / (stage_ymax - stage_ymin)
    wl_river_stages_scaled = wl_ymin + stage_frac * (wl_ymax - wl_ymin)
    wl_elevation_axis.plot(
        wl_trends_dates_filtered,
        wl_river_stages_scaled,
        linewidth=0.75,
        color=COLOR_LIGHT_BLUE,
        label="River Stage",
        zorder=1,
    )

    page_size = page_fig.get_size_inches() * 2.54  # cm
    screen_xrange = wl_trends_dates.max().year - wl_trends_dates.min().year
    screen_xgrid = screen_xrange / (0.96 * page_size[0])
    screen_xmax = pd.Timestamp(year=chem_dates.min().year - 1, month=10, day=1)
    screen_xmin = screen_xmax - pd.Timedelta(
        days=0.02 * page_size[0] * screen_xgrid * 365.25
    )
    epoch = pd.Timestamp(year=1970, month=1, day=1)
    screen_xmax = (screen_xmax - epoch) / pd.Timedelta(days=1)
    screen_xmin = (screen_xmin - epoch) / pd.Timedelta(days=1)
    screen_ymin = wl_rs_well["BOT"].iloc[0]
    screen_ymax = wl_rs_well["TOP"].iloc[0]

    screened_interval = patches.Rectangle(
        xy=(screen_xmin, screen_ymin),
        width=screen_xmax - screen_xmin,
        height=screen_ymax - screen_ymin,
        facecolor=COLOR_BISQUE,
        edgecolor=COLOR_BLACK,
        linewidth=0.5,
        zorder=2,
    )

    if (
        screen_ymin <= wl_elevations_clean.max()
        and screen_ymax >= wl_elevations_clean.min()
    ):
        wl_elevation_axis.add_patch(screened_interval)

        n_lines = 5
        screen_line_ypositions = np.linspace(screen_ymin, screen_ymax, n_lines + 2)[
            1:-1
        ]

        for y in screen_line_ypositions:
            wl_elevation_axis.hlines(
                xmin=screen_xmin,
                xmax=screen_xmax,
                y=y,
                colors=COLOR_BLACK,
                linewidth=0.5,
                zorder=3,
            )

    wl_elevation_axis.plot(
        wl_trends_dates_clean,
        wl_elevations_clean,
        marker="o",
        linewidth=1,
        linestyle="-",
        color="#050607",
        markersize=4,
        markerfacecolor="#FFFFFF",
        markeredgewidth=0.75,
        markeredgecolor="#050607",
        label="Observed Groundwater Elevation",
        zorder=4,
    )

    wl_elevation_axis.set_xlim(chem_river_stage_axis.get_xlim())
    wl_river_stage_axis.set_xlim(chem_river_stage_axis.get_xlim())
    return wl_elevation_axis, wl_river_stage_axis


def plt_legend(
    page_fig: Figure,
    grid_spec: GridSpec,
    chem_concentrations_axis: Axes,
    chem_concentrations_axis2: Axes,
    wl_elevation_axis: Axes,
    wl_river_stage_axis: Axes,
):
    wl_elevation_handles, wl_elevation_labels = [], []
    wl_river_stage_handles, wl_river_stage_labels = [], []

    if wl_elevation_axis is not None:
        wl_elevation_handles, wl_elevation_labels = (
            wl_elevation_axis.get_legend_handles_labels()
        )
    if wl_river_stage_axis is not None:
        wl_river_stage_handles, wl_river_stage_labels = (
            wl_river_stage_axis.get_legend_handles_labels()
        )
    chem_concentrations_handles, chem_concentrations_labels = (
        chem_concentrations_axis.get_legend_handles_labels()
    )
    calculated_concentrations_handles, calculated_concentrations_labels = (
        chem_concentrations_axis2.get_legend_handles_labels()
    )

    all_handles = (
        wl_elevation_handles
        + wl_river_stage_handles
        + chem_concentrations_handles
        + calculated_concentrations_handles
    )
    all_labels = (
        wl_elevation_labels
        + wl_river_stage_labels
        + chem_concentrations_labels
        + calculated_concentrations_labels
    )

    # Eventually we'll need to make a custom legend instead of grabbing all the legend info from the plots
    # Example:
    #
    # legend_elements = [
    #     (
    #       Line2D([0, 0.5], [0, 0], color='black', linewidth=0.75),
    #       Line2D([0.5, 1], [0, 0], color='black', linewidth=0.75),
    #       Line2D([0.5], [0], marker='o', linestyle='None', label='Observed Groundwater Elevation'
    #     ),
    #     Line2D([0], [0], color='green', linewidth=0.75, marker='o', label='Observed Concentration'),
    #     Line2D([0], [0], color='red', marker='v', linestyle='None', label='Non-Detect'),
    #
    # ]
    # ax.legend(handles=legend_elements)

    legend_axis = page_fig.add_subplot(grid_spec[1, 3])
    legend_axis.set_axis_off()
    legend_axis.legend(all_handles, all_labels, loc="center", frameon=False)


def generate_report(
    *,
    output_dir: Path,
    wells: pd.DataFrame,
    dist: pd.DataFrame,
    wl_trends: pd.DataFrame,
    chem_trends: pd.DataFrame,
    wl_rs: pd.DataFrame,
    chem_rs: pd.DataFrame,
    no_rs: pd.DataFrame,
    river_shapefile: Path,
    roads_shapefile: Path,
    ou_shapefile: Path,
    ous: list[str],
    map_crs: str,
    run_ver: str,
) -> None:
    # merge DIST to wells
    if "DIST" not in wells.columns:
        if "NAME" not in dist.columns or "DIST" not in dist.columns:
            raise KeyError("dist must contain NAME and DIST columns.")

        wells = wells.merge(
            dist[["NAME", "DIST"]],
            on="NAME",
            how="left",
            validate="one_to_one",
        )

        if wells["DIST"].isna().any():
            missing = wells.loc[wells["DIST"].isna(), "NAME"].tolist()
            raise ValueError(f"Missing DIST values for wells: {missing}")

    gis_wells = gpd.GeoDataFrame(
        wells,
        geometry=gpd.points_from_xy(wells["XCOORDS"], wells["YCOORDS"]),
        crs=map_crs,
    )

    gis_wells_set = set(gis_wells["NAME"])
    wl_wells_set: set[str] = set(wl_rs["NAME"])
    chem_wells_set = set(chem_rs["NAME"])
    valid_wells = gis_wells_set & chem_wells_set
    wells = wells[wells["NAME"].isin(valid_wells)]

    plt.rcParams["font.family"] = "Arial"

    plt_report(
        ous,
        wells,
        wl_trends,
        chem_trends,
        gis_wells,
        wl_rs,
        chem_rs,
        no_rs,
        wl_wells_set,
        river_shapefile,
        roads_shapefile,
        ou_shapefile,
        output_dir,
        run_ver,
    )
