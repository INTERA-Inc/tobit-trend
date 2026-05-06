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
COLOR_LIGHT_GRAY = '#E5E5E5'
COLOR_DARK_GRAY = '#B3B3B3'
COLOR_GRAY_EDGE = '#7F7F7F'
COLOR_GREEN = '#006400'
COLOR_RED = '#FF0000'
COLOR_BLACK = '#000000'
COLOR_LIGHT_BLUE = '#97C4EF'
COLOR_BISQUE = 'bisque'
COLOR_ND = '#D19999'
COLOR_ND_EDGE = '#B12224'
COLOR_CALC = '#E9B831'

# Marker colors
MARKER_FACE_COLORS = ['#ADD2BD', '#C9C5E0', '#B9B9B9', '#FFDC9B', '#9B9BEB']
MARKER_EDGE_COLORS = ['#2E8B57', '#756BB1', '#4D4D4D', '#FFA500', '#0000CD']

def plt_report(
        OUs: list[str], 
        wells: pd.DataFrame, 
        ifile_wl_trends: pd.DataFrame, 
        ifile_chem_trends: pd.DataFrame, 
        gis_wells: pd.DataFrame, 
        ifile_wl_rs: pd.DataFrame, 
        ifile_chem_rs: pd.DataFrame,
        ifile_no_rs: pd.DataFrame,
        wl_wells_set: set[str]
    ):
    for ou in OUs:
        wells_ou = wells[wells['OU'] == ou]
        output_file = f"outputs/v6_042026/TobitRegression_WLlag - {ou}_CY2026_v4_042026.pdf"
    
        with PdfPages(output_file) as pdf:
            for well in wells_ou['NAME']:
                wl_trends_well = ifile_wl_trends[ifile_wl_trends['KEY'] == well]
                chem_trends_well = ifile_chem_trends.loc[ifile_chem_trends['WELL'] == well, 'ITER']
                gis_well = gis_wells[gis_wells['NAME'] == well]
                wl_rs_well = ifile_wl_rs[ifile_wl_rs['NAME'] == well]
                chem_rs_well = ifile_chem_rs[ifile_chem_rs['NAME'] == well]

                print(f"Processing well {well} in {ou}\n")

                page_fig = plt.figure(figsize=FIGURE_SIZE)
                grid_spec = GridSpec(nrows=GRID_ROWS, ncols=GRID_COLS, wspace=GRID_WSPACE, hspace=GRID_HSPACE)

                plt_header(
                    ifile_no_rs, 
                    ifile_wl_trends, 
                    ifile_chem_trends, 
                    well, 
                    wl_trends_well, 
                    chem_trends_well, 
                    gis_well, 
                    page_fig, 
                    grid_spec
                )
                plt_gis(
                    gis_wells, 
                    ou, 
                    gis_well, 
                    page_fig, 
                    grid_spec
                )
                chem_dates, chem_concentrations_axis, chem_river_stage_axis, chem_concentrations_axis2 = plt_chem(
                    ifile_chem_trends, 
                    wl_wells_set, 
                    well, 
                    chem_trends_well, 
                    chem_rs_well, 
                    page_fig, 
                    grid_spec
                )
                wl_elevation_axis = None
                wl_river_stage_axis = None
                if well in wl_wells_set:
                    wl_elevation_axis, wl_river_stage_axis = plt_wl_rs(
                        wl_rs_well, 
                        page_fig, 
                        grid_spec, 
                        chem_dates, 
                        chem_river_stage_axis
                    )
                plt_legend(
                    page_fig, 
                    grid_spec, 
                    chem_concentrations_axis, 
                    chem_concentrations_axis2, 
                    wl_elevation_axis, 
                    wl_river_stage_axis
                )

                page_fig.subplots_adjust(left=FIGURE_LEFT_MARGIN, right=FIGURE_RIGHT_MARGIN)
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
        grid_spec: GridSpec
    ):
    page_fig.text(0.5, 0.95, f"{well}", ha='center', va='top', fontsize=FONT_SIZE_TITLE, fontstyle='italic', fontweight='light')
    page_fig.text(
                0.5, 
                0.925, 
f"""
Distance to River: {round(gis_well.at[gis_well.index[0], 'DIST'])} m
Number of Trends Calculated: {len(chem_trends_well)}
""",
                ha='center',
                va='top',
                fontsize=FONT_SIZE_HEADER,
                fontstyle='italic'
            )

    trends_axis = page_fig.add_subplot(grid_spec[0, 2])
    trends_axis.axis('off')
    trends_rows = [
                'Est. Lag Time (days):',
                'Significance of Trend (p-value):',
                'Significance of Date (p-value):',
                'Number of Observations:',
                'Percent NDs:'
            ]
    trends_columns: list[str] = []
    trends_values: list[str] = []

    no_river_stage = well in ifile_no_rs['NAME'].values
    if not no_river_stage:
        trends_rows.insert(2, 'Significance of River Stage (p-value):')
            
    if (len(wl_trends_well) == 1):
        trends_columns.append('WL')
    if (len(chem_trends_well) == 1):
        trends_columns.append('Conc')
    else:
        trends_columns += [f'Trend{n}' for n in range(1, len(chem_trends_well) + 1)]

    for _, row_label in enumerate(trends_rows):
        row_values: list[str] = []

        # WL column
        if len(wl_trends_well) == 1:
            wl_lag = ifile_wl_trends.loc[ifile_wl_trends['KEY'] == well, 'LAG'].item()
            wl_p_trend = ifile_wl_trends.loc[ifile_wl_trends['KEY'] == well, 'p_trend'].item()
            wl_p_interp = ifile_wl_trends.get('p_interp', pd.Series(0, index=ifile_wl_trends.index)).loc[ifile_wl_trends['KEY'] == well].item()
            wl_p_event = ifile_wl_trends.get('p_event', pd.Series(0, index=ifile_wl_trends.index)).loc[ifile_wl_trends['KEY'] == well].item()
            wl_n_obs = ifile_wl_trends.get('n_obs', pd.Series(0, index=ifile_wl_trends.index)).loc[ifile_wl_trends['KEY'] == well].item()
                    
            if row_label == 'Est. Lag Time (days):':
                row_values.append(f"{round(wl_lag)}" if ~np.isnan(wl_lag) else 'NA')
            elif row_label == 'Significance of Trend (p-value):':
                row_values.append(f'{wl_p_trend:.2g}' if ~np.isnan(wl_p_trend) else 'NA')
            elif row_label == 'Significance of River Stage (p-value):':
                row_values.append(f'{wl_p_interp:.2g}' if ~np.isnan(wl_p_interp) else 'NA')
            elif row_label == 'Significance of Date (p-value):':
                row_values.append(f'{wl_p_event:.2g}' if ~np.isnan(wl_p_event) else 'NA')
            elif row_label == 'Number of Observations:':
                row_values.append(f'{round(wl_n_obs)}' if ~np.isnan(wl_n_obs) else 'NA')
            else:
                row_values.append('')
                
        # Chem columns
        for trend_idx in range(len(chem_trends_well)):
            chem_trend_row = ifile_chem_trends[ifile_chem_trends['WELL'] == well].iloc[trend_idx]
            chem_lag = 'NA'
                    
            if no_river_stage and trend_idx == 0:
                chem_lag = 'No Covariate'
            elif no_river_stage and trend_idx != 0:
                chem_lag = 'No RS'
            elif ~np.isnan(chem_trend_row['LAG']):
                chem_lag = f"{round(chem_trend_row['LAG'])}"

            if row_label == 'Est. Lag Time (days):':
                row_values.append(chem_lag)
            elif row_label == 'Significance of Trend (p-value):':
                row_values.append(f'{chem_trend_row['p_trend']:.2g}' if ~np.isnan(chem_trend_row['p_trend']) else 'NA')
            elif row_label == 'Significance of River Stage (p-value):':
                row_values.append(f'{chem_trend_row['p_interp']:.2g}' if ~np.isnan(chem_trend_row['p_interp']) else 'NA')
            elif row_label == 'Significance of Date (p-value):':
                row_values.append(f'{chem_trend_row['p_event']:.2g}' if ~np.isnan(chem_trend_row['p_event']) else 'NA')
            elif row_label == 'Number of Observations:':
                row_values.append(f'{round(chem_trend_row['n_obs'])}' if ~np.isnan(chem_trend_row['n_obs']) else 'NA')
            elif row_label == 'Percent NDs:':
                row_values.append(f'  {round(chem_trend_row['n_cens'] / chem_trend_row['n_obs'] * 100)}%' if ~np.isnan(chem_trend_row['n_cens']) else 'NA')
                
        trends_values.append(row_values)

    trends_table = trends_axis.table(
                cellText=trends_values,
                rowLabels=trends_rows,
                colLabels=trends_columns,
                loc='center',
                rowLoc='right',
                cellLoc='center',
                edges='open',
            )
    trends_table.auto_set_column_width(col=list(range(len(trends_columns))))
    trends_table.auto_set_font_size(False)
    trends_table.set_fontsize(FONT_SIZE_TABLE)

    for (row, col), cell in trends_table.get_celld().items():
        if col == -1:
            cell.set_text_props(fontproperties=FontProperties(style='italic'))
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

def plt_gis(
        gis_wells: gpd.GeoDataFrame, 
        ou: str, 
        gis_well: gpd.GeoDataFrame, 
        page_fig: Figure, 
        grid_spec: GridSpec
    ):
    ifile_gis_highriv = gpd.read_file('input/gis_export/HIGHRIV.shp')
    ifile_gis_roads = gpd.read_file('input/gis_export/ROADS.shp')
    ifile_gis_ous = gpd.read_file('input/gis_export/OU.shp')
    gis_roads = ifile_gis_roads.to_crs(ifile_gis_highriv.crs)
    ifile_gis_ous = ifile_gis_ous.to_crs(ifile_gis_highriv.crs)
    gis_ou = ifile_gis_ous[ifile_gis_ous['Name'] == ou]
    gis_ou = gis_ou.to_crs(ifile_gis_highriv.crs)

    gis_axis = page_fig.add_subplot(grid_spec[1, 0:2])
    gis_roads.plot(ax=gis_axis, color=COLOR_LIGHT_GRAY, linewidth=1, markersize=50, zorder=1)
    ifile_gis_highriv.plot(ax=gis_axis, color=COLOR_LIGHT_GRAY, edgecolor=COLOR_BLACK, linewidth=1, zorder=2)
    gis_ou.plot(ax=gis_axis, color='none', edgecolor=COLOR_GREEN, linewidth=1.5, zorder=3)
    gis_wells.plot(ax=gis_axis, color=COLOR_DARK_GRAY, edgecolor=COLOR_GRAY_EDGE, markersize=4, linewidth=0.5, zorder=4)
    gis_well.plot(ax=gis_axis, color=COLOR_RED, marker='o', edgecolor=COLOR_BLACK, markersize=12, linewidth=0.75, zorder=5)
    gis_axis.set_xticks([])
    gis_axis.set_yticks([])
    gis_axis.set_xlim([GIS_X_MIN, GIS_X_MAX])
    gis_axis.set_ylim([GIS_Y_MIN, GIS_Y_MAX])

def plt_chem(
        ifile_chem_trends: pd.DataFrame,
        wl_wells_set: set[str],
        well: str,
        chem_trends_well: pd.Series,
        chem_rs_well: pd.DataFrame,
        page_fig: Figure,
        grid_spec: GridSpec
    ) -> Tuple[pd.DatetimeIndex, Axes, Axes, Axes]:
    chem_rs_well_clean = chem_rs_well.loc[chem_rs_well['NDS'].notna() & chem_rs_well['TERM'].notna()]
    chem_concentrations = chem_rs_well['VAL']
    chem_concentrations_clean = chem_concentrations[~np.isnan(chem_concentrations)]
    chem_concentrations_NDS = chem_rs_well_clean.loc[chem_rs_well_clean['NDS'] == True, 'VAL']
    chem_dates = pd.to_datetime(chem_rs_well['EVENT'])
    chem_dates_clean = chem_dates[~np.isnan(chem_concentrations)]
    chem_dates_NDS = pd.to_datetime(chem_rs_well_clean.loc[chem_rs_well_clean['NDS'] == True, 'EVENT'])
    chem_river_stages = chem_rs_well['INTERP']
            
    print(f"Chem TRENDS WELL CLEAN\n{chem_rs_well_clean}\n")
            
    chem_concentrations_axis = page_fig.add_subplot(grid_spec[3, :] if well in wl_wells_set else grid_spec[2, :])
    chem_concentrations_axis.set_facecolor(COLOR_LIGHT_GRAY)
    chem_min = chem_concentrations.min(skipna=True)
    chem_max = chem_concentrations.max(skipna=True)
            
    chem_concentrations_axis.xaxis.set_major_locator(YearLocator())
    chem_concentrations_axis.xaxis.set_major_formatter(DateFormatter('%Y'))
            
    chem_concentrations_axis.set_yscale('log')
    chem_concentrations_axis.yaxis.set_major_locator(LogLocator(base=10))
    chem_concentrations_axis.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    chem_concentrations_axis.yaxis.set_minor_formatter(NullFormatter())
    ymin = 10 ** np.floor(np.log10(chem_min))
    ymax = 10 **np.ceil(np.log10(chem_max))
    chem_concentrations_axis.set_ylim(ymin - 0.1, ymax + 1)
    chem_concentrations_axis.set_ylabel('Hex. & Filt. Cr (µg/L)')

    chem_concentrations_axis.grid(True, which='major', axis='x', linewidth=0.5, color='#FFFFFF')
    chem_concentrations_axis.grid(True, which='major', axis='y', linewidth=0.5, color=COLOR_BLACK)
    chem_concentrations_axis.grid(True, which='minor', axis='y', linewidth=0.5, color='#FFFFFF')

    chem_river_stage_axis = chem_concentrations_axis.twinx()
    chem_river_stage_axis.set_ylabel('River Stage (m amsl)')
    min_stage = 2 * np.floor((chem_river_stages.min(skipna=True) - 1) / 2) + 1
    max_stage = 2 * np.ceil((chem_river_stages.max(skipna=True) - 1) / 2) + 1
    ticks = np.arange(min_stage, max_stage + 2, 2)

    chem_river_stage_axis.set_yticks(ticks)
    chem_river_stage_axis.set_ylim(min_stage - 0.25, max_stage + 0.25)

    chem_trends_dates_clean_numeric = (chem_dates_clean - chem_dates_clean.min()).dt.days / 365.25
    chem_trends_dates_numeric = (chem_dates - chem_dates.min()).dt.days / 365.25

    chem_concentrations_clean_log = np.log1p(chem_concentrations_clean)
            
    coeffs = np.polyfit(chem_trends_dates_clean_numeric, chem_concentrations_clean_log, deg=1)
    mock_trend = np.poly1d(coeffs)

    calculated_concentrations_trend = np.expm1(mock_trend(chem_trends_dates_numeric))
    calculated_concentrations_trend = np.maximum(calculated_concentrations_trend, 0)

    chem_concentrations_axis2 = page_fig.add_subplot(grid_spec[4, :] if well in wl_wells_set else grid_spec[3, :])
    chem_concentrations_axis2.set_facecolor(COLOR_LIGHT_GRAY)

    chem_concentrations_axis2.xaxis.set_major_locator(YearLocator())
    chem_concentrations_axis2.xaxis.set_major_formatter(DateFormatter('%Y'))

    chem_concentrations_axis2.set_yscale('log')
    chem_concentrations_axis2.yaxis.set_major_locator(LogLocator(base=10))
    chem_concentrations_axis2.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    chem_concentrations_axis2.yaxis.set_minor_formatter(NullFormatter())
    ymin = 10 ** np.floor(np.log10(chem_min))
    ymax = 10 **np.ceil(np.log10(chem_max))
    chem_concentrations_axis2.set_ylim(ymin - 0.1, ymax + 1)
    chem_concentrations_axis2.set_ylabel('Hex. & Filt. Cr (µg/L)')
            
    chem_concentrations_axis2.grid(True, which='major', axis='x', linewidth=0.5, color='#FFFFFF')
    chem_concentrations_axis2.grid(True, which='major', axis='y', linewidth=0.5, color=COLOR_BLACK)
    chem_concentrations_axis2.grid(True, which='minor', axis='y', linewidth=0.5, color='#FFFFFF')

    marker_face_colors = MARKER_FACE_COLORS
    marker_edge_colors = MARKER_EDGE_COLORS

    chem_river_stage_axis.plot(
                chem_dates, 
                chem_river_stages, 
                linewidth=0.75, 
                color=COLOR_LIGHT_BLUE,
                label='River Stage'
            )

    model_equation_text: list[str] = []
    num_trend_equations = 0
            
    for trend_idx in range(len(chem_trends_well)):
        row = ifile_chem_trends.loc[
                    (ifile_chem_trends['WELL'] == well) &
                    (ifile_chem_trends['ITER'] == trend_idx + 1)
                ].iloc[0]

        beta_interp: float = row['beta_interp']
        beta_event: float = row['beta_event']
        beta_intercept: float = row['beta_intercept']

        if row['p_trend'] > 0.05:
            model_equation_text.append(f'Trend{trend_idx + 1}:\nTrend Not Significant\n')
            num_trend_equations += 1
        elif ~np.isnan(beta_interp) and ~np.isnan(beta_event) and ~np.isnan(beta_intercept):
            model_equation_text.append(
f"""Trend{trend_idx + 1}:
In Conc. = {beta_interp} (+/- MISSING)*River Stage + {beta_event} (+/- MISSING)*Date + {beta_intercept} (+/- MISSING)
"""
                    )
            num_trend_equations += 1
        else:
            model_equation_text.append(f'Trend{trend_idx + 1}:\nNo Trend Analysis Performed')
            
    if num_trend_equations == 0:
        chem_concentrations_axis.tick_params(axis='x', labelrotation=90)
        page_fig.text(
                    x=0.5,
                    y=0.35,
                    s='No Regression Analysis Performed',
                    fontsize=FONT_SIZE_TEXT,
                    ha='center',
                    va='center'
                )
    else:
        chem_concentrations_axis.set_xticklabels([])
        chem_concentrations_axis2.tick_params(axis='x', labelrotation=90)
        page_fig.text(
                    x=0.5,
                    y=0.17,
                    s='Censored Regression (Tobit) Model',
                    fontsize=FONT_SIZE_TEXT,
                    ha='center',
                    va='center'
                )
        for i in range(len(model_equation_text)):
            page_fig.text(
                        x=0.5,
                        y=0.15 - 0.025 * i,
                        s=model_equation_text[i],
                        fontsize=FONT_SIZE_TEXT,
                        ha='center',
                        va='top'
                    )
            
    if not chem_concentrations_NDS.empty:
        chem_concentrations_axis.plot(
                    chem_dates_NDS, 
                    chem_concentrations_NDS, 
                    marker='v', 
                    linewidth=0, 
                    markersize=4,
                    markerfacecolor=COLOR_ND,
                    markeredgewidth=0.75,
                    markeredgecolor=COLOR_ND_EDGE,
                    label='Non-Detect',
                )

    for trend_idx in range(len(chem_trends_well)):
        chem_concentrations_trend = chem_rs_well_clean.loc[chem_rs_well_clean['TERM'].astype(int) == trend_idx + 1, 'VAL']
        chem_concentrations_clean_trend = chem_concentrations_trend[~np.isnan(chem_concentrations_trend)]
        chem_concentrations_detects_trend = chem_rs_well_clean.loc[
                    (chem_rs_well_clean['NDS'] == False) & 
                    (chem_rs_well_clean['TERM'].astype(int) == trend_idx + 1), 'VAL'
                ]
        chem_dates_trend = pd.to_datetime(chem_rs_well_clean.loc[chem_rs_well_clean['TERM'].astype(int) == trend_idx + 1, 'EVENT'])
        chem_dates_clean_trend = chem_dates_trend[~np.isnan(chem_concentrations_trend)]
        chem_dates_detects_trend = pd.to_datetime(chem_rs_well_clean.loc[
                    (chem_rs_well_clean['NDS'] == False) &
                    (chem_rs_well_clean['TERM'].astype(int) == trend_idx + 1), 'EVENT'
                ])

        chem_concentrations_axis.plot(
                    chem_dates_clean_trend, 
                    chem_concentrations_clean_trend, 
                    linewidth=1,
                    color=marker_edge_colors[trend_idx],
                )
                
        if not chem_concentrations_detects_trend.empty:
            chem_concentrations_axis.plot(
                        chem_dates_detects_trend,
                        chem_concentrations_detects_trend,
                        linewidth=0,
                        marker='o',
                        markersize=4,
                        markerfacecolor=marker_face_colors[trend_idx],
                        markeredgewidth=0.75,
                        markeredgecolor=marker_edge_colors[trend_idx],
                        label='Observed Concentration' if len(chem_trends_well) == 1 else f'Observed Conc. (Trend{trend_idx + 1})',
                    )
                
        if num_trend_equations > 0:
            chem_concentrations_axis2.plot(
                        chem_dates_clean_trend, 
                        chem_concentrations_clean_trend, 
                        linewidth=1,
                        color=marker_edge_colors[trend_idx],
                    )
            if not chem_concentrations_detects_trend.empty:
                chem_concentrations_axis2.plot(
                            chem_dates_detects_trend,
                            chem_concentrations_detects_trend,
                            linewidth=0,
                            marker='o',
                            markersize=4,
                            markerfacecolor=marker_face_colors[trend_idx],
                            markeredgewidth=0.75,
                            markeredgecolor=marker_edge_colors[trend_idx],
                        )
    if num_trend_equations == 0:
        chem_concentrations_axis2.set_axis_off()
    else:
        if not chem_concentrations_NDS.empty:
            chem_concentrations_axis2.plot(
                        chem_dates_NDS, 
                        chem_concentrations_NDS, 
                        marker='v', 
                        linewidth=0, 
                        markersize=4,
                        markerfacecolor=COLOR_ND,
                        markeredgewidth=0.75,
                        markeredgecolor=COLOR_ND_EDGE,
                    )
        chem_concentrations_axis2.plot(
                    chem_dates, 
                    calculated_concentrations_trend, 
                    linestyle='-', 
                    color=COLOR_CALC,
                    label='Calculated Conc.'
                )
        
    return chem_dates, chem_concentrations_axis, chem_river_stage_axis, chem_concentrations_axis2

def plt_wl_rs(
        wl_rs_well: pd.DataFrame,
        page_fig: Figure,
        grid_spec: GridSpec,
        chem_dates: pd.DatetimeIndex,
        chem_river_stage_axis: Axes
    ) -> Tuple[Axes, Axes]:
    wl_elevations: pd.Series[float] = wl_rs_well['WLE']
    wl_river_stages: pd.Series[float] = wl_rs_well['INTERP']
    wl_trends_dates = pd.to_datetime(wl_rs_well['EVENT'])
    wl_elevations_clean = wl_elevations[~np.isnan(wl_elevations)]
    wl_trends_dates_clean = wl_trends_dates[~np.isnan(wl_elevations)]

    wl_elevation_axis = page_fig.add_subplot(grid_spec[2, :])
    wl_elevation_axis.set_facecolor(COLOR_LIGHT_GRAY)
    wl_elevation_axis.grid(True, linewidth=0.5, color='#FFFFFF')
    wl_elevation_axis.xaxis.set_major_locator(YearLocator())
    wl_elevation_axis.xaxis.set_major_formatter(DateFormatter('%Y'))
    wl_elevation_axis.set_xticklabels([])
    wl_elevation_axis.set_ylabel('Water-Level (m amsl)')

    wl_river_stage_axis = wl_elevation_axis.twinx()
    min_stage = 2 * np.floor((wl_river_stages.min(skipna=True) - 1) / 2) + 1
    max_stage = 2 * np.ceil((wl_river_stages.max(skipna=True) - 1) / 2) + 1
    ticks = np.arange(min_stage, max_stage + 2, 2)
    wl_river_stage_axis.set_ylim(min_stage - 0.25, max_stage + 0.25)
    wl_river_stage_axis.set_yticks(ticks)
    wl_river_stage_axis.set_ylabel('River Stage (m amsl)')

    wl_trends_dates_filtered = wl_trends_dates[wl_trends_dates >= chem_dates.min()]
    wl_river_stages_filtered = wl_river_stages[-1 * wl_trends_dates_filtered.count():]

    page_size = page_fig.get_size_inches() * 2.54 # cm
    screen_xrange = wl_trends_dates.max().year - wl_trends_dates.min().year
    screen_xgrid = screen_xrange / (0.96 * page_size[0])
    screen_xmax = pd.Timestamp(year=chem_dates.min().year - 1, month=10, day=1)
    screen_xmin = screen_xmax - pd.Timedelta(days=0.02 * page_size[0] * screen_xgrid * 365.25)
    epoch = pd.Timestamp(year=1970, month=1, day=1)
    screen_xmax = (screen_xmax - epoch) / pd.Timedelta(days=1)
    screen_xmin = (screen_xmin - epoch) / pd.Timedelta(days=1)
    screen_ymin = wl_rs_well['BOT'].iloc[0]
    screen_ymax = wl_rs_well['TOP'].iloc[0]

    screened_interval = patches.Rectangle(
                xy=(screen_xmin, screen_ymin), 
                width=screen_xmax - screen_xmin, 
                height=screen_ymax - screen_ymin,
                facecolor=COLOR_BISQUE,
                edgecolor=COLOR_BLACK,
                linewidth=0.5,
                zorder=10,
            )

    if (screen_ymin <= wl_elevations_clean.max() and 
                screen_ymax >= wl_elevations_clean.min()):
        wl_elevation_axis.add_patch(screened_interval)
                
        n_lines = 5
        screen_line_ypositions = np.linspace(screen_ymin, screen_ymax, n_lines + 2)[1:-1]

        for y in screen_line_ypositions:
            wl_elevation_axis.hlines(
                        xmin=screen_xmin,
                        xmax=screen_xmax,
                        y=y,
                        colors=COLOR_BLACK,
                        linewidth=0.5,
                        zorder=11
                    )

    wl_elevation_axis.plot(
                wl_trends_dates_clean, 
                wl_elevations_clean, 
                marker='o', 
                linewidth=1, 
                linestyle='-', 
                color='#050607', 
                markersize=4,
                markerfacecolor='#FFFFFF', 
                markeredgewidth=0.75, 
                markeredgecolor='#050607',
                label='Observed Groundwater Elevation',
                zorder=3
            )

    wl_river_stage_axis.plot(
                wl_trends_dates_filtered,
                wl_river_stages_filtered,
                linewidth=0.75, 
                color=COLOR_LIGHT_BLUE,
                label='River Stage',
                zorder=2
            )

    wl_river_stage_axis.set_xlim(chem_river_stage_axis.get_xlim())
    return wl_elevation_axis, wl_river_stage_axis

def plt_legend(
        page_fig: Figure, 
        grid_spec: GridSpec, 
        chem_concentrations_axis: Axes, 
        chem_concentrations_axis2: Axes, 
        wl_elevation_axis: Axes, 
        wl_river_stage_axis: Axes
    ):
    wl_elevation_handles, wl_elevation_labels = [], []
    wl_river_stage_handles, wl_river_stage_labels = [], []
    
    if wl_elevation_axis is not None:
        wl_elevation_handles, wl_elevation_labels = wl_elevation_axis.get_legend_handles_labels()
    if wl_river_stage_axis is not None:
        wl_river_stage_handles, wl_river_stage_labels = wl_river_stage_axis.get_legend_handles_labels()    
    chem_concentrations_handles, chem_concentrations_labels = chem_concentrations_axis.get_legend_handles_labels()
    calculated_concentrations_handles, calculated_concentrations_labels = chem_concentrations_axis2.get_legend_handles_labels()

    all_handles = wl_elevation_handles + wl_river_stage_handles + chem_concentrations_handles + calculated_concentrations_handles
    all_labels = wl_elevation_labels + wl_river_stage_labels + chem_concentrations_labels + calculated_concentrations_labels

    # Eventually we'll need to make a custom legend instead of grabbing all the legend info from the plots
    # Example:
    # legend_elements = [
    #     Line2D([0], [0], color='green', lw=1, marker='o', label='Observed Concentration'),
    #     Line2D([0], [0], color='red', marker='v', linestyle='None', label='Non-Detect')
    # ]
    # ax.legend(handles=legend_elements)

    legend_axis = page_fig.add_subplot(grid_spec[1, 3])
    legend_axis.set_axis_off()
    legend_axis.legend(all_handles, all_labels, loc='center', frameon=False)

def main():
    ifile_wells = pd.read_csv('input/Cr_TrendData/Cr_TrendData_unique_wells.csv')
    ifile_wl_trends = pd.read_csv('input/WLTrends_flat.csv')
    ifile_chem_trends = pd.read_csv('outputs/v6_042026/TobitResults.csv')
    ifile_wl_rs = pd.read_parquet('input/WL_TrendData_2024/WL_RS.parquet')
    ifile_chem_rs = pd.read_parquet('input/Cr_TrendData/R_prepped_chem_rs.parquet')
    ifile_no_rs = pd.read_csv('input/NoRS.csv')
    
    print("Wells\n", ifile_wells.head(), "\n")
    print("WL Trends\n", ifile_wl_trends.head(), "\n")
    print("Cr Trends WL Lag\n", ifile_chem_trends.head(), "\n")
    print("WL RS Parquet\n", ifile_wl_rs.head(), "\n")
    print("Prepped Chem RS Parquet\n", ifile_chem_rs[ifile_chem_rs['VAL'].notna()], "\n")
    print("No River Stage\n", ifile_no_rs.head(), "\n")

    OUs = ['100-HR-3-D', '100-HR-3-H', '100-KR-4']

    gis_wells = gpd.GeoDataFrame(ifile_wells, geometry=gpd.points_from_xy(ifile_wells['XCOORDS'], ifile_wells['YCOORDS']), crs='EPSG:2926')
    gis_wells_set = set(gis_wells['NAME'])
    wl_wells_set: set[str] = set(ifile_wl_rs['NAME'])
    chem_wells_set = set(ifile_chem_rs['NAME'])
    valid_wells = gis_wells_set & chem_wells_set
    wells = ifile_wells[ifile_wells["NAME"].isin(valid_wells)]

    plt.rcParams['font.family'] = 'Arial'

    plt_report(OUs, wells, ifile_wl_trends, ifile_chem_trends, gis_wells, ifile_wl_rs, ifile_chem_rs, ifile_no_rs, wl_wells_set)

if __name__ == "__main__":
    main()
