import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.backends.backend_pdf import PdfPages

# Import well information data
ifile_wells = pd.read_csv('input/Cr_TrendData_unique_wells.csv')
print("IFILE WELLS\n", ifile_wells.head(), "\n")

ifile_no_riv_stage = pd.read_csv('input/NoRS.csv')
print("No River Stage\n", ifile_no_riv_stage.head(), "\n")

cr_trends_wl_lag = pd.read_csv('input/R_CrTrends_WLLAG.csv')
wl_trends = pd.read_csv('input/WLTrends_flat.csv')
print("Cr Trends WL Lag\n", cr_trends_wl_lag.head(), "\n")
print("WL Trends\n", wl_trends.head(), "\n")

wl_rs_parquet = pd.read_parquet('input/WL_RS.parquet')
print("WL RS Parquet\n", wl_rs_parquet.head(), "\n")

prepped_chem_rs_parquet = pd.read_parquet('input/Cr_TrendData.parquet')
print("Prepped Chem RS Parquet\n", prepped_chem_rs_parquet[prepped_chem_rs_parquet['VAL'].notna()], "\n")

ifile_gis_highriv = gpd.read_file('input/gis_export/HIGHRIV.shp')
ifile_gis_roads = gpd.read_file('input/gis_export/ROADS.shp')
ifile_gis_ous = gpd.read_file('input/gis_export/OU.shp')
ifile_gis_wells = gpd.GeoDataFrame(ifile_wells, geometry=gpd.points_from_xy(ifile_wells['XCOORDS'], ifile_wells['YCOORDS']), crs='EPSG:2926')

CUTOFF = pd.DataFrame({
    'SYSTEM': ['DX', 'HX', 'KX_KW_KR4'],
    'EVENT': pd.to_datetime(['2011-01-01', '2011-11-01', '2009-04-01'])
})

OUs = ['100-HR-3-D', '100-HR-3-H', '100-KR-4']

gis_wells = set(ifile_gis_wells['NAME'])
wl_wells = set(wl_rs_parquet['NAME'])
cr_wells = set(prepped_chem_rs_parquet['NAME'])

valid_wells = gis_wells & wl_wells & cr_wells

wells = ifile_wells[ifile_wells["NAME"].isin(valid_wells)]
print("ALL VALID WELLS:", wells, "\n")

gis_roads = ifile_gis_roads.to_crs(ifile_gis_highriv.crs)
ifile_gis_ous = ifile_gis_ous.to_crs(ifile_gis_highriv.crs)

plt.rcParams['font.family'] = 'Arial'

for ou in OUs:
    wells_ou = wells[wells['OU'] == ou]
    num_wells = len(wells_ou)
    print("WELLS OU:\n", wells_ou, "\n")

    fn = f"output/TobitRegression_WLlag - {ou}_CY2026_v4_041526.pdf"
    with PdfPages(fn) as pdf:
        for i in range(num_wells):
            well = wells_ou.iloc[i, 0]
            gis_well = ifile_gis_wells[ifile_gis_wells['NAME'] == well]
            print(f"GIS well for {well} in {ou}\n", gis_well, "\n")
            wl_trends_well = wl_rs_parquet[wl_rs_parquet['NAME'] == well]
            print(f"WL trends for well {well} in {ou}\n", wl_trends_well.head(), "\n")
            cr_trends_well = prepped_chem_rs_parquet[prepped_chem_rs_parquet['NAME'] == well]
            print(f"Cr trends for well {well} in {ou}\n", cr_trends_well.head(), "\n")

            gis_ou = ifile_gis_ous[ifile_gis_ous['Name'] == ou]
            print("GIS OU HEAD\n", gis_ou.head(), "\n")

            gis_ou = gis_ou.to_crs(ifile_gis_highriv.crs)

            page_fig = plt.figure(figsize=(8.5, 11))
            grid_spec = GridSpec(nrows=6, ncols=4, wspace=0.1, hspace=0.1)

            page_fig.text(0.5, 0.95, f"{well}", ha='center', va='top', fontsize=20, fontstyle='italic', fontweight='light')

            well_cr_trends = cr_trends_wl_lag.loc[cr_trends_wl_lag['WELL'] == well, 'ITER'].values

            page_fig.text(
                0.5, 
                0.91, 
f"""Distance to River: {round(gis_well.at[gis_well.index[0], 'DIST'])} m
Number of Trends Calculated: {len(well_cr_trends)}
""",
                ha='center',
                va='top',
                fontsize=10,
                fontstyle='italic'
            )

            trends_axis = page_fig.add_subplot(grid_spec[0, 2])
            trends_axis.axis('off')
            trends_rows = [
                'Est. Lag Time (days):', 
                'Significance of Trend (p-value):', 
                'Significance of River Stage (p−value):', 
                'Significance of Date (p−value):', 
                'Number of Observations:', 
                'Percent NDs:'
            ]
            trends_columns = ['WL'] + ['Conc' if len(well_cr_trends) == 1 else f'Trend{n}' for n in range(1, len(well_cr_trends) + 1)]
            trends_values = []
            for row_idx, row_label in enumerate(trends_rows):
                row_values = []
                wl_lag = wl_trends.loc[wl_trends["KEY"] == well, "LAG"].item()
                wl_p_trend = wl_trends.loc[wl_trends["KEY"] == well, "p_trend"].item()
                # WL column
                if row_idx == 0:  # Lag Time
                    row_values.append(f'{round(wl_lag) if ~np.isnan(wl_lag) else 'NA'}')
                elif row_idx == 1:  # p_trend
                    row_values.append(f'{round(wl_p_trend, 3) if ~np.isnan(wl_p_trend) else 'NA'}')
                else:
                    row_values.append('')
                
                # Conc/Trend columns
                for trend_idx in range(len(well_cr_trends)):
                    trend_row = cr_trends_wl_lag[cr_trends_wl_lag['WELL'] == well].iloc[trend_idx]
                    if row_idx == 0:  # Lag Time
                        row_values.append(f'{round(trend_row["LAG"])}' if ~np.isnan(trend_row["LAG"]) else 'NA')
                    elif row_idx == 1:  # p_trend
                        row_values.append(f'{round(trend_row["p_trend"], 3)}' if ~np.isnan(trend_row["p_trend"]) else 'NA')
                    elif row_idx == 2:  # p_interp
                        row_values.append(f'{round(trend_row["p_interp"], 3)}' if ~np.isnan(trend_row["p_interp"]) else 'NA')
                    elif row_idx == 3:  # p_event
                        row_values.append(f'{round(trend_row["p_event"], 3)}' if ~np.isnan(trend_row["p_event"]) else 'NA')
                    elif row_idx == 4:  # n_obs
                        row_values.append(f'{round(trend_row["n_obs"], 3)}' if ~np.isnan(trend_row["n_obs"]) else 'NA')
                    elif row_idx == 5:  # Percent NDs
                        row_values.append('')
                
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
            trends_table.set_fontsize(10)

            for (row, col), cell in trends_table.get_celld().items():
                if col == -1:
                    cell.set_text_props(fontproperties=FontProperties(style='italic'))
                if row == 0:
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

            gis_axis = page_fig.add_subplot(grid_spec[1, 0:2])

            gis_roads.plot(ax=gis_axis, color='#E5E5E5', linewidth=1, markersize=50, zorder=1)
            ifile_gis_highriv.plot(ax=gis_axis, color='#E5E5E5', edgecolor='black', linewidth=1, zorder=2)
            gis_ou.plot(ax=gis_axis, color='none', edgecolor='#006400', linewidth=1.5, zorder=3)
            ifile_gis_wells.plot(ax=gis_axis, color='#B3B3B3', edgecolor='#7F7F7F', markersize=4, linewidth=0.5, zorder=4)
            gis_well.plot(ax=gis_axis, color='#FF0000', marker='o', edgecolor='#000000', markersize=12, linewidth=0.75, zorder=5)
            
            gis_axis.set_xticks([])
            gis_axis.set_yticks([])

            gis_axis.set_xlim([561500, 586000])
            gis_axis.set_ylim([145000, 155000])

            wl_elevations = wl_trends_well['WLE']
            wl_river_stages = wl_trends_well['INTERP']
            wl_trends_dates = pd.to_datetime(wl_trends_well['EVENT'])

            wl_elevations_clean = wl_elevations[~np.isnan(wl_elevations)]
            wl_trends_dates_clean = wl_trends_dates[~np.isnan(wl_elevations)]

            wl_elevation_axis = page_fig.add_subplot(grid_spec[2, :])
            wl_elevation_axis.set_facecolor('#E5E5E5')
            wl_elevation_axis.grid(True, linewidth=0.5, color='#FFFFFF')
            
            wl_elevation_axis.xaxis.set_major_locator(YearLocator())
            wl_elevation_axis.xaxis.set_major_formatter(DateFormatter('%Y'))
            wl_elevation_axis.set_xticklabels([])

            wl_elevation_axis.set_ylim(wl_elevations.min(skipna=True) - 2, wl_elevations.max(skipna=True) + 2)
            wl_elevation_axis.set_ylabel('Water-Level (m amsl)')
            wl_elevation_plot_lines = wl_elevation_axis.plot(
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
                label='Observed Groundwater Elevation'
            )

            wl_river_stage_axis = wl_elevation_axis.twinx()
            min_stage = 2 * np.floor((wl_river_stages.min(skipna=True) - 1) / 2) + 1
            max_stage = 2 * np.ceil((wl_river_stages.max(skipna=True) - 1) / 2) + 1
            ticks = np.arange(min_stage, max_stage + 2, 2)
            wl_river_stage_axis.set_yticks(ticks)
            wl_river_stage_axis.set_ylim(min_stage - 0.25, max_stage + 0.25)
            wl_river_stage_axis.set_ylabel('River Stage (m amsl)')
            wl_river_stage_plot_lines = wl_river_stage_axis.plot(wl_trends_dates, wl_river_stages, linewidth=0.75, color='#97C4EF', label='River Stage')

            cr_trends_dates = pd.to_datetime(cr_trends_well['EVENT'])
            cr_concentrations = cr_trends_well['VAL']
            cr_trends_dates_clean = cr_trends_dates[~np.isnan(cr_concentrations)]
            cr_concentrations_clean = cr_concentrations[~np.isnan(cr_concentrations)]
            cr_river_stages = cr_trends_well['INTERP']
            print(f"Processing well {wells_ou.iloc[i, 0]} in {ou}\n")
            print("CR CONCENTRATIONS CLEAN\n", cr_concentrations_clean, "\n")

            cr_concentrations_axis = page_fig.add_subplot(grid_spec[3, :])
            cr_concentrations_axis.set_facecolor('#E5E5E5')
            cr_min = cr_concentrations_clean.min(skipna=True)
            cr_max = cr_concentrations_clean.max(skipna=True)
            
            cr_concentrations_axis.xaxis.set_major_locator(YearLocator())
            cr_concentrations_axis.xaxis.set_major_formatter(DateFormatter('%Y'))
            cr_concentrations_axis.set_xticklabels([]) 
            
            cr_concentrations_axis.set_yscale('log')
            cr_concentrations_axis.yaxis.set_major_locator(LogLocator(base=10))
            cr_concentrations_axis.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            cr_concentrations_axis.yaxis.set_minor_formatter(NullFormatter())
            ymin = 10 ** np.floor(np.log10(cr_min))
            ymax = 10 **np.ceil(np.log10(cr_max))
            print("clean size:", len(cr_concentrations_clean))
            print("min:", cr_min)
            print("max:", cr_max)
            cr_concentrations_axis.set_ylim(ymin - 0.1, ymax + 1)
            cr_concentrations_axis.set_ylabel('Hex. & Filt. Cr (µg/L)')

            cr_concentrations_axis.grid(True, which='major', axis='x', linewidth=0.5, color='#FFFFFF')
            cr_concentrations_axis.grid(True, which='major', axis='y', linewidth=0.5, color='#000000')
            cr_concentrations_axis.grid(True, which='minor', axis='y', linewidth=0.5, color='#FFFFFF')
            
            cr_concentrations_plot_lines = cr_concentrations_axis.plot(
                cr_trends_dates_clean, 
                cr_concentrations_clean, 
                marker='o', 
                linestyle='-', 
                linewidth=1, 
                color='#2E8B57', 
                markersize=4,
                markerfacecolor='#9DC2AD', 
                markeredgewidth=0.75,
                markeredgecolor='#2E8B57',
                label='Observed Concentration'
            )

            cr_river_stage_axis = cr_concentrations_axis.twinx()
            cr_river_stage_axis.set_ylabel('River Stage (m amsl)')
            min_stage = 2 * np.floor((cr_river_stages.min(skipna=True) - 1) / 2) + 1
            max_stage = 2 * np.ceil((cr_river_stages.max(skipna=True) - 1) / 2) + 1
            ticks = np.arange(min_stage, max_stage + 2, 2)

            cr_river_stage_axis.set_yticks(ticks)
            cr_river_stage_axis.set_ylim(min_stage - 0.25, max_stage + 0.25)
            cr_river_stage_axis.plot(cr_trends_dates, cr_river_stages, linewidth=0.75, color='#97C4EF', label='River Stage')

            cr_trends_dates_clean_numeric = (cr_trends_dates_clean - cr_trends_dates_clean.min()).dt.days / 365.25
            cr_trends_dates_numeric = (cr_trends_dates - cr_trends_dates.min()).dt.days / 365.25

            cr_concentrations_clean_log = np.log1p(cr_concentrations_clean)
            
            coeffs = np.polyfit(cr_trends_dates_clean_numeric, cr_concentrations_clean_log, deg=1)
            mock_trend = np.poly1d(coeffs)

            calculated_concentrations_trend = np.expm1(mock_trend(cr_trends_dates_numeric))
            calculated_concentrations_trend = np.maximum(calculated_concentrations_trend, 0)

            cr_concentrations_axis2 = page_fig.add_subplot(grid_spec[4, :])
            cr_concentrations_axis2.set_facecolor('#E5E5E5')

            cr_concentrations_axis2.xaxis.set_major_locator(YearLocator())
            cr_concentrations_axis2.xaxis.set_major_formatter(DateFormatter('%Y'))
            cr_concentrations_axis2.tick_params(axis='x', labelrotation=90)

            cr_concentrations_axis2.set_yscale('log')
            cr_concentrations_axis2.yaxis.set_major_locator(LogLocator(base=10))
            cr_concentrations_axis2.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            cr_concentrations_axis2.yaxis.set_minor_formatter(NullFormatter())
            ymin = 10 ** np.floor(np.log10(cr_min))
            ymax = 10 **np.ceil(np.log10(cr_max))
            cr_concentrations_axis2.set_ylim(ymin - 0.1, ymax + 1)
            cr_concentrations_axis2.set_ylabel('Hex. & Filt. Cr (µg/L)')

            cr_concentrations_axis2.plot(
                cr_trends_dates_clean, 
                cr_concentrations_clean, 
                marker='o', 
                linestyle='-', 
                linewidth=1, 
                color='#2E8B57', 
                markersize=4,
                markerfacecolor='#9DC2AD', 
                markeredgewidth=0.75,
                markeredgecolor='#2E8B57',
            )
            calculated_concentrations_plot_lines = cr_concentrations_axis2.plot(
                cr_trends_dates, 
                calculated_concentrations_trend, 
                linestyle='-', 
                color='#E9B831', 
                label='Calculated Conc.'
            )
            cr_concentrations_axis2.grid(True, which='major', axis='x', linewidth=0.5, color='#FFFFFF')
            cr_concentrations_axis2.grid(True, which='major', axis='y', linewidth=0.5, color='#000000')
            cr_concentrations_axis2.grid(True, which='minor', axis='y', linewidth=0.5, color='#FFFFFF')
            
            wl_elevation_handles, wl_elevation_labels = wl_elevation_axis.get_legend_handles_labels()
            wl_river_stage_handles, wl_river_stage_labels = wl_river_stage_axis.get_legend_handles_labels()
            cr_concentrations_handles, cr_concentrations_labels = cr_concentrations_axis.get_legend_handles_labels()
            calculated_concentrations_handles, calculated_concentrations_labels = cr_concentrations_axis2.get_legend_handles_labels()

            all_handles = wl_elevation_handles + wl_river_stage_handles + cr_concentrations_handles + calculated_concentrations_handles
            all_labels = wl_elevation_labels + wl_river_stage_labels + cr_concentrations_labels + calculated_concentrations_labels

            legend_axis = page_fig.add_subplot(grid_spec[1, 3])
            legend_axis.axis('off')
            legend_axis.legend(all_handles, all_labels, loc='center', frameon=False)
    
            page_fig.subplots_adjust(left = 1.25 / 8.5, right = 1 - 1.5 / 11)
            pdf.savefig(page_fig)
            plt.close(page_fig)

        pdf.close()
