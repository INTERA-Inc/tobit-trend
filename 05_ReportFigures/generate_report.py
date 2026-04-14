import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, LogLocator, LogFormatterSciNotation, FuncFormatter, MultipleLocator, NullFormatter
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.backends.backend_pdf import PdfPages

# Import well information data
ifile_wells = pd.read_csv('input/Cr_TrendData_unique_wells.csv')

ifile_no_riv_stage = pd.read_csv('input/NoRS.csv')
print("No River Stage\n", ifile_no_riv_stage.head(), "\n")

cr_trends_wl_lag = pd.read_csv('input/CrTrends_WLLAG_flat.csv')
wl_trends = pd.read_csv('input/WLTrends_flat.csv')
print("Cr Trends WL Lag\n", cr_trends_wl_lag.head(), "\n")
print("WL Trends\n", wl_trends.head(), "\n")

cr_trends_parquet = pd.read_parquet('input/Cr_TrendData.parquet')
print("Cr Trends Parquet\n", cr_trends_parquet[cr_trends_parquet['VAL'].notna()], "\n")

wl_rs_parquet = pd.read_parquet('input/WL_RS.parquet')
print("WL RS Parquet\n", wl_rs_parquet.head(), "\n")

ifile_gis_highriv = gpd.read_file('input/gis_export/HIGHRIV.shp')
ifile_gis_roads = gpd.read_file('input/gis_export/ROADS.shp')
ifile_gis_ous = gpd.read_file('input/gis_export/OU.shp')
ifile_gis_wells = gpd.GeoDataFrame(ifile_wells, geometry=gpd.points_from_xy(ifile_wells['XCOORDS'], ifile_wells['YCOORDS']), crs='EPSG:2926')

prepped_chem_rs_parquet = pd.read_parquet('input/R_prepped_chem_rs.parquet')
print("Prepped Chem RS Parquet\n", prepped_chem_rs_parquet[prepped_chem_rs_parquet['VAL'].notna()], "\n")

coords = ifile_wells[['NAME', 'XCOORDS', 'YCOORDS']]
print("Coordinates\n", coords.head(), "\n")

coords = coords[coords['NAME'].isin(cr_trends_wl_lag['KEY'].unique()) | coords['NAME'].isin(wl_trends["KEY"].unique())]
print("Filtered Coordinates\n", coords.head(), "\n")

CUTOFF = pd.DataFrame({
    'SYSTEM': ['DX', 'HX', 'KX_KW_KR4'],
    'EVENT': pd.to_datetime(['2011-01-01', '2011-11-01', '2009-04-01'])
})

col = ['rgb(46,139,87,255)', 'rgb(117,107,177,255)', 'rgb(77,77,77,255)', 'rgb(255,165,0,255)', 'rgb(0,0,205,255)']
colline = col
pch = [21, 21, 21, 21, 21]
bg = ['rgb(46,139,87,100)', 'rgb(117,107,177,100)', 'rgb(77,77,77,100)', 'rgb(255,165,0,100)', 'rgb(0,0,205,100)']
OUs = ['100-HR-3-D', '100-HR-3-H', '100-KR-4']

plt.rcParams['font.family'] = 'Arial'

for ou in OUs:
    # area = ifile_wells[ifile_wells['OU'] == ou][['NAME', 'OU']]
    # print(f"Area {ou}\n", area.head(), "\n")
    # wells = pd.DataFrame(sorted(area['NAME'].unique()))
    # print(f"Wells in {ou}\n", wells, "\n")

    gis_wells = set(ifile_gis_wells['NAME'])
    wl_wells = set(wl_rs_parquet['NAME'])
    cr_wells = set(prepped_chem_rs_parquet['NAME'])

    valid_wells = gis_wells & wl_wells & cr_wells

    wells = ifile_wells[ifile_wells["NAME"].isin(valid_wells)]

    num_wells = len(wells)

    gis_roads = ifile_gis_roads.to_crs(ifile_gis_highriv.crs)
    ifile_gis_ous = ifile_gis_ous.to_crs(ifile_gis_highriv.crs)

    fn = f"output/TobitRegression_WLlag - {ou}_CY2026_v4_041526.pdf"
    with PdfPages(fn) as pdf:
        for i in range(num_wells):
            gis_well = ifile_gis_wells[ifile_gis_wells['NAME'] == wells.iloc[i, 0]]
            print(f"GIS well for {wells.iloc[i, 0]} in {ou}\n", gis_well, "\n")
            wl_trends_well = wl_rs_parquet[wl_rs_parquet['NAME'] == wells.iloc[i, 0]]
            print(f"WL trends for well {wells.iloc[i, 0]} in {ou}\n", wl_trends_well.head(), "\n")
            cr_trends_well = prepped_chem_rs_parquet[prepped_chem_rs_parquet['NAME'] == wells.iloc[i, 0]]
            print(f"Cr trends for well {wells.iloc[i, 0]} in {ou}\n", cr_trends_well.head(), "\n")

            gis_ou = ifile_gis_ous[ifile_gis_ous['Name'] == ou]
            print("GIS OU HEAD\n", gis_ou.head(), "\n")

            gis_ou = gis_ou.to_crs(ifile_gis_highriv.crs)

            page_fig = plt.figure(figsize=(8.5, 11))
            grid_spec = GridSpec(nrows=4, ncols=2, wspace=0.1, hspace=0.1)

            page_fig.text(0.5, 0.95, f"{wells.iloc[i, 0]}", ha='center', va='top', fontsize=20, fontstyle='italic', fontweight='light')
            page_fig.text(
                0.5, 
                0.9, 
f"""Distance to river: {round(gis_well.at[gis_well.index[0], 'DIST'])} m
Number of Trends Calculated: 
""", 
                ha='center',
                va='top',
                fontsize=10,
                fontstyle='italic'
            )

            ax = page_fig.add_subplot(grid_spec[0, 0])

            gis_roads.plot(ax=ax, color='#E5E5E5', linewidth=1, markersize=50, zorder=1)
            ifile_gis_highriv.plot(ax=ax, color='#E5E5E5', edgecolor='black', linewidth=1, zorder=2)
            gis_ou.plot(ax=ax, color='none', edgecolor='#006400', linewidth=1, zorder=3)
            ifile_gis_wells.plot(ax=ax, color='#B3B3B3', edgecolor='#7F7F7F', markersize=4, linewidth=0.5, zorder=4)
            gis_well.plot(ax=ax, color='#FF0000', marker='o', edgecolor='#000000', markersize=12, linewidth=0.75, zorder=5)
            
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim([561500, 586000])
            ax.set_ylim([145000, 155000])
            # plt.show()

            wl_elevations = wl_trends_well['WLE']
            wl_river_stages = wl_trends_well['INTERP']
            wl_trends_dates = pd.to_datetime(wl_trends_well['EVENT'])

            wl_elevations_clean = wl_elevations[~np.isnan(wl_elevations)]
            wl_trends_dates_clean = wl_trends_dates[~np.isnan(wl_elevations)]

            # wl_rs_fig, wl_elevation_axis = plt.subplots(figsize=(15, 3))
            wl_elevation_axis = page_fig.add_subplot(grid_spec[1, :])
            wl_elevation_axis.set_facecolor('#E5E5E5')
            wl_elevation_axis.patch.set_alpha(0.5)
            wl_elevation_axis.grid(True, linewidth=1, color='#FFFFFF')
            wl_elevation_axis.set_xticklabels([])
            wl_elevation_axis.set_ylim(wl_elevations.min(skipna=True) - 2, wl_elevations.max(skipna=True) + 2)
            wl_elevation_axis.yaxis.set_major_locator(MultipleLocator(1))
            wl_elevation_axis.set_ylabel('Water-Level (m amsl)')
            wl_elevation_axis.plot(
                wl_trends_dates_clean, 
                wl_elevations_clean, 
                marker='o', 
                linewidth=1, 
                linestyle='-', 
                color='#050607', 
                markerfacecolor='#FFFFFF', 
                markeredgewidth=2, 
                markeredgecolor='#050607'
            )

            wl_river_stage_axis = wl_elevation_axis.twinx()
            min_stage = 2 * np.floor((wl_river_stages.min(skipna=True) - 1) / 2) + 1
            max_stage = 2 * np.ceil((wl_river_stages.max(skipna=True) - 1) / 2) + 1
            ticks = np.arange(min_stage, max_stage + 2, 2)
            wl_river_stage_axis.set_yticks(ticks)
            wl_river_stage_axis.set_ylim(min_stage - 0.25, max_stage + 0.25)
            wl_river_stage_axis.set_ylabel('River Stage (m amsl)')
            wl_river_stage_axis.plot(wl_trends_dates, wl_river_stages, color='#97C4EF')

            # plt.show()

            cr_trends_dates = pd.to_datetime(cr_trends_well['EVENT'])
            cr_concentrations = cr_trends_well['VAL']
            cr_trends_dates_clean = cr_trends_dates[~np.isnan(cr_concentrations)]
            cr_concentrations_clean = cr_concentrations[~np.isnan(cr_concentrations)]
            cr_river_stages = cr_trends_well['INTERP']
            print(f"Processing well {wells.iloc[i, 0]} in {ou}\n")

            cr_concentrations_axis = page_fig.add_subplot(grid_spec[2, :])
            # cr_rs_fig, cr_concentrations_axis = plt.subplots(figsize=(15, 3))
            cr_concentrations_axis.set_facecolor('#E5E5E5')
            cr_concentrations_axis.patch.set_alpha(0.5)
            cr_min = cr_concentrations_clean.min(skipna=True)
            cr_max = cr_concentrations_clean.max(skipna=True)
            
            cr_concentrations_axis.xaxis.set_major_locator(YearLocator())
            cr_concentrations_axis.xaxis.set_major_formatter(DateFormatter('%Y'))
            cr_concentrations_axis.tick_params(axis='x', labelrotation=90)
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

            cr_concentrations_axis.grid(True, which='major', axis='x', linewidth=1, color='#FFFFFF')
            cr_concentrations_axis.grid(True, which='major', axis='y', linewidth=1, color='#000000')
            cr_concentrations_axis.grid(True, which='minor', axis='y', linewidth=1, color='#FFFFFF')
            
            cr_concentrations_axis.plot(
                cr_trends_dates_clean, 
                cr_concentrations_clean, 
                marker='o', 
                linestyle='-', 
                linewidth=1, 
                color='#2E8B57', 
                markerfacecolor='#9DC2AD', 
                markeredgewidth=2,
                markeredgecolor='#2E8B57',
                zorder=3
            )

            cr_river_stage_axis = cr_concentrations_axis.twinx()
            cr_river_stage_axis.set_ylabel('River Stage (m amsl)')
            min_stage = 2 * np.floor((cr_river_stages.min(skipna=True) - 1) / 2) + 1
            max_stage = 2 * np.ceil((cr_river_stages.max(skipna=True) - 1) / 2) + 1
            ticks = np.arange(min_stage, max_stage + 2, 2)

            cr_river_stage_axis.set_yticks(ticks)
            cr_river_stage_axis.set_ylim(min_stage - 0.25, max_stage + 0.25)
            cr_river_stage_axis.plot(cr_trends_dates, cr_river_stages, color='#97C4EF')

            # plt.show()
    
            pdf.savefig(page_fig)
            plt.close(page_fig)

        pdf.close()