import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import pyreadr

# Import well information data
ifile_well_info_data_dict = pyreadr.read_r('./input/WELL.RData')
ifile_well_info_data = pd.DataFrame(ifile_well_info_data_dict['WELL'])
print("WELL.RData\n", ifile_well_info_data.head(), "\n")

ifile_sys_wells = pd.read_csv('input/SystemWells.csv')
ifile_no_riv_stage = pd.read_csv('input/NoRS.csv')
print("System Wells\n", ifile_sys_wells.head(), "\n")
print("No River Stage\n", ifile_no_riv_stage.head(), "\n")

cr_trends_wl_lag = pd.read_csv('input/CrTrends_WLLAG_flat.csv')
wl_trends = pd.read_csv('input/WLTrends_flat.csv')
print("Cr Trends WL Lag\n", cr_trends_wl_lag.head(), "\n")
print("WL Trends\n", wl_trends.head(), "\n")

cr_trends_parquet = pd.read_parquet('input/Cr_TrendData.parquet')
print("Cr Trends Parquet\n", cr_trends_parquet[cr_trends_parquet['VAL'].notna()], "\n")

wl_rs_parquet = pd.read_parquet('input/WL_RS.parquet')
print("WL RS Parquet\n", wl_rs_parquet.head(), "\n")

coords = ifile_well_info_data[['NAME', 'XCOORDS', 'YCOORDS', 'ZCOORDS']]
print("Coordinates\n", coords.head(), "\n")

coords = coords[coords['NAME'].isin(cr_trends_wl_lag['KEY'].unique()) | coords['NAME'].isin(wl_trends["KEY"].unique())]
print("Filtered Coordinates\n", coords.head(), "\n")

CUTOFF = pd.DataFrame({
    'SYSTEM': ['DX', 'HX', 'KX_KW_KR4'],
    'EVENT': pd.to_datetime(['2011-01-01', '2011-11-01', '2009-04-01'])
})

ifile_sys_wells = ifile_sys_wells.merge(CUTOFF, on='SYSTEM', how='left')
print("System wells merged\n", ifile_sys_wells, "\n")

col = ['rgb(46,139,87,255)', 'rgb(117,107,177,255)', 'rgb(77,77,77,255)', 'rgb(255,165,0,255)', 'rgb(0,0,205,255)']
colline = col
pch = [21, 21, 21, 21, 21]
bg = ['rgb(46,139,87,100)', 'rgb(117,107,177,100)', 'rgb(77,77,77,100)', 'rgb(255,165,0,100)', 'rgb(0,0,205,100)']
OUs = ['100-HR-3-D', '100-HR-3-H', '100-KR-4']

for ou in OUs[0:1]:  # Process only the first OU
    area = ifile_well_info_data[ifile_well_info_data['OU'] == ou][['NAME', 'OU']]
    print(f"Area {ou}\n", area.head(), "\n")
    wells = pd.DataFrame(sorted(area['NAME'].unique()))
    print(f"Wells in {ou}\n", wells, "\n")
    
    ifile_gis_highriv = gpd.read_file('input/gis_export/HIGHRIV.shp')
    ifile_gis_roads = gpd.read_file('input/gis_export/ROADS.shp')
    ifile_gis_wells = gpd.read_file('input/gis_export/WELLS.shp')

    ifile_gis_roads = ifile_gis_roads.to_crs(ifile_gis_highriv.crs)
    ifile_gis_wells = ifile_gis_wells.to_crs(ifile_gis_highriv.crs)
    row = ifile_gis_wells.iloc[0]
    print("GIS well row\n", row, "\n")

    # ax = ifile_gis_roads.plot(color='#E5E5E5', linewidth=2, markersize=50, zorder=1)
    # ifile_gis_highriv.plot(ax=ax, color='#E5E5E5', edgecolor='black', linewidth=2.5, figsize=(10, 10), zorder=2)
    # ifile_gis_wells.plot(ax=ax, color='#B3B3B3', edgecolor='#7F7F7F', linewidth=2, figsize=(10, 10), zorder=3)

    # plt.show()

    num_wells = len(wells)

    for i in range(num_wells):
        cr_trends_well = cr_trends_parquet[cr_trends_parquet['NAME'] == wells.iloc[i, 0]]
        print(f"Cr trends for well {wells.iloc[i, 0]} in {ou}\n", cr_trends_well.head(), "\n")
        cr_trends_dates = pd.to_datetime(cr_trends_well['EVENT'])
        cr_concentrations = cr_trends_well['VAL']
        cr_trends_dates_clean = cr_trends_dates[~np.isnan(cr_concentrations)]
        cr_concentrations_clean = cr_concentrations[~np.isnan(cr_concentrations)]
        cr_river_stages = cr_trends_well['INTERP']
        print(f"Processing well {wells.iloc[i, 0]} in {ou}\n")

        fig, ax1 = plt.subplots(figsize=(15, 3))
        # ax1.set_facecolor('#E5E5E5')
        # ax1.patch.set_alpha(0.5)
        ax1.set_yscale('log')
        cr_min = cr_concentrations_clean.min(skipna=True)
        cr_max = cr_concentrations_clean.max(skipna=True)
        log_cr_min = np.floor(np.log10(cr_min))
        log_cr_max = np.ceil(np.log10(cr_max))
        ticks = np.logspace(log_cr_min, log_cr_max, num=int(log_cr_max - log_cr_min) + 1)
        print("ticks: ", ticks, "\n")
        ax1.set_ylabel('Hex. & Filt. Cr (µg/L)')
        ax1.set_yticks(ticks)
        # ax1.set_yticklabels([f"{t:.0e}" for t in ticks])
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.yaxis.set_major_locator(LogLocator(base=10.0))
        ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax1.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        ax1.yaxis.set_minor_formatter(plt.NullFormatter())
        ax1.tick_params(axis='y', which='both', labelleft=True, left=True)
        # ax1.yaxis.set_ticks_position('left')
        ax1.grid(True, which='major', axis='y', linewidth=2, color='#838B8B')
        # ax1.grid(True, which='major', axis='x', linewidth=2, color='#FFFFFF')
        # ax1.grid(True, which='major', axis='y', linewidth=2, color='#000000')
        # ax1.grid(True, which='minor', axis='y', linewidth=2, color='#FFFFFF')
        ax1.plot(
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

        ax2 = ax1.twinx()
        ax2.plot(cr_trends_dates, cr_river_stages, color='#97C4EF')
        ax2.set_ylabel('River Stage (m amsl)')

        plt.show()

    # fn = f"output/TobitRegression_WLlag - {ou}_CY2024_v3_032525_2.pdf"
    # with PdfPages(fn) as pdf:
    #     for well in wells:
    #         X = cr_trends_wl_lag[cr_trends_wl_lag['KEY'] == well]
    #         print(f"Processing well {well} in {ou}\n", X.head(), "\n")
    #         print(well)
    #         if not X.empty:
    #             NM = X['NAME'].iloc[0]
    #             AN = X['ANALYTE'].iloc[0]
    #             AN_UNITS = X['CHEM_UNITS'].dropna().iloc[0] if not X['CHEM_UNITS'].dropna().empty else None
    #             OU = X['OU'].iloc[0]
    #             RS = NM not in ifile_no_riv_stage['NAME'].values
    #             WL = wl_trends[wl_trends['KEY'] == well] if well in wl_trends['KEY'].values else None
    #             LAG = X['LAG'].tolist()
    #             if AN == 'Hex. Chromium and Fil. Chromium':
    #                 AN = 'Hex. & Filt. Cr'
                # if LAG:
                #     pltReport(X, WL, coords, BASE, OU=OU, ylab=f"{AN} ({AN_UNITS})", col=col, colline=colline, pch=pch, bg=bg, cex=0.7)
                # else:
                #     pltPlaceholder(X, WL, coords, BASE, OU=OU, ylab=f"{AN} ({AN_UNITS})", col=col, colline=colline, pch=pch, bg=bg, cex=0.7)
                # pdf.savefig()
    