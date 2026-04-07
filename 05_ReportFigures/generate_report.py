import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pyreadr

# Specify input files and parameters
yr = 2024
run_ver = 'v5_032725_check'

# Import well information data
ifile_well_info_data = pyreadr.read_r('00_Data/Well_Data/CY24/WELL.RData')
# print(ifile_well_info_data['WELL'], "\n")

# This doesn't work because the rdata is too complex for this python package
# ifile_gis_base = rdata.read_rda('00_Data/GIS/BASE.RData')

ifile_gis_base_wells = gpd.read_file('gis_export/WELLS.shp')
row = ifile_gis_base_wells.loc[0]
print(row, "\n")

gdf1 = gpd.read_file('gis_export/HIGHRIV.shp')
gdf2 = gpd.read_file('gis_export/ROADS.shp')
gdf3 = gpd.read_file('gis_export/WELLS.shp')
gdf2 = gdf2.to_crs(gdf1.crs)
gdf3 = gdf3.to_crs(gdf1.crs)
ax = gdf2.plot(color='#E5E5E5', linewidth=2, markersize=50, zorder=1)
gdf1.plot(ax=ax, color='#E5E5E5', edgecolor='black', linewidth=2.5, figsize=(10, 10), zorder=2)
gdf3.plot(ax=ax, color='#B3B3B3', edgecolor='#7F7F7F', linewidth=2, figsize=(10, 10), zorder=3)

plt.show()

ifile_gis_base_waste = gpd.read_file('gis_export/WASTE.shp')
row3 = ifile_gis_base_waste.loc[0]
# print(row3, "\n")

ifile_gis_base_gwia = gpd.read_file('gis_export/GWIA.shp')
row4 = ifile_gis_base_gwia.loc[0]
# print(row4, "\n")

ifile_sys_wells = pd.read_csv('05_Trends/Input/CY23/SystemWells.csv')
ifile_no_riv_stage = pd.read_csv('05_Trends/Input/CY23/NoRS.csv')
# print(ifile_sys_wells.head(), "\n")
# print(ifile_no_riv_stage.head())

# Tried to import trend data but the rdata here is too complex for pyreadr as well
# cr_trends_wl_lag = pyreadr.read_r(f"{new_output_dir}CrTrends_WLlag.RData")
# wl_trends = f"{new_output_dir}WLTrends.RData"