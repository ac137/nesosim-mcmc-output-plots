# calculate sea ice thickness and uncertainty estimate from nesosim output 
# & regridded ICESat-2 freeboard

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import cartopy.crs as ccrs

# estimate based on retrieval in Petty et al 2020

# monthday='2018-11'
# monthday='2019-01'
monthday='2019-03'
# monthday='2019-04'

is2_data = xr.open_dataset('gridded_freeboard_{}.nc'.format(monthday))


#DATA_FLAG = 'oib_averaged'
DATA_FLAG = 'oib_detailed'

if DATA_FLAG == 'oib_averaged':
	# oib averaged
	# this contains the entire 2018-2019 year; if I do another year then I'll have to change this!
	nesosim_data_path = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF1.7284668037515452e-06_WPT5_LLF1.2174787315012357e-07-100kmv112par_oib_averaged_final_5k/final/NESOSIMv11_01092018-30042019.nc'
	nesosim_uncert_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_averagedfinal_5k_2018_2019_cov/averagedfinal_5k_2018_2019_covuncert_100_iter_final.nc'
elif DATA_FLAG == 'oib_detailed':
	# oib detailed
	nesosim_data_path = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.0504155592128743e-06_WPT5_LLF4.0059442776163867e-07-100kmv112par_oib_detailed_final_5k/final/NESOSIMv11_01092018-30042019.nc'
	nesosim_uncert_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_detailedfinal_5k_2018_2019_cov/detailedfinal_5k_2018_2019_covuncert_100_iter_final.nc'

nesosim_data = xr.open_dataset(nesosim_data_path)
nesosim_uncertainty = xr.open_dataset(nesosim_uncert_path)


days = pd.to_datetime(nesosim_data['day'].values,format='%Y%m%d')

nesosim_data['day'] = days
nesosim_uncertainty['day'] = days

# select corresponding month & calculate monthly mean
print(nesosim_data)

print(nesosim_data.sel(day=monthday))

nesosim_data_monthly = nesosim_data.sel(day=monthday)
nesosim_uncert_monthly = nesosim_uncertainty.sel(day=monthday)


# make sure latitude and longitude are lined up!!!

# density of water
r_w =  1024 #kg/m^3
# density of ice
r_i = 915 #kg/m^3 (as assumed in Petty et al 2020)
# density of snow
r_s = nesosim_data_monthly['snow_density'].mean(axis=0).values
# snow density comes fron nesosim

# freeboard error
e_h_f = is2_data['freeboard uncertainty'].values[0,:,:]
# snow depth error
e_h_s = nesosim_uncert_monthly['snow_depth'].mean(axis=0).values
# snow density error
e_r_s = nesosim_uncert_monthly['snow_density'].mean(axis=0).values
# ice density error
e_r_i = 10 #kg/m^3 based on Alexandrov et al. (2013)
# snow depth
h_s = nesosim_data_monthly['snow_depth'].mean(axis=0).values
# freeboard height
h_f = is2_data['freeboard'].values[0,:,:]
#

# 1/(r_w-r_s) because this term shows up a lot
inverse_r_w_minus_r_i = 1/(r_w - r_i)



sea_ice_thickness = h_f*r_w*inverse_r_w_minus_r_i + h_s*(r_s-r_w)*inverse_r_w_minus_r_i

# random uncertainty only including snow-related terms (no ice-related terms)
random_uncert_snow_only = (e_h_s*inverse_r_w_minus_r_i*(r_s-r_w))**2 + (e_r_s*h_s*inverse_r_w_minus_r_i)**2 
random_uncert_snow_only = np.sqrt(random_uncert_snow_only)

# bit redundant (shares terms with snow-only above) but I'll keep this separate for clarity
random_uncert = inverse_r_w_minus_r_i*r_w*e_h_f**2 + (e_h_s*inverse_r_w_minus_r_i*(r_s-r_w))**2 + (e_r_s*h_s*inverse_r_w_minus_r_i)**2 + ((h_f*r_w + h_s*r_s - h_s*r_w)*e_r_i*inverse_r_w_minus_r_i**2)**2

random_uncert = np.sqrt(random_uncert)


# create nice maps for the plots
# todo: make maps nicer (need to fix axis labels etc, make maps circular maybe?)


proj=ccrs.NorthPolarStereo(central_longitude=-45)
proj_coord = ccrs.PlateCarree()

lons = nesosim_data['longitude']
lats = nesosim_data['latitude']
var = sea_ice_thickness #-1 to select last day of season

fig=plt.figure(dpi=200)
ax = plt.axes(projection = proj)
pcm = ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat',vmin=0,vmax=5) # using flat shading avoids artefacts
ax.coastlines(zorder=3)
ax.gridlines(draw_labels=True,
          linewidth=0.22, color='gray', alpha=0.5, linestyle='--')

# for some reason this extent complains if you set set -180 to +180
ax.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())

plt.title('Sea ice thickness for {} (m)'.format(monthday))
plt.colorbar(pcm)
plt.savefig('sea_ice_thickness_estimate_{}_{}.png'.format(DATA_FLAG,monthday))



var = random_uncert
fig=plt.figure(dpi=200)
ax = plt.axes(projection = proj)
pcm = ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat',vmin=0, vmax=0.7) # using flat shading avoids artefacts
ax.coastlines(zorder=3)
ax.gridlines(draw_labels=True,
          linewidth=0.22, color='gray', alpha=0.5, linestyle='--')

# for some reason this extent complains if you set set -180 to +180
ax.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())

plt.colorbar(pcm)
plt.title('Sea ice thickness uncertainty for {} (m)'.format(monthday))
plt.savefig('sea_ice_thickness_uncert_{}_{}.png'.format(DATA_FLAG, monthday))



var = random_uncert
fig=plt.figure(dpi=200)
ax = plt.axes(projection = proj)
pcm = ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat',vmin=0, vmax=0.7) # using flat shading avoids artefacts
ax.coastlines(zorder=3)
ax.gridlines(draw_labels=True,
          linewidth=0.22, color='gray', alpha=0.5, linestyle='--')

# for some reason this extent complains if you set set -180 to +180
ax.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())

plt.colorbar(pcm)
plt.title('Sea ice thickness uncertainty (from snow only) for {} (m)'.format(monthday))
plt.savefig('sea_ice_thickness_uncert_snow_only_{}_{}.png'.format(DATA_FLAG, monthday))


# next step: save output separately (for making distribution plots, since will need to compare with ensemble)
