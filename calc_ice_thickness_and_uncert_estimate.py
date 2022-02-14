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

is2_data = xr.open_dataset('gridded_freeboard_2019-03.nc')

nesosim_data_path = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF1.7284668037515452e-06_WPT5_LLF1.2174787315012357e-07-100kmv112par_oib_averaged_final_5k/final/NESOSIMv11_01092018-30042019.nc'
nesosim_uncert_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_averagedfinal_5k_2018_2019_cov/averagedfinal_5k_2018_2019_covuncert_100_iter_final.nc'


nesosim_data = xr.open_dataset(nesosim_data_path)
nesosim_uncertainty = xr.open_dataset(nesosim_uncert_path)


days = pd.to_datetime(nesosim_data['day'].values,format='%Y%m%d')

nesosim_data['day'] = days
nesosim_uncertainty['day'] = days

# select corresponding month & calculate monthly mean
print(nesosim_data)

print(nesosim_data.sel(day='2019-03'))

nesosim_data_monthly = nesosim_data.sel(day="2019-03")
nesosim_uncert_monthly = nesosim_uncertainty.sel(day="2019-03")


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

random_uncert = inverse_r_w_minus_r_i*r_w*e_h_f**2 + (e_h_s*inverse_r_w_minus_r_i*(r_s-r_w))**2 + (e_r_s*h_s*inverse_r_w_minus_r_i)**2 + ((h_f*r_w + h_s*r_s - h_s*r_w)*e_r_i*inverse_r_w_minus_r_i**2)**2

plt.figure()
plt.imshow(sea_ice_thickness,origin='lower',vmin=0,vmax=5)
plt.colorbar()
plt.savefig('sea_ice_thickness_estimate.png')

plt.figure()
plt.imshow(random_uncert,origin='lower',vmin=0,vmax=0.2)
plt.colorbar()
plt.savefig('sea_ice_thickness_uncert.png')