# plot nesosim output snow depth time series by layer

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import cartopy.crs as ccrs

# mean
# file_path_1 = '/home/alex/modeldev/output_with_uncert/mean_detailed_100_iter.nc'
# # uncertainty
# file_path_2 = '/home/alex/modeldev/output_with_uncert/uncert_detailed_100_iter.nc'

# file_path_1 = '/home/alex/modeldev/output_with_uncert/mean_averaged_100_iter.nc'
# file_path_2 = '/home/alex/modeldev/output_with_uncert/uncert_averaged_100_iter.nc'

# make these file paths easier to change later


# for oib detailed
file_path_1 = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF1.0424574017128326e-06_WPT5_LLF5.32239712044108e-07-100kmv113par_oib_detailed_ic/final/NESOSIMv11_01092010-30042011.nc'


budget_path_1 = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF1.0424574017128326e-06_WPT5_LLF5.32239712044108e-07-100kmv113par_oib_detailed_ic/budgets/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF1.0424574017128326e-06_WPT5_LLF5.32239712044108e-07-100kmv113par_oib_detailed_ic-01092010-30042011.nc'

# for oib averaged:

file_path_1 = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF5.77125249688052e-07_WPT5_LLF3.500788217903482e-07-100kmv113par_oib_averaged_ic/final/NESOSIMv11_01092010-30042011.nc'
budget_path_1 = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF5.77125249688052e-07_WPT5_LLF3.500788217903482e-07-100kmv113par_oib_averaged_ic/budgets/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF5.77125249688052e-07_WPT5_LLF3.500788217903482e-07-100kmv113par_oib_averaged_ic-01092010-30042011.nc'
# BE SURE TO CHANGE FILE SAVE PATH (todo: throw in an if statement?)


# 3par with ic loglike

file_path_1 = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.3450925692135826e-06_WPT5_LLF1.5380250062998322e-07-100kmv113par_oib_averaged_ic_with_ic_loglike/final/NESOSIMv11_01092010-30042011.nc'
budget_path_1 = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.3450925692135826e-06_WPT5_LLF1.5380250062998322e-07-100kmv113par_oib_averaged_ic_with_ic_loglike/budgets/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.3450925692135826e-06_WPT5_LLF1.5380250062998322e-07-100kmv113par_oib_averaged_ic_with_ic_loglike-01092010-30042011.nc'



# budget_path_2 = ''
#TODO: more descriptive variable names here
# clean this up (taken from jupyter notebook)
# set up to run on fileserver


plot_suffix = '3par_with_ic_loglike'



# data
data_1 = xr.open_dataset(file_path_1)

# data_2 = xr.open_dataset(file_path_2)

budget_1 = xr.open_dataset(budget_path_1)
# budget_2 = xr.open_dataset(budget_path_2)


ice_conc_mask = 0.5

budget_masked_1 = budget_1.where(budget_1['iceConc']>=ice_conc_mask)
# print(budget_masked_1)

# original nesosim masking: snowDepthT[np.where(iceConcT<ice_conc_mask)]=np.nan 

# time series of snow depth by layer

sd_1_ts = budget_masked_1['snowDepth'].mean(axis=(2,3))
# sd_2_ts = budget_2['snowDepth'].mean(axis=(2,3))


# time series of snow density by layer

dens_1 = data_1['snow_density'].mean(axis=(1,2))
# dens_2 = data_2['snow_density'].mean(axis=(1,2))

# density and depth time series together; layer 0 is the upper (less dense) layer
# colours for plots
c_old = '#187485'
c_new = '#2d8796'
c_old = '#5fb9c9'

ndays = len(sd_1_ts[:,0])
x = np.arange(ndays)

snow_ylim = 0.4

# plt.figure(dpi=200)
fig, ax1 = plt.subplots(dpi=200)

plt.title('Basin mean time series')
ax1.fill_between(x, sd_1_ts[:,1], sd_1_ts[:,0]+sd_1_ts[:,1],label='Layer 0',color=c_new)
ax1.fill_between(x,sd_1_ts[:,1],label='Layer 1',color=c_old)
ax1.set_ylabel('Snow depth (m)')
ax1.set_ylim(0,snow_ylim)
ax1.set_xlabel('Days since 1 Sep.')
plt.legend(loc='upper left')
d_len = len(dens_1)
x_d = np.arange(1,d_len)
ax2 = ax1.twinx()
ax2.set_ylabel('Snow density (kg/m^3)',color='C1')
ax2.plot(x_d,dens_1[1:],color='C1')
ax2.set_ylim(200,350)


# be sure to change file name here
plt.savefig('basin_mean_time_series_{}.png'.format(plot_suffix))

# plt.show()

# fig, ax1 = plt.subplots(dpi=200)

# plt.title('Basin mean time series (OIB clim)')
# ax1.fill_between(x, sd_2_ts[:,1], sd_2_ts[:,0]+sd_2_ts[:,1],label='Layer 0',color=c_new)
# ax1.fill_between(x,sd_2_ts[:,1],label='Layer 1',color=c_old)
# ax1.set_ylabel('Snow depth (m)')
# ax1.set_ylim(0,snow_ylim)
# ax1.set_xlabel('Days since 1 Sep.')


# d_len = len(dens_2)
# x_d = np.arange(1,d_len)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Snow density (kg/m^3)',color='C1')
# ax2.plot(x_d,dens_2[1:],color='C1')
# ax2.set_ylim(200,350)
# plt.show()
