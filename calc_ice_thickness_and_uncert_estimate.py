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


# which plots to make (to avoid excessive re-running)
MAKE_MAP_PLOTS = True# plot maps of uncertainty for the month
MAKE_UNCERT_CORREL_PLOTS = True# plot correlation plots of the uncertainties
MAKE_SIT_CORREL_PLOTS = True # plot correlation between nesosim-mcmc and regridded is2 product sit


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
#print(nesosim_data)

#print(nesosim_data.sel(day=monthday))

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

# uncertainties as per Petty et al 2020, for comparison
e_h_s_previous = 0.2*h_f + 0.01 # based on fit; is this valid in this case?
e_r_s_previous = 40 # kg/m^3, based on Warren et al

# it's not random uncertainty but systematic uncertainty actually? double-check article


# 1/(r_w-r_s) because this term shows up a lot
inverse_r_w_minus_r_i = 1/(r_w - r_i)



sea_ice_thickness = h_f*r_w*inverse_r_w_minus_r_i + h_s*(r_s-r_w)*inverse_r_w_minus_r_i

# random uncertainty only including snow-related terms (no ice-related terms)
random_uncert_snow_only = (e_h_s*inverse_r_w_minus_r_i*(r_s-r_w))**2 + (e_r_s*h_s*inverse_r_w_minus_r_i)**2 
random_uncert_snow_only = np.sqrt(random_uncert_snow_only)

# bit redundant (shares terms with snow-only above) but I'll keep this separate for clarity
random_uncert = inverse_r_w_minus_r_i*r_w*e_h_f**2 + (e_h_s*inverse_r_w_minus_r_i*(r_s-r_w))**2 + (e_r_s*h_s*inverse_r_w_minus_r_i)**2 + ((h_f*r_w + h_s*r_s - h_s*r_w)*e_r_i*inverse_r_w_minus_r_i**2)**2

random_uncert = np.sqrt(random_uncert)

# uncertainties as per Petty et al 2020, for comparison
uncert_previous = inverse_r_w_minus_r_i*r_w*e_h_f**2 + (e_h_s_previous*inverse_r_w_minus_r_i*(r_s-r_w))**2 + (e_r_s_previous*h_s*inverse_r_w_minus_r_i)**2 + ((h_f*r_w + h_s*r_s - h_s*r_w)*e_r_i*inverse_r_w_minus_r_i**2)**2
uncert_previous = np.sqrt(uncert_previous)

# create nice maps for the plots
# todo: make maps nicer (need to fix axis labels etc, make maps circular maybe?)

if MAKE_MAP_PLOTS:

	# should probably make a plotting function? lots of redundancy here
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



	var = random_uncert_snow_only
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


	var = uncert_previous
	fig=plt.figure(dpi=200)
	ax = plt.axes(projection = proj)
	pcm = ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat',vmin=0, vmax=0.7) # using flat shading avoids artefacts
	ax.coastlines(zorder=3)
	ax.gridlines(draw_labels=True,
	          linewidth=0.22, color='gray', alpha=0.5, linestyle='--')

	# for some reason this extent complains if you set set -180 to +180
	ax.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())

	plt.colorbar(pcm)
	plt.title('Sea ice thickness uncertainty (previous estimate) for {} (m)'.format(monthday))
	plt.savefig('sea_ice_thickness_uncert_prev_{}_{}.png'.format(DATA_FLAG, monthday))


# next step: compare with ensemble

# load ensemble output



ens_data_flag = '{}_ensemble_uncert'.format(DATA_FLAG)


if MAKE_SIT_CORREL_PLOTS:

	sit_is2 = xr.open_dataset('gridded_sit_{}.nc'.format(monthday))['sit'][0,:,:]

	sit_uncert_is2 = xr.open_dataset('gridded_sit_{}.nc'.format(monthday))['sit uncertainty'][0,:,:]

	print(sit_uncert_is2)

	# gridded_sit_2019-03.nc
	# correlate sit and uncertainty plots

	mask1 = ~np.isnan(sit_is2.values) & ~np.isnan(sea_ice_thickness)

	mask2 = ~np.isnan(sit_uncert_is2.values) & ~np.isnan(random_uncert)
	mask3 = ~np.isnan(sit_uncert_is2.values) & ~np.isnan(uncert_previous)

	nbins = 20
	plt.figure(dpi=200)

	plt.hist2d(sit_is2.values[mask1].flatten(), sea_ice_thickness[mask1].flatten(),bins=nbins)
	plt.title('SIT comparison for {} (m)'.format(monthday))
	plt.xlabel('IS2SITMOGR4')
	plt.ylabel('NESOSIM-MCMC SIT')
	plt.colorbar()
	plt.savefig('hist_is2_vs_mcmc_sit_{}_{}.png'.format(DATA_FLAG, monthday))

	#plt.figure(dpi=200)
	#plt.imshow(sit_uncert_is2,vmin=0,vmax=1)
	#plt.title('ISTSITMOGR4 uncertainty')
	#plt.colorbar()
	#plt.savefig('is2_uncert.png')
	
	#plt.figure(dpi=200)
	#plt.imshow(random_uncert,vmin=0,vmax=1)
	#plt.title('NESOSIM-MCMC uncert')
	#plt.colorbar()
	#plt.savefig('nesosim-mcmc-uncert.png')
	
	plt.figure(dpi=200)
	uncert_diff = sit_uncert_is2.values - random_uncert
	plt.imshow(uncert_diff,vmin=-1,vmax=1,cmap='RdBu')
	plt.title('uncertainty difference')
	plt.colorbar()
	plt.savefig('mcmc-is2-uncert-diff_{}_{}.png'.format(DATA_FLAG, monthday))

	plt.figure(dpi=200)

	# there's some extreme values of uncertainties = 14 that need to be excluded;
	# try explicit range argument?
	# could also mask but maybe do that later

	plt.hist2d(sit_uncert_is2.values[mask2].flatten(), random_uncert[mask2].flatten(),bins=nbins, range=[[0,1.2],[0,1.2]])
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('IS2SITMOGR4 uncert')
	plt.ylabel('NESOSIM-MCMC SIT uncert')
	plt.colorbar()
	plt.savefig('hist_is2_vs_mcmc_sit_uncert_{}_{}.png'.format(DATA_FLAG, monthday))


	plt.figure(dpi=200)

	plt.hist2d(sit_uncert_is2.values[mask3].flatten(), uncert_previous[mask3].flatten(),bins=nbins, range=[[0,1.2],[0,1.2]])
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('IS2SITMOGR4 uncert')
	plt.ylabel('NESOSIM-MCMC SIT uncert P2020')
	plt.colorbar()
	plt.savefig('hist_is2_vs_mcmc_p2020_sit_uncert_{}_{}.png'.format(DATA_FLAG, monthday))

	# plt.figure(dpi=200)
	# plt.scatter(sit_uncert_is2.values.flatten(),random_uncert.flatten(),alpha=0.8)
	# plt.xlim(0,1)
	# plt.xlabel('IS2SITMOGR4 uncert')
	# plt.ylabel('NESOSIM-MCMC SIT uncert')
	# plt.savefig('scatter-nesosim-mcmc-vs-is2-uncert_{}_{}.png'.format(DATA_FLAG, monthday)



if MAKE_UNCERT_CORREL_PLOTS:
	# can make the nicer seaborn plots later maybe
	# for now just make regular histograms

	# random_uncert, random_uncert_snow_only, ens_uncert

	# want 3 histograms: 

	ens_uncert = xr.open_dataarray('sit_uncert_ensemble_{}.nc'.format(ens_data_flag))

	# mask out nan; see if this works (need to be consistent between arrays)
	mask1 = ~np.isnan(ens_uncert.values) & ~np.isnan(random_uncert)
	mask2 = ~np.isnan(ens_uncert.values) & ~np.isnan(random_uncert_snow_only)
	mask3 = ~np.isnan(random_uncert) & ~np.isnan(random_uncert_snow_only)
	mask4 = ~np.isnan(uncert_previous) & ~np.isnan(ens_uncert.values)
	mask5 = ~np.isnan(uncert_previous) & ~np.isnan(random_uncert)
	mask6 = ~np.isnan(uncert_previous) & ~np.isnan(random_uncert_snow_only)



	# should maybe just make functions for this?

	nbins = 20
	plt.figure(dpi=200)

	plt.hist2d(ens_uncert.values[mask1].flatten(), random_uncert[mask1].flatten(),bins=nbins)
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('Ensemble uncertainty')
	plt.ylabel('Total random uncertainty')
	plt.colorbar()
	plt.savefig('hist_ensemble_vs_total_random_{}_{}.png'.format(DATA_FLAG, monthday))


	plt.figure(dpi=200)

	plt.hist2d(ens_uncert.values[mask2].flatten(), random_uncert_snow_only[mask2].flatten(),bins=nbins)
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))

	plt.xlabel('Ensemble uncertainty')
	plt.ylabel('Snow-only random uncertainty')
	plt.colorbar()
	plt.savefig('hist_ensemble_vs_random_snow_{}_{}.png'.format(DATA_FLAG, monthday))

	plt.figure(dpi=200)

	plt.hist2d(random_uncert[mask3].flatten(), random_uncert_snow_only[mask3].flatten(),bins=nbins)
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('Total random uncertainty')
	plt.ylabel('Snow-only random uncertainty')
	plt.colorbar()
	plt.savefig('hist_random_snow_vs_total_random_{}_{}.png'.format(DATA_FLAG, monthday))
	plt.figure(dpi=200)

	plt.hist2d(uncert_previous[mask4].flatten(), ens_uncert.values[mask4].flatten(),bins=nbins)
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('Previous uncertainty')
	plt.ylabel('Ensemble uncertainty')
	plt.colorbar()
	plt.savefig('hist_previous_vs_ensemble_{}_{}.png'.format(DATA_FLAG, monthday))

	plt.figure(dpi=200)
	plt.hist2d(uncert_previous[mask5].flatten(), random_uncert[mask5].flatten(),bins=nbins)
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('Previous uncertainty')
	plt.ylabel('Total uncertainty')
	plt.colorbar()
	plt.savefig('hist_previous_vs_total_{}_{}.png'.format(DATA_FLAG, monthday))
	plt.figure(dpi=200)

	plt.hist2d(uncert_previous[mask6].flatten(), random_uncert_snow_only[mask6].flatten(),bins=nbins)
	plt.title('SIT uncertainty comparison for {} (m)'.format(monthday))
	plt.xlabel('Previous uncertainty')
	plt.ylabel('Snow-only uncertainty')
	plt.colorbar()
	plt.savefig('hist_previous_vs_snow_only_{}_{}.png'.format(DATA_FLAG, monthday))

