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



def plot_map(var, lons, lats, title, filename, sic, **kwargs):
	'''create a map plot of variable var, longitudes lons, latitudes lats
	title (for plot), to be saved to file named filename.
	sic is a 2d array of sea ice concentration (binary mask?)
	kwargs are for pcolormesh (eg. vmin, vmax, cmap, etc.)
	'''
	proj=ccrs.NorthPolarStereo(central_longitude=-45)
	proj_coord = ccrs.PlateCarree()


	fig=plt.figure(dpi=200)
	ax = plt.axes(projection = proj)
	# sic mask; factor of 0.3 to adjust shading colour
	ax.pcolormesh(lons,lats,sic*0.3, transform=proj_coord, shading='flat', cmap="Greys",vmin=0,vmax=1, label='SIC >= 0.5')
	pcm = ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat',**kwargs) # using flat shading avoids artefacts
	ax.coastlines(zorder=3)
	ax.gridlines(draw_labels=True,
	          linewidth=0.22, color='gray', alpha=0.5, linestyle='--')

	# for some reason this extent complains if you set set -180 to +180
	ax.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())

	plt.title(title)
	plt.colorbar(pcm)

	# need to manually add legend, apparently
	sic_legend_patch = matplotlib.patches.Rectangle((0, 0), 1, 1, facecolor="#CECECE")
	labels = ['SIC >= 0.5']
	plt.legend([sic_legend_patch], labels)#,
             #  loc='lower left')#, bbox_to_anchor=(0.025, -0.1), fancybox=True)
	plt.savefig(filename)


def plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename, nbins=20, **kwargs):
	''' assumes x and y are unflattened nd arrays (same shape/size)
	title, xlabel, ylabel, filename are str for plot labels
	masks out nan values
	kwargs for additional args to hist2d (eg. range)'''

	# todo: marginal axes? adjust colormap, etc.

	mask = ~np.isnan(x) & ~np.isnan(y)

	plt.figure(dpi=200)

	plt.hist2d(x[mask].flatten(), y[mask].flatten(),bins=nbins, **kwargs)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.colorbar()
	plt.savefig(filename)


# estimate based on retrieval in Petty et al 2020

# monthday='2018-11'
# monthday='2019-01'
monthday='2019-03'
# monthday='2019-04'

is2_data = xr.open_dataset('gridded_freeboard_{}.nc'.format(monthday))


DATA_FLAG = 'oib_averaged'
#DATA_FLAG = 'oib_detailed'

FIG_PATH = 'Figures/'

# which plots to make (to avoid excessive re-running)
MAKE_MAP_PLOTS = True# plot maps of uncertainty for the month
MAKE_SIT_CORREL_PLOTS = False# plot correlation between nesosim-mcmc and regridded is2 product sit
MAKE_UNCERT_CORREL_PLOTS = False# plot correlation plots of the uncertainties


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
nesosim_sic = nesosim_data_monthly['ice_concentration'].mean(axis=0).values

# create a binary sic mask
SIC_THRESHOLD = 0.5
ice_mask_idx = (nesosim_sic >= SIC_THRESHOLD) & ~np.isnan(nesosim_sic) # true if we want to use sic
ice_mask_idx = ice_mask_idx.astype(float) # 1 if sic is >= threshold 
print(ice_mask_idx)

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
	print('plotting sit and uncertainty map plots')

	# should probably make a plotting function? lots of redundancy here
	# proj=ccrs.NorthPolarStereo(central_longitude=-45)
	# proj_coord = ccrs.PlateCarree()

	lons = nesosim_data['longitude'] # same lat and lon used everywhere I think
	lats = nesosim_data['latitude']


	# I don't have to assign variables each time (could just plug into function) 
	# but this is more readable for me and easier to change

	# sea ice thickness from NESOSIM-mcmc
	var = sea_ice_thickness 
	title = 'Sea ice thickness for {} (m)'.format(monthday)
	filename = '{}sea_ice_thickness_estimate_{}_{}.png'.format(FIG_PATH,DATA_FLAG,monthday)
	plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=5)


	# sea ice thickness unertainty from NESOSIM-mcmc using uncertainty formula
	var = random_uncert
	title = 'Sea ice thickness uncertainty for {} (m)'.format(monthday)
	filename = '{}sea_ice_thickness_uncert_{}_{}.png'.format(FIG_PATH,DATA_FLAG, monthday)
	plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=0.7)

	# 'snow only' sea ice thickness uncertainty (exclude contribution from other terms)
	var = random_uncert_snow_only
	title = 'Sea ice thickness uncertainty (from snow only) for {} (m)'.format(monthday)
	filename = '{}sea_ice_thickness_uncert_snow_only_{}_{}.png'.format(FIG_PATH,DATA_FLAG, monthday)
	plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=0.7)


	# uncertainty estimate using values from P2020 paper for snow uncert
	var = uncert_previous
	title = 'Sea ice thickness uncertainty (P2020 estimate) for {} (m)'.format(monthday)
	filename = '{}sea_ice_thickness_uncert_p2020_{}_{}.png'.format(FIG_PATH,DATA_FLAG, monthday)
	plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=0.7)


# next step: compare with ensemble

# load ensemble output



ens_data_flag = '{}_ensemble_uncert'.format(DATA_FLAG)


if MAKE_SIT_CORREL_PLOTS:
	print('plotting is2 correlation plots')

	# load gridded is2 plots (provided by gridIS2thickness.py)
	sit_is2 = xr.open_dataset('gridded_sit_{}.nc'.format(monthday))['sit'][0,:,:]

	sit_uncert_is2 = xr.open_dataset('gridded_sit_{}.nc'.format(monthday))['sit uncertainty'][0,:,:]

	# gridded_sit_2019-03.nc
	# correlate sit and uncertainty plots

	x, y = sit_is2.values, sea_ice_thickness
	title = 'SIT comparison for {} (m)'.format(monthday)
	xlabel = 'IS2SITMOGR4'
	ylabel = 'NESOSIM-MCMC SIT'
	
	filename = '{}hist_is2_vs_mcmc_sit_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)


	# difference between is2 sit and nesosim-mcmc sit
	var = sit_is2.values - sea_ice_thickness # sit value difference
	lons = nesosim_data['longitude'] # same lat and lon used everywhere 
	lats = nesosim_data['latitude']
	title = 'IS2 - NESOSIM-MCMC SIT difference'
	filename = '{}mcmc-is2-diff-map_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)
	plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=-4, vmax=4, cmap='RdBu')


	# difference between is2 uncertainty and nesosim-mcmc uncertainty
	var = sit_uncert_is2.values - random_uncert # uncertainty value difference
	lons = nesosim_data['longitude'] 
	lats = nesosim_data['latitude']
	title = 'IS2 - NESOSIM-MCMC SIT uncert difference'
	filename = '{}mcmc-is2-uncert-diff-map_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=-1,vmax=1,cmap='RdBu')


	# nesosim uncertainty vs is2 uncertainty
	x, y = sit_uncert_is2.values, random_uncert
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'IS2SITMOGR4 uncert'
	ylabel = 'NESOSIM-MCMC SIT uncert'
	
	filename = '{}hist_is2_vs_mcmc_sit_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename, range=[[0,1.2],[0,1.2]])

	# nesosim uncertainty calculated using p2020 vs. is2 uncertainty
	x, y = sit_uncert_is2.values, uncert_previous
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'IS2SITMOGR4 uncert'
	ylabel = 'NESOSIM-MCMC SIT uncert P2020'
	
	filename = '{}hist_is2_vs_mcmc_p2020_sit_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename, range=[[0,1.2],[0,1.2]])




if MAKE_UNCERT_CORREL_PLOTS:

	print('plotting nesosim uncertainty correlation plots')

	# load calculated ensemble uncertainty (provided by calc_ice_thickness_uncert_ensemble.py)
	ens_uncert = xr.open_dataarray('sit_uncert_ensemble_{}.nc'.format(ens_data_flag))


	# total vs ensemble uncertainty
	x, y = ens_uncert.values, random_uncert
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'Ensemble uncertainty'
	ylabel = 'Total uncertainty'
	filename = '{}hist_ensemble_vs_total_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)

	# snow-only vs ensemble uncertainty
	x, y = ens_uncert.values, random_uncert_snow_only
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'Ensemble uncertainty'
	ylabel = 'Snow-only uncertainty'
	filename = '{}hist_ensemble_vs_snow_only_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)


	# snow-only vs total uncertainty
	x, y = random_uncert, random_uncert_snow_only
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'Total uncertainty'
	ylabel = 'Snow-only uncertainty'
	filename = '{}hist_snow_only_vs_total_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)

	# ensemble vs. p2020 uncertainty
	x, y = uncert_previous, ens_uncert.values
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'P2020 uncertainty'
	ylabel = 'Ensemble uncertainty'
	filename = '{}hist_p2020_vs_ensemble_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)

	# total vs p2020 uncertainty
	x, y = uncert_previous, random_uncert
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'P2020 uncertainty'
	ylabel = 'Total uncertainty'
	filename = '{}hist_p2020_vs_total_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)

	# snow-only vs p2020 uncertainty
	x, y = uncert_previous, random_uncert_snow_only
	title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
	xlabel = 'P2020 uncertainty'
	ylabel = 'Snow-only uncertainty'
	filename = '{}hist_p2020_vs_snow_only_uncert_{}_{}.png'.format(FIG_PATH, DATA_FLAG, monthday)

	plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
