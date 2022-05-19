# calculate sea ice thickness and uncertainty estimate from nesosim output 
# & regridded ICESat-2 freeboard

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import cartopy.crs as ccrs
import seaborn as sns



def plot_map(var, lons, lats, title, filename, sic, cmap='Blues', **kwargs):
	'''create a map plot of variable var, longitudes lons, latitudes lats
	title (for plot), to be saved to file named filename.
	sic is a 2d array of sea ice concentration (binary mask?)
	kwargs are for pcolormesh (eg. vmin, vmax, cmap, etc.)
	'''
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	proj=ccrs.NorthPolarStereo(central_longitude=-45)
	proj_coord = ccrs.PlateCarree()


	fig=plt.figure(dpi=200)
	ax = plt.axes(projection = proj)
	# sic mask; factor of 0.3 to adjust shading colour
	ax.pcolormesh(lons,lats,sic*0.3, transform=proj_coord, shading='flat', cmap="Greys",vmin=0,vmax=1, label='SIC >= 0.5')
	pcm = ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat', cmap=cmap, **kwargs) # using flat shading avoids artefacts
	ax.coastlines(zorder=3)
	ax.gridlines(draw_labels=False,
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


def plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename, nbins=20, cmap='Blues', color='m',**kwargs):
	''' assumes x and y are unflattened nd arrays (same shape/size)
	title, xlabel, ylabel, filename are str for plot labels
	masks out nan values
	kwargs for additional args to hist2d (eg. range)'''

	# todo: marginal axes? adjust colormap, etc.

	sns.set_theme(style="white")
	sns.set_context("talk")

	mask = ~np.isnan(x) & ~np.isnan(y)

	# plt.figure(dpi=200)
	# sns.jointplot(x[mask].flatten(), y[mask].flatten(), color='m')
	histplot = sns.jointplot(x[mask].flatten(), y[mask].flatten(), kind="hist",bins=nbins, cbar=True,marginal_ticks=False, color=color,space=0.4,marginal_kws=dict(bins=nbins,**kwargs),cbar_kws=dict(label='Count'),**kwargs)#,cmap='viridis')#,cbar_kws = dict(use_gridspec=False,location="bottom"))

	plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
	# get the current positions of the joint ax and the ax for the marginal x
	pos_joint_ax = histplot.ax_joint.get_position()
	pos_marg_x_ax = histplot.ax_marg_x.get_position()
	# reposition the joint ax so it has the same width as the marginal x ax
	histplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
	# reposition the colorbar using new x positions and y positions of the joint ax
	histplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
	# plt.tight_layout()
	histplot.ax_marg_y.tick_params(labelleft=False)
	histplot.ax_marg_x.tick_params(labelbottom=False)
	sns.despine(left=True, bottom=True)# offset=20)
	# sns.despine(offset=40)

	histplot.set_axis_labels(xlabel, ylabel)
	histplot.fig.suptitle(title)



	# # plt.hist2d(x[mask].flatten(), y[mask].flatten(),bins=nbins, color='m', **kwargs)
	# plt.title(title)
	# plt.xlabel(xlabel)
	# plt.ylabel(ylabel)
	# plt.colorbar()
	#plt.tight_layout()
	plt.savefig(filename,bbox_inches = 'tight')

def plot_nan_masked_hist_1d(data1, data2, title, data1_name, data2_name, filename, xlabel, ylabel, nbins=20, alpha=0.7,**kwargs):
	''' assumes x and y are unflattened nd arrays (same shape/size)
	title, xlabel, ylabel, filename are str for plot labels
	masks out nan values
	kwargs for additional args to hist2d (eg. range)'''

	# todo: marginal axes? adjust colormap, etc.

	mask = ~np.isnan(data1) & ~np.isnan(data2)

	plt.figure(dpi=200)
	plt.hist(data1[mask].flatten(),bins=nbins, alpha=alpha, label=data1_name, **kwargs)
	plt.hist(data2[mask].flatten(),bins=nbins, alpha=alpha, label=data2_name, **kwargs)
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.legend()
	fn2 = filename
	plt.savefig(fn2)


def plot_single_hist(data, title, filename, xlabel, ylabel, bins=20, **kwargs):
	'''plot a histogram for a single value'''

	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	plt.figure(dpi=200)
	plt.hist(data, bins=bins, **kwargs)
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.savefig(filename)


# which plots to make (to avoid excessive re-running)
MAKE_MAP_PLOTS = False# plot maps of uncertainty for the month
MAKE_SIT_CORREL_PLOTS = False# plot nesosim-mcmc and regridded is2 product sit
MAKE_UNCERT_CORREL_PLOTS = False# plot comparison plots of the uncertainties
MAKE_SNOW_DEPTH_DENS_PLOTS = False 
MAKE_1D_HIST_PLOTS = False
MAKE_BOX_PLOTS = True
MAKE_PERCENT_PLOTS = False


# estimate based on retrieval in Petty et al 2020

# monthday='2018-11'
# monthday='2019-01'
# monthday='2019-03'
# monthday='2019-04'

#data_flag = 'oib_averaged'
# data_flag = 'oib_detailed'


# data_flag_list = ['oib_averaged', 'oib_detailed']
date_list = ['2018-11', '2019-01', '2019-03']

data_flag_list = ['oib_detailed']
#date_list = ['2018-11', '2019-01']
# date_list = ['2019-03']

# big for loop? iterate over data_flag and monthday
# data_flag is no longer a constant I guess

# dictionary for collecting values

val_dict = {}
val_dict['hs'] = []
val_dict['ehs'] = []
val_dict['sit_mcmc'] = []
val_dict['sit_is2'] = []
val_dict['month'] = []
val_dict['e_sit_mcmc'] = []
val_dict['e_sit_is2'] = []
val_dict['hs_default'] = []
val_dict['rs_default'] = []
val_dict['rs'] = []
val_dict['ers'] = []

for data_flag, monthday in itertools.product(data_flag_list, date_list):




	print('making plots for {} in {}'.format(data_flag, monthday))

	is2_data = xr.open_dataset('gridded_freeboard_{}.nc'.format(monthday))
	fig_path = 'Figures/{}/'.format(monthday)

	if data_flag == 'oib_averaged':
		# oib averaged
		# this contains the entire 2018-2019 year; if I do another year then I'll have to change this!
		nesosim_data_path = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF1.7284668037515452e-06_WPT5_LLF1.2174787315012357e-07-100kmv112par_oib_averaged_final_5k/final/NESOSIMv11_01092018-30042019.nc'
		nesosim_uncert_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_averagedfinal_5k_2018_2019_cov/averagedfinal_5k_2018_2019_covuncert_100_iter_final.nc'
	elif data_flag == 'oib_detailed':
		# oib detailed
		nesosim_data_path = '/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.0504155592128743e-06_WPT5_LLF4.0059442776163867e-07-100kmv112par_oib_detailed_final_5k/final/NESOSIMv11_01092018-30042019.nc'
		nesosim_uncert_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_detailedfinal_5k_2018_2019_cov/detailedfinal_5k_2018_2019_covuncert_100_iter_final.nc'

	nesosim_data = xr.open_dataset(nesosim_data_path)
	nesosim_uncertainty = xr.open_dataset(nesosim_uncert_path)

	nesosim_default = xr.open_dataset('/users/jk/18/acabaj/NESOSIM/output/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF5.8e-07_WPT5_LLF2.9e-07-100kmv11mcmc/final/NESOSIMv11_01092018-30042019.nc')


	days = pd.to_datetime(nesosim_data['day'].values,format='%Y%m%d')

	nesosim_data['day'] = days
	nesosim_uncertainty['day'] = days
	nesosim_default['day'] = days

	# select corresponding month & calculate monthly mean
	print(nesosim_data)

	#print(nesosim_data.sel(day=monthday))

	nesosim_data_monthly = nesosim_data.sel(day=monthday)
	nesosim_uncert_monthly = nesosim_uncertainty.sel(day=monthday)
	nesosim_default_monthly = nesosim_default.sel(day=monthday)


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

	# default snow depth
	h_s_default = nesosim_default_monthly['snow_depth'].mean(axis=0).values
	r_s_default = nesosim_default_monthly['snow_density'].mean(axis=0).values

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
		filename = '{}sea_ice_thickness_estimate_{}_{}.png'.format(fig_path,data_flag,monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=5, cmap='Purples')


		# sea ice thickness unertainty from NESOSIM-mcmc using uncertainty formula
		var = random_uncert
		title = 'Sea ice thickness uncertainty for {} (m)'.format(monthday)
		filename = '{}sea_ice_thickness_uncert_{}_{}.png'.format(fig_path,data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=0.7, cmap='Greens')

		# 'snow only' sea ice thickness uncertainty (exclude contribution from other terms)
		var = random_uncert_snow_only
		title = 'Sea ice thickness uncertainty (from snow only) for {} (m)'.format(monthday)
		filename = '{}sea_ice_thickness_uncert_snow_only_{}_{}.png'.format(fig_path,data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=0.7)


		# uncertainty estimate using values from P2020 paper for snow uncert
		var = uncert_previous
		title = 'Sea ice thickness uncertainty (P2020 estimate) for {} (m)'.format(monthday)
		filename = '{}sea_ice_thickness_uncert_p2020_{}_{}.png'.format(fig_path,data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=0.7)


	# next step: compare with ensemble

	# load ensemble output



	ens_data_flag = '{}_ensemble_uncert'.format(data_flag)

	# load is2 sea ice thickness
	sit_is2 = xr.open_dataset('gridded_sit_{}.nc'.format(monthday))['sit'][0,:,:]

	sit_uncert_is2 = xr.open_dataset('gridded_sit_{}.nc'.format(monthday))['sit uncertainty'][0,:,:]

	if MAKE_SIT_CORREL_PLOTS:
		print('plotting is2 comparison plots')

		# load gridded is2 plots (provided by gridIS2thickness.py)
		
		sit_lto = sea_ice_thickness < 0
		random_uncert[sit_lto] = np.nan

		sea_ice_thickness[sit_lto] = np.nan
		random_uncert[sit_lto] = np.nan

		# gridded_sit_2019-03.nc
		# compare sit and uncertainty plots

		x, y = sit_is2.values, sea_ice_thickness
		title = 'SIT comparison for {}'.format(monthday)
		xlabel = 'IS2SITMOGR4 (m)'
		ylabel = 'NESOSIM-MCMC SIT (m)'
		horiz_label = 'SIT (m)'
		vert_label = 'Number of grid cells'
		
		filename = '{}hist_is2_vs_mcmc_sit_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)


		# plot is2 map
		var = sit_is2.values
		lons = nesosim_data['longitude']
		lats = nesosim_data['latitude']
		title = 'IS2SITMOGR4 sea ice thickness for {} (m)'.format(monthday)
		filename = '{}mcmc-is2-sit-map_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=0, vmax=5)

		# difference between is2 sit and nesosim-mcmc sit
		var = sea_ice_thickness - sit_is2.values # sit value difference
		lons = nesosim_data['longitude'] # same lat and lon used everywhere 
		lats = nesosim_data['latitude']
		title = 'NESOSIM-MCMC - IS2 SIT difference for {} (m)'.format(monthday)
		filename = '{}mcmc-is2-diff-map_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=-4, vmax=4, cmap='RdBu')


		# difference between is2 uncertainty and nesosim-mcmc uncertainty
		var = random_uncert - sit_uncert_is2.values # uncertainty value difference
		lons = nesosim_data['longitude'] 
		lats = nesosim_data['latitude']
		title = 'NESOSIM-MCMC - IS2 SIT uncert difference for {} (m)'.format(monthday)
		filename = '{}mcmc-is2-uncert-diff-map_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=-1,vmax=1,cmap='RdBu')


		# difference between is2 uncertainty and nesosim-mcmc-p2020 uncertainty
		var = sit_uncert_is2.values - uncert_previous # uncertainty value difference
		lons = nesosim_data['longitude'] 
		lats = nesosim_data['latitude']
		title = 'IS2 - P2020 NESOSIM-MCMC SIT uncert difference for {} (m)'.format(monthday)
		filename = '{}mcmc-p2020-is2-uncert-diff-map_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_map(var, lons, lats, title, filename, ice_mask_idx, vmin=-1,vmax=1,cmap='RdBu')	


		# nesosim uncertainty vs is2 uncertainty
		x, y = sit_uncert_is2.values, random_uncert
		title = 'SIT uncertainty comparison for {}'.format(monthday)
		xlabel = 'IS2SITMOGR4 uncert (m)'
		ylabel = 'NESOSIM-MCMC SIT uncert (m)'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'		
		filename = '{}hist_is2_vs_mcmc_sit_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		# plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename, range=[[0,1.2],[0,1.2]])
		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename,binrange=[0,1.2],color='g')#,xlim=(0,1.2),ylim=(0,1.2))

		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)


		# nesosim uncertainty calculated using p2020 vs. is2 uncertainty
		x, y = sit_uncert_is2.values, uncert_previous
		title = 'SIT uncertainty comparison for {}'.format(monthday)
		xlabel = 'IS2SITMOGR4 uncert (m)'
		ylabel = 'NESOSIM-MCMC SIT uncert P2020 (m)'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_is2_vs_mcmc_p2020_sit_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		# plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename, range=[[0,1.2],[0,1.2]])
		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename,binrange=[0,1.2])#,xlim=(0,1.2),ylim=(0,1.2))

		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)

		# sea ice thickness vs snow depth 
		x, y = h_s, sea_ice_thickness
		
		title = 'SIT vs snow for {}'.format(monthday)
		xlabel = 'NESOSIM-MCMC snow depth (m)'
		ylabel = 'NESOSIM-MCMC SIT (m)'
		
		filename = '{}hist_hs_vs_mcmc_sit_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)


		# snow vs freeboard 
		x,y = h_s, h_f
		x[sit_lto] = np.nan
		title = 'Freeboard vs snow for {}'.format(monthday)
		xlabel = 'NESOSIM-MCMC snow depth (m)'
		ylabel = 'IS2 ATL20 freeboard (m)'
		
		filename = '{}hist_hs_vs_freeboard_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename,color='c')
		


	if MAKE_UNCERT_CORREL_PLOTS:

		print('plotting nesosim uncertainty comparison plots')

		# load calculated ensemble uncertainty (provided by calc_ice_thickness_uncert_ensemble.py)
		ens_uncert = xr.open_dataarray('sit_uncert_ensemble_{}.nc'.format(ens_data_flag))


		# total vs ensemble uncertainty
		x, y = ens_uncert.values, random_uncert
		title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
		xlabel = 'Ensemble uncertainty'
		ylabel = 'Total uncertainty'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_ensemble_vs_total_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)

		# snow-only vs ensemble uncertainty
		x, y = ens_uncert.values, random_uncert_snow_only
		title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
		xlabel = 'Ensemble uncertainty'
		ylabel = 'Snow-only uncertainty'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_ensemble_vs_snow_only_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)

		# snow-only vs total uncertainty
		x, y = random_uncert, random_uncert_snow_only
		title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
		xlabel = 'Total uncertainty'
		ylabel = 'Snow-only uncertainty'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_snow_only_vs_total_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)

		# ensemble vs. p2020 uncertainty
		x, y = uncert_previous, ens_uncert.values
		title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
		xlabel = 'P2020 uncertainty'
		ylabel = 'Ensemble uncertainty'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_p2020_vs_ensemble_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)

		# total vs p2020 uncertainty
		x, y = uncert_previous, random_uncert
		title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
		xlabel = 'P2020 uncertainty'
		ylabel = 'Total uncertainty'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_p2020_vs_total_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)


		# snow-only vs p2020 uncertainty
		x, y = uncert_previous, random_uncert_snow_only
		title = 'SIT uncertainty comparison for {} (m)'.format(monthday)
		xlabel = 'P2020 uncertainty'
		ylabel = 'Snow-only uncertainty'
		horiz_label = 'SIT uncert (m)'
		vert_label = 'Number of grid cells'
		filename = '{}hist_p2020_vs_snow_only_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)

		plot_nan_masked_hist(x, y, title, xlabel, ylabel, filename)
		# plot_nan_masked_hist_1d(x, y, title, xlabel, ylabel, filename, horiz_label, vert_label)



	if MAKE_SNOW_DEPTH_DENS_PLOTS:
		lons = nesosim_data['longitude'] # same lat and lon used everywhere I think
		lats = nesosim_data['latitude']


		

		# snow depth
		var = h_s
		# mask out where sit is unphysical
		var[sea_ice_thickness < 0] = np.nan
		title = 'NESOSIM-MCMC snow depth for {} (m)'.format(monthday)
		filename = '{}snow_depth_map_{}_{}.png'.format(fig_path,data_flag,monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx,vmax=0.6)

		var = r_s
		var[sea_ice_thickness < 0] = np.nan
		title = 'NESOSIM-MCMC snow density for {} (kg/m^3)'.format(monthday)
		filename = '{}snow_density_map_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx,vmin=300,cmap='YlOrBr')
		
		var = e_h_s
		var[sea_ice_thickness < 0] = np.nan
		title = 'NESOSIM-MCMC snow depth uncert. for {} (m)'.format(monthday)
		filename = '{}snow_depth_unc_map_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx)
		
		var = e_r_s
		var[sea_ice_thickness < 0] = np.nan
		title = 'NESOSIM-MCMC snow density uncert. for {} (kg/m^3)'.format(monthday)
		filename = '{}snow_density_unc_map_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx)

	if MAKE_1D_HIST_PLOTS:

		# make histogram plots


		var = h_s # snow depth
		# mask out where sit is unphysical
		var[sea_ice_thickness < 0] = np.nan

		title = 'NESOSIM-MCMC snow depth distribution for {}'.format(monthday)
		filename = '{}snow_depth_distribution_1d_{}_{}.png'.format(fig_path,data_flag,monthday)

		# should I normalize these??
		xlabel = 'Snow depth (m)'
		ylabel = 'Count'

		plot_single_hist(var.flatten(), title, filename, xlabel, ylabel, bins=np.linspace(0,1,20))


		var = e_h_s
		var[sea_ice_thickness < 0] = np.nan
		title = 'NESOSIM-MCMC snow depth uncert. distribution for {} (m)'.format(monthday)
		filename = '{}snow_depth_uncert_dist_1d_{}_{}.png'.format(fig_path,data_flag,monthday)

		xlabel = 'Snow depth uncert (m)'
		ylabel = 'Count'

		plot_single_hist(var.flatten(), title, filename, xlabel, ylabel, bins=20)#np.linspace(0,1,20))


		var = sea_ice_thickness # sit
		# mask out where sit is unphysical
		var[sea_ice_thickness < 0] = np.nan

		title = 'NESOSIM-MCMC SIT distribution for {}'.format(monthday)
		filename = '{}sit_distribution_1d_{}_{}.png'.format(fig_path,data_flag,monthday)

		# should I normalize these??
		xlabel = 'Sea ice thickness (m)'
		ylabel = 'Count'

		plot_single_hist(var.flatten(), title, filename, xlabel, ylabel, bins=20)#, bins=np.linspace(0,1,20))

		

		var = random_uncert
		var[sea_ice_thickness < 0] = np.nan
		title = 'NESOSIM-MCMC SIT uncert distribution for {}'.format(monthday)

		filename = '{}sit_uncert_dist_1d_{}_{}.png'.format(fig_path,data_flag,monthday)

		xlabel = 'Sea ice thickness uncert (m)'
		ylabel = 'Count'
		plot_single_hist(var.flatten(), title, filename, xlabel, ylabel, bins=20)#, bins=np.linspace(0,1,20))

	if MAKE_BOX_PLOTS:
		# collect values

		# assuming doing this only for either oib averaged or oib detailed for now;
		# can disentangle later if necessary (but sticking to oib detailed for the moment)

		# add flattened values
		lto = sea_ice_thickness < 0
		h_s[lto] = np.nan
		e_h_s[lto] = np.nan
		sit_is2.values[lto] = np.nan
		sea_ice_thickness[lto] = np.nan
		random_uncert[lto] = np.nan
		sit_uncert_is2.values[lto] = np.nan
		sit_uncert_is2.values[sit_uncert_is2.values > 10]=np.nan
		h_s_default[lto] = np.nan 
		r_s_default[lto] = np.nan
		r_s[lto] = np.nan
		e_r_s[lto] = np.nan

		# accumulate in dictionary
		val_dict['month'].append(monthday)
		val_dict['hs'].append(h_s.flatten())
		val_dict['ehs'].append(e_h_s.flatten())
		val_dict['sit_mcmc'].append(sea_ice_thickness.flatten())
		val_dict['sit_is2'].append(sit_is2.values.flatten())
		val_dict['e_sit_mcmc'].append(random_uncert.flatten())
		val_dict['e_sit_is2'].append(sit_uncert_is2.values.flatten())
		val_dict['hs_default'].append(h_s_default.flatten())
		val_dict['rs_default'].append(r_s_default.flatten())
		val_dict['rs'].append(r_s.flatten())
		val_dict['ers'].append(e_r_s.flatten())


	if MAKE_PERCENT_PLOTS:
		# snow-only divided by is2 and make a map

		uncert_ratio = 100*random_uncert_snow_only / uncert_previous

		var = uncert_ratio # sit value difference

		# mask out ice
		uncert_ratio[sea_ice_thickness < 0] = np.nan
		lons = nesosim_data['longitude'] # same lat and lon used everywhere 
		lats = nesosim_data['latitude']
		title = 'Snow uncertainty contribution to total uncertainty (%) for {}'.format(monthday)
		filename = '{}snow_over_p2020_percent_uncert_{}_{}.png'.format(fig_path, data_flag, monthday)
		plot_map(var, lons, lats, title, filename, ice_mask_idx)#,vmax=20)






if MAKE_BOX_PLOTS:

	########### violin plot: mcmc vs is2 sit
	df1 = pd.DataFrame(np.array(val_dict['sit_mcmc']).transpose(),columns=val_dict['month'])
	df1 = df1.stack()
	df1.rename('MCMC',inplace=True) #is the renaming redundant?
	df2 = pd.DataFrame(np.array(val_dict['sit_is2']).transpose(),columns=val_dict['month'])
	df2 = df2.stack()
	df2.rename('IS2',inplace=True)
	

	df = pd.concat([df1, df2],keys=['MCMC','IS2'], axis=0).reset_index()

	df.columns = ['Product','idx','Month','value']

	plt.figure(dpi=200)
	sns.violinplot(data=df,x='Month',y='value',hue='Product',palette='crest',split=True,order=val_dict['month'],inner='quartile',cut=0) 

	#plt.xticks(ticks=range(len(val_dict['month'])), labels=val_dict['month'])
	plt.legend(loc='upper center')
	plt.ylabel('Sea ice thickness (m)')
	plt.title('Monthly sea ice thickness spatial distribution')
	plt.savefig('{}sit_mcmc_plot_violin_{}.png'.format(fig_path, data_flag))


	############# error sit violin
# 	df1 = pd.DataFrame(np.array(val_dict['e_sit_mcmc']).transpose(),columns=val_dict['month'])
# 	df1 = df1.stack()
# 	df1.rename('MCMC',inplace=True)
# 	df2 = pd.DataFrame(np.array(val_dict['e_sit_is2']).transpose(),columns=val_dict['month'])
# 	df2 = df2.stack()
# 	df2.rename('IS2',inplace=True)
	

# 	df = pd.concat([df1, df2],keys=['MCMC','IS2'], axis=0).reset_index()
# #	df.columns = ['date','value']

# 	print(df)
# 	df.columns = ['Product','idx','Month','value']
# 	# create dataframe?
# 	plt.figure(dpi=200)

# 	sns.violinplot(data=df,x='Month',y='value',hue='Product',palette='crest',split=True,order=val_dict['month'],inner='quartile',cut=0) 
# #	sns.boxplot(data=df,x='Month',y='value',hue='Product',palette='crest') 
# 	#plt.xticks(ticks=range(len(val_dict['month'])), labels=val_dict['month'])
# 	plt.legend(loc='upper center')
# 	plt.ylabel('Sea ice thickness uncertainty (m)')
# 	plt.title('Monthly sea ice thickness uncert spatial distribution')
# 	plt.savefig('{}sit_uncert_mcmc_plot_violin_{}.png'.format(fig_path, data_flag))



	########## violin default vs optimized snow depth
	# df1 = pd.DataFrame(np.array(val_dict['hs_default']).transpose(),columns=val_dict['month'])
	# df1 = df1.stack()
	# df1.rename('default',inplace=True) #is the renaming redundant?
	# df2 = pd.DataFrame(np.array(val_dict['hs']).transpose(),columns=val_dict['month'])
	# df2 = df2.stack()
	# df2.rename('MCMC',inplace=True)


	# df = pd.concat([df2, df1],keys=['MCMC','Default'], axis=0).reset_index()

	# df.columns = ['Product','idx','Month','value']

	# plt.figure(dpi=200)
	# sns.violinplot(data=df,x='Month',y='value',hue='Product',palette='Blues',split=True,order=val_dict['month'],inner='quartile',cut=0) 

	# plt.legend(loc='upper center')
	# plt.ylabel('Snow depth (m)')
	# plt.title('Monthly snow depth spatial distribution')
	# plt.savefig('{}snow_depth_plot_violin_{}.png'.format(fig_path, data_flag))


	# ######### violing default vs optimized snow density
	# df1 = pd.DataFrame(np.array(val_dict['rs_default']).transpose(),columns=val_dict['month'])
	# df1 = df1.stack()
	# df1.rename('default',inplace=True) #is the renaming redundant?
	# df2 = pd.DataFrame(np.array(val_dict['rs']).transpose(),columns=val_dict['month'])
	# df2 = df2.stack()
	# df2.rename('MCMC',inplace=True)


	# df = pd.concat([df2, df1],keys=['MCMC','Default'], axis=0).reset_index()

	# df.columns = ['Product','idx','Month','value']

	# plt.figure(dpi=200)
	# sns.violinplot(data=df,x='Month',y='value',hue='Product',palette='YlOrBr',split=True,order=val_dict['month'],inner='quartile',cut=0) 

	# plt.legend(loc='lower right')
	# plt.ylabel('Snow density (kg/m$^3$)')
	# plt.title('Monthly snow density spatial distribution')
	# plt.savefig('{}snow_density_plot_violin_{}.png'.format(fig_path, data_flag))


	################ snow depth uncertainty



	# df1 = pd.DataFrame(np.array(val_dict['ehs']).transpose(),columns=val_dict['month'])
	# df1 = df1.stack().to_frame().reset_index()


	# df1.columns = ['idx','Month','value']

	# plt.figure(dpi=200)
	# sns.violinplot(data=df1,x='Month',y='value', palette='Blues', split=True, order=val_dict['month'], inner='quartile',cut=0) 

	# plt.ylabel('Snow depth uncert (m)')
	# plt.title('Monthly snow depth uncertainty spatial distribution')
	# plt.savefig('{}snow_depth_uncert_plot_violin_{}.png'.format(fig_path, data_flag))



	############## snow density uncertainty
	# df1 = pd.DataFrame(np.array(val_dict['ers']).transpose(),columns=val_dict['month'])
	# df1 = df1.stack().to_frame().reset_index()


	# df1.columns = ['idx','Month','value']

	# plt.figure(dpi=200)
	# sns.violinplot(data=df1,x='Month',y='value', palette='YlOrBr', split=True, order=val_dict['month'], inner='quartile',cut=0) 

	# plt.ylabel('Snow density uncert (kg/m$^3$)')
	# plt.title('Monthly snow density uncertainty spatial distribution')
	# plt.savefig('{}snow_dens_uncert_plot_violin_{}.png'.format(fig_path, data_flag))



	
	####### snow depth + dens uncertainty double figure plot
	df1 = pd.DataFrame(np.array(val_dict['ehs']).transpose(),columns=val_dict['month'])
	df1 = df1.stack().to_frame().reset_index()

	df1.columns = ['idx','Month','value']

	df2 = pd.DataFrame(np.array(val_dict['ers']).transpose(),columns=val_dict['month'])
	df2 = df2.stack().to_frame().reset_index()


	df2.columns = ['idx','Month','value']

	fig, (ax1, ax2) = plt.subplots(1, 2,dpi=200,figsize=(8,4))
	sns.violinplot(data=df1,x='Month',y='value', palette='Blues', split=True, order=val_dict['month'], inner='quartile',cut=0,ax=ax1)
	sns.violinplot(data=df2,x='Month',y='value', palette='YlOrBr', split=True, order=val_dict['month'], inner='quartile',cut=0,ax=ax2) 
 

	ax1.set_ylabel('Snow depth uncert (m)')
	ax2.set_ylabel('Snow density uncert (kg/m$^3$)')
	fig.suptitle('Monthly snow uncertainty spatial distributions')


	plt.savefig('{}snow_depth_dens_uncert_subplots_violin_{}.png'.format(fig_path, data_flag))


	############## snow depth and snow density double figure plot
	df1 = pd.DataFrame(np.array(val_dict['hs_default']).transpose(),columns=val_dict['month'])
	df1 = df1.stack()
	df1.rename('default',inplace=True) #is the renaming redundant?
	df2 = pd.DataFrame(np.array(val_dict['hs']).transpose(),columns=val_dict['month'])
	df2 = df2.stack()
	df2.rename('MCMC',inplace=True)


	df_depth = pd.concat([df2, df1],keys=['MCMC','Default'], axis=0).reset_index()

	df_depth.columns = ['Product','idx','Month','value']

	df1 = pd.DataFrame(np.array(val_dict['rs_default']).transpose(),columns=val_dict['month'])
	df1 = df1.stack()
	df1.rename('default',inplace=True) #is the renaming redundant?
	df2 = pd.DataFrame(np.array(val_dict['rs']).transpose(),columns=val_dict['month'])
	df2 = df2.stack()
	df2.rename('MCMC',inplace=True)


	df_dens = pd.concat([df2, df1],keys=['MCMC','Default'], axis=0).reset_index()

	df_dens.columns = ['Product','idx','Month','value']


	# have df_depth and df_dens and want to plot subfigures
	fig, (ax1, ax2) = plt.subplots(1, 2,dpi=200,figsize=(8,4))

	sns.violinplot(data=df_depth,x='Month',y='value',hue='Product',palette='Blues',split=True,order=val_dict['month'],inner='quartile',cut=0,ax=ax1) 
	sns.violinplot(data=df_dens,x='Month',y='value',hue='Product',palette='YlOrBr',split=True,order=val_dict['month'],inner='quartile',cut=0,ax=ax2) 
	ax1.legend(loc='upper center')
	ax2.legend(loc='lower right')
	ax1.set_ylabel('Snow depth (m)')
	ax2.set_ylabel('Snow density (kg/m$^3$)')
	fig.suptitle('Monthly snow spatial distributions')
	plt.savefig('{}snow_depth_dens_subplots_violin_{}.png'.format(fig_path, data_flag))