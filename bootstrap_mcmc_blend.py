# bootstrap sampling of 3 100-member ensembles to get the distribution of the inter-product spread

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
# import xskillscore as xs

OIB_STATUS = 'detailed'
EXTRA_FMT = '40_years_final_5k_J5_cscalib_cov'
VARNAME = 'snow_depth'

# TRYING JUST A SINGLE YEAR FOR NOW
current_year=2000
N_ITER_BS = 10 # number of bootstrap iterations
LOOP_ITER = 10

# load jra
model_data_j5 = xr.open_mfdataset('/users/jk/20/acabaj/nesosim_uncert_output_oib_{}{}/100km/JRA55*/final/NESOSIMv11_0109{}-3004{}.nc'.format(OIB_STATUS,EXTRA_FMT,current_year,current_year+1),combine='nested',concat_dim='iteration_number')

# load m2
EXTRA_FMT = '40_years_final_5k_M2_cscalib_cov'

model_data_m2 = xr.open_mfdataset('/users/jk/20/acabaj/nesosim_uncert_output_oib_{}{}/100km/MERRA*/final/NESOSIMv11_0109{}-3004{}.nc'.format(OIB_STATUS,EXTRA_FMT,current_year,current_year+1),combine='nested',concat_dim='iteration_number')

# load era5
EXTRA_FMT = '40_years_final_5k_cov'


model_data_e5 = xr.open_mfdataset('/users/jk/20/acabaj/nesosim_uncert_output_oib_{}{}/100km/ERA5*/final/NESOSIMv11_0109{}-3004{}.nc'.format(OIB_STATUS,EXTRA_FMT,current_year,current_year+1),combine='nested',concat_dim='iteration_number')

# combine these datasets
data_list = [model_data_e5, model_data_j5,model_data_m2]
DATASET_LIST = ['ERA5','JRA-55','MERRA-2']
data_all = xr.concat(data_list, pd.Index(DATASET_LIST, name='product'))


# stack the data and reindex! call the iteration number for the ensemble ensemble_number
stacked_data = data_all.stack(ensemble_number=("iteration_number", "product"))

# # dask will complain a lot about the below operation. may want to try rechunking this?
# # 'rechunking an array created with open_mfdataset is not recommended'
# stacked_data = stacked_data.reindex(ensemble_number=np.arange(300))
# won't reindex because it raises mem errors; instead just try manually doing bootstrap


# random integers for bootstrap, 1000x300 array of random integers
# trying this so that we can make sure there isn't a bias against selections that cause
# memory errors
bootstrap_idx_arr = np.loadtxt('rand_int_for_bootstrap.csv',dtype=int)


# take just snow depth (or other var)
stacked_data_1var = stacked_data[VARNAME]

# multiindex for ensemble members; used to convert from numerical indices
# to location in multi-reanalysis array for bootstrap sampling
ensemble_coords = stacked_data['ensemble_number']

# iterate over the number of bootstrap iterations
for i in range(N_ITER_BS):
	print('iteration count {}'.format(i))

	# select 300 indices for this iteration from bootstrap-generated list
	idx_rand300 = bootstrap_idx_arr[i] 
	# select corresponding coordinates for the array
	coord_rand300 = ensemble_coords[idx_rand300]
	# select the 300 random samples from the array using the coordinates
	selected_data_bs = stacked_data_1var.sel(ensemble_number=coord_rand300)

	# calculate statistics
	mean_bs_1iter = selected_data_bs.mean(dim='ensemble_number')
	sd_bs_1iter = selected_data_bs.std(dim='ensemble_number')

	print('saving data for iteration {}'.format(i))
	# save as netcdf
	mean_bs_1iter.to_netcdf('/users/jk/20/acabaj/bootstrap_samples/bootstrap_iter_{}_year_{}_mean_{}_idx_{}.nc'.format(N_ITER_BS, current_year, VARNAME, i))
	sd_bs_1iter.to_netcdf('/users/jk/20/acabaj/bootstrap_samples/bootstrap_iter_{}_year_{}_standard_dev_{}_idx_{}.nc'.format(N_ITER_BS, current_year, VARNAME, i))
	print('saved data for iteration {}'.format(i))
	
print('completed all iterations')
