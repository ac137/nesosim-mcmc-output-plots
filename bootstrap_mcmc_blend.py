# bootstrap sampling of 3 100-member ensembles to get the distribution of the inter-product spread

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import xskillscore as xs

OIB_STATUS = 'detailed'
EXTRA_FMT = '40_years_final_5k_J5_cscalib_cov'
VARNAME = 'snow_depth'

# TRYING JUST A SINGLE YEAR FOR NOW
current_year=2000
N_ITER_BS = 10
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

# dask will complain a lot about this operation. may want to try rechunking this?
# 'rechunking an array created with open_mfdataset is not recommended'
stacked_data = stacked_data.reindex(ensemble_number=np.arange(300))


# take just snow depth

md_sd = stacked_data[VARNAME]

for i in range(5,LOOP_ITER):
	print('iteration count {}'.format(i))

	# number of bootstrap iterations
	resamp_1 = xs.resample_iterations(md_sd, N_ITER_BS, 'ensemble_number')

	# calculate mean and standard dev.
	mean = resamp_1.mean(dim='ensemble_number')

	sd = resamp_1.std(dim='ensemble_number')


	# save as netcdf
	mean.to_netcdf('/users/jk/20/acabaj/bootstrap_samples/bootstrap_iter_{}_year_{}_mean_{}_nsamp_{}.nc'.format(N_ITER_BS, current_year, VARNAME, i))
	sd.to_netcdf('/users/jk/20/acabaj/bootstrap_samples/bootstrap_iter_{}_year_{}_standard_dev_{}_nsamp_{}.nc'.format(N_ITER_BS, current_year, VARNAME, i))

print('completed all iterations')
