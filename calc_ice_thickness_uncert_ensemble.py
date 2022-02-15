# propagate uncertainty from "parameter ensemble" from mcmc output

import numpy as np
import pandas as pd
import xarray as xr

# icesat-2 data
is2_data = xr.open_dataset('gridded_freeboard_2019-03.nc')

OIB_STATUS = 'averaged'
EXTRA_FMT = 'final_5k_2018_2019_cov'
DATA_FLAG = 'oib_averaged_ensemble_uncert'

# open all the ensembe members and concatenate into a big dataset
nesosim_data = xr.open_mfdataset('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/100km/ERA*/final/*.nc'.format(OIB_STATUS,EXTRA_FMT),combine='nested',concat_dim='iteration_number')

print(nesosim_data)

# dimensions should be (iteration number, day, x, y)

# convert days to datetime
days = pd.to_datetime(nesosim_data['day'].values,format='%Y%m%d')

nesosim_data['day'] = days

print(nesosim_data)

# select single month
nesosim_data_monthly = nesosim_data.sel(day="2019-03")

print('single month?')
print(nesosim_data_monthly)
# calculate monthly mean for each ensemble member, I guess?

# density of water
r_w =  1024 #kg/m^3
# density of ice
r_i = 915 #kg/m^3 (as assumed in Petty et al 2020)
# density of snow
r_s = nesosim_data_monthly['snow_density'].mean(axis=1)#.values
# snow density comes fron nesosim

print('monthly mean snow density')
print(r_s)

# looks like we get the right shape now!
# # freeboard error # don't need this
# e_h_f = is2_data['freeboard uncertainty'].values[0,:,:]

# ice density error
# e_r_i = 10 #kg/m^3 based on Alexandrov et al. (2013)
# snow depth

h_s = nesosim_data_monthly['snow_depth'].mean(axis=1).values
# freeboard height from is2
h_f = is2_data['freeboard'].values[0,:,:]



# 1/(r_w-r_s) because this term shows up a lot
inverse_r_w_minus_r_i = 1/(r_w - r_i)


# do the math; get a sea ice thickness ensemble (hopefully it doesn't complain about dimension mismatch? may need transpose)
ens_sea_ice_thickness = h_f*r_w*inverse_r_w_minus_r_i + h_s*(r_s-r_w)*inverse_r_w_minus_r_i

# this would now be an n_iter*90*90 array?

# calculate standard deviation to produce an uncertainty estimate!
sea_ice_uncert = np.std(ens_sea_ice_thickness,axis=0)

print(sea_ice_uncert)
print(sea_ice_uncert.shape) #should be 90x90 if all goes well...


plt.figure()
plt.imshow(sea_ice_uncert,origin='lower',vmin=0,vmax=5)
plt.colorbar()
plt.savefig('sea_ice_thickness_uncert_estimate_{}.png'.format(DATA_FLAG))
