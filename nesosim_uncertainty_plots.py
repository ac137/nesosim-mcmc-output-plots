# plot nesosim output and uncertainty


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import cartopy.crs as ccrs

file_path_1 = '/home/alex/modeldev/output_with_uncert/mean_detailed_100_iter.nc'
file_path_2 = '/home/alex/modeldev/output_with_uncert/uncert_detailed_100_iter.nc'

# file_path_1 = '/home/alex/modeldev/output_with_uncert/mean_averaged_100_iter.nc'
# file_path_2 = '/home/alex/modeldev/output_with_uncert/uncert_averaged_100_iter.nc'

print('file paths')


#TODO: more descriptive variable names here
# clean this up (taken from jupyter notebook)
# set up to run on fileserver

data_1 = xr.open_dataset(file_path_1)
data_2 = xr.open_dataset(file_path_2)


# testing map plots

proj=ccrs.NorthPolarStereo(central_longitude=-45)
proj_coord = ccrs.PlateCarree()

print('snow depth (m, end of season)')

lons = data_1['longitude']
lats = data_1['latitude']
var = data_1['snow_depth'][-1] #-1 to select last day of season

fig=plt.figure(dpi=200)
ax = plt.axes(projection = proj)
ax.pcolormesh(lons,lats,var,transform=proj_coord,shading='flat') # using flat shading avoids artefacts
ax.coastlines(zorder=3)
ax.gridlines(draw_labels=True,
          linewidth=0.22, color='gray', alpha=0.5, linestyle='--')

# for some reason this extent complains if you set set -180 to +180
ax.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())
plt.show()


# plotting this as 2 figures with subplots; bit difficult to read

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(1, 2, 1, projection=proj)
ax2 = fig.add_subplot(1, 2, 2, projection=proj)
ax1.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())
ax2.set_extent([-180, 179.9, 45, 90], ccrs.PlateCarree())

im = ax1.pcolormesh(lons,lats,data_1['snow_depth'][-1],transform=proj_coord,shading='flat')
ax1.set_title('mean snow depth')
fig.colorbar(im, ax=ax1)
im = ax2.pcolormesh(lons,lats,data_2['snow_depth'][-1],transform=proj_coord,shading='flat')
ax2.set_title('snow depth uncertainty')
ax1.gridlines()
ax1.coastlines()
ax2.gridlines()
ax2.coastlines()
fig.colorbar(im, ax=ax2)
plt.show()


# same plots as above but just using imshow, max set to 1 on colourbar (cutting off higher values)

fig, (ax1, ax2) = plt.subplots(1, 2,dpi=200)
im = ax1.imshow(data_1['snow_depth'][-1],origin='lower',vmax=1) #-1 to select last day of season
ax1.set_title('Snow depth mean')
fig.colorbar(im, ax=ax1)
im = ax2.imshow(data_2['snow_depth'][-1],origin='lower',vmax=0.1)
ax2.set_title('Snow depth uncert')
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()



# snow density plots
print('snow density (end of season)')

fig, (ax1, ax2) = plt.subplots(1, 2,dpi=200)
im = ax1.imshow(data_1['snow_density'][-1],origin='lower',vmin=240,vmax=350)
ax1.set_title('Mean snow density')
fig.colorbar(im, ax=ax1)
im = ax2.imshow(data_2['snow_density'][-1],origin='lower')
ax2.set_title('Uncertainty of snow density')
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()


# basin-wide daily mean for total snow depth
# calculate these and save in a different file maybe?


mean_1 = data_1['snow_depth'].mean(axis=(1,2))
mean_2 = data_2['snow_depth'].mean(axis=(1,2))


# basin-wide snow depth & uncertainty plot; maybe do one with shaded fill instead
plt.figure(dpi=200)
plt.title('basin-wide means of total snow depth with uncertainty')
# plt.plot(mean_1, label='mean')
# plt.plot(mean_2, label='2')
plt.errorbar(x=np.arange(len(mean_1)),y=mean_1,yerr=mean_2)
plt.ylabel('snow depth (m)')
plt.xlabel('days since 1 sept.')
# plt.legend()
plt.show()


#1-year mean snow depth maps

reg_m_1 = data_1['snow_depth'].mean(axis=0)
reg_m_2 = data_2['snow_depth'].mean(axis=0)

# plots of 1-year mean snow depth

print('1-year mean snow depth')
fig, (ax1, ax2) = plt.subplots(1, 2,dpi=200)
im = ax1.imshow(reg_m_1,origin='lower')
ax1.set_title('mean 1-y mean')
fig.colorbar(im, ax=ax1)
im = ax2.imshow(reg_m_2,origin='lower')
ax2.set_title('1-year mean uncertainty')
fig.colorbar(im, ax=ax2)
plt.show()


# time series of snow density by layer

dens_1 = data_1['snow_density'].mean(axis=(1,2))
dens_2 = data_2['snow_density'].mean(axis=(1,2))

plt.figure(dpi=200)
plt.title('basin-wide means of snow density with uncertainty')
# plt.plot(mean_1, label='mean')
# plt.plot(mean_2, label='2')
plt.errorbar(x=np.arange(len(dens_1)-1),y=dens_1[1:],yerr=dens_2[1:])
plt.ylabel('snow density (m)')
plt.xlabel('days since 1 sept.')
# plt.legend()
plt.show()