import matplotlib.pyplot as plt
import netCDF4

# open a local NetCDF file or remote OPeNDAP URL
url = '/mnt/napster_disk/space_apps/nasa_deploy/scripts/S5P_OFFL_L2__NO2____20181130T132824_20181130T150953_05862_01_010202_20181206T150915.nc'
nc = netCDF4.Dataset(url)

# examine the variables
print(nc.variables.keys())
print(nc.variables['precipitationCal'])

# sample every 10th point of the 'z' variable
topo = nc.variables['precipitationCal'][::10, ::10]

# make image
plt.figure(figsize=(10,10))
plt.savefig('image.png', bbox_inches=0)