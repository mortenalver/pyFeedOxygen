# Script to calculate ensemble average of pyFeedOxygen ensemble run

import netCDF4 as nc
import datetime
import numpy as np
import os
import glob

saveDir = "./bjoroya/"
#saveDir = "./saga/"
#simNamePrefix = "bjoroya_15_"
#saveDir = "D:/nn9828k/pyFeedOxygen/bjoroya/tmp/"
simNamePrefix = 'bjoroya_27j_1_'
#simNamePrefix = 'bjoroya_asm_all_'

def createNcFile(filename, oldNcfile, timeSteps):
    ncfile = nc.Dataset(filename, "w", True)
    # Create time dimension:
    ncfile.createDimension('time', minTimeSteps)
    # Duplicate all other dimensions:
    for dim in oldNcfile.dimensions:
        if not dim == "time":
            ncfile.createDimension(dim, oldNcfile.dimensions[dim].size)
    # Create time variable:
    ncfile.createVariable('time', 'f8', ('time',))
    ncfile.variables['time'].units = oldNcfile.variables['time'].units
    return ncfile

def copyFileContents(ncfile, ncfiles, minTimeSteps):
    # Copy time values:
    ncfile.variables['time'][:] = ncfiles[0].variables['time'][0:minTimeSteps]
    for var in ncfiles[0].variables:
        if not var == 'time':
            print(var)
            first = True
            for ncf in ncfiles:
                if first:
                    first = False
                    ncfile.createVariable(var, 'f8', ncf.variables[var].dimensions)
                    value = ncf.variables[var][0:minTimeSteps, ...]
                else:
                    value = value + ncf.variables[var][0:minTimeSteps, ...]
            ncfile.variables[var][:] = value / len(ncfiles)

files = os.listdir(saveDir)

# Handle scalar files:
foundFiles = []
count = 0
while simNamePrefix+(str(count)).zfill(2)+"_fish.nc" in files:
    foundFiles.append(simNamePrefix+(str(count)).zfill(2)+"_fish.nc")
    count += 1

minTimeSteps = -1
ncfiles = []
for file in foundFiles:
    ncfile = nc.Dataset(saveDir+file, "r")
    ncfiles.append(ncfile)
    timeSteps = ncfile.dimensions['time'].size
    if minTimeSteps==-1 or timeSteps < minTimeSteps:
        minTimeSteps = timeSteps

ncfile = createNcFile(saveDir+simNamePrefix+"mean_fish.nc", ncfiles[0], minTimeSteps)
copyFileContents(ncfile, ncfiles, minTimeSteps)

# Handle 3D files:
foundFiles = []
count = 0
while simNamePrefix+(str(count)).zfill(2)+".nc" in files:
    foundFiles.append(simNamePrefix+(str(count)).zfill(2)+".nc")
    count += 1

minTimeSteps = -1
ncfiles = []
for file in foundFiles:
    ncfile = nc.Dataset(saveDir+file, "r")
    ncfiles.append(ncfile)
    timeSteps = ncfile.dimensions['time'].size
    if minTimeSteps==-1 or timeSteps < minTimeSteps:
        minTimeSteps = timeSteps

ncfile = createNcFile(saveDir+simNamePrefix+"mean.nc", ncfiles[0], minTimeSteps)
copyFileContents(ncfile, ncfiles, minTimeSteps)



ncfile.close()


