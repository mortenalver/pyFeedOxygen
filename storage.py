import numpy as np
import netCDF4 as nc

ensembleStateName = 'X'
ensembleAnalysisName = 'X_a'
twinStateName = 'X_t'
maskName = 'mask'
ensembleDeviationName = 'deviation'
ensembleExactDeviationName = 'deviation_exact'
ensembleCovName = 'cov'
ensembleKName = 'K'
ensembleMName = 'M'
ensembleLocName = 'Xloc'
thetaName = 'theta'
omegaName = 'omega'

def addMatdispAttributes(ncfile, dxy, dz):
    ncfile.TITLE = 'AQUAEXCEL'
    ncfile._FillValue = -32768
    ncfile.grid_mapping_name = 'polar_stereographic'
    ncfile.horizontal_resolution = dxy
    ncfile.vertical_resolution = dz
    ncfile.coordinate_north_pole = (0, 0)
    ncfile.latitude_of_projection_origin = 90
    ncfile.standard_parallel = 0.0

def initScalarFile(filename, timeUnitString, cageDims, nGroups):
    ncfile = nc.Dataset(filename, "w", True)
    time_dim = ncfile.createDimension('time', None)
    xc = ncfile.createDimension('xc', nGroups)
    zc = ncfile.createDimension('zc', cageDims[2])
    # Create variables:
    tVar = ncfile.createVariable('time', 'f8', ('time'))
    tVar.units = timeUnitString
    return ncfile


def init3DFile(filename, timeUnitString, cageDims, dxy, dz):
    ncfile = nc.Dataset(filename, "w", True)
    time_dim = ncfile.createDimension('time', None)
    xc = ncfile.createDimension('xc', cageDims[0])
    yc = ncfile.createDimension('yc', cageDims[1])
    zc = ncfile.createDimension('zc', cageDims[2])

    # Set attributes:
    addMatdispAttributes(ncfile, dxy, dz)

    # Create variables:
    tVar = ncfile.createVariable('time', 'f8', ('time'))
    tVar.units = timeUnitString

    return ncfile


def initEnsembleFile(filename, nstates, npar, N, cageDims, nmeas):
    ncfile = nc.Dataset(filename, "w", True)
    time_dim = ncfile.createDimension('time', None)
    xc = ncfile.createDimension('xc', nstates+npar)
    yc = ncfile.createDimension('yc', N)
    meas = ncfile.createDimension('zc', nmeas)
    ncfile.cageDims = cageDims
    # Create variables:
    ncfile.createVariable('time', 'f8', ('time'))
    ncfile.createVariable(ensembleStateName, 'f8', ('time', 'yc', 'xc'))
    ncfile.createVariable(twinStateName, 'f8', ('time', 'xc'))
    ncfile.createVariable(maskName, 'f8', ('time', 'xc'))
    ncfile.createVariable(ensembleAnalysisName, 'f8', ('time', 'yc', 'xc'))
    ncfile.createVariable(ensembleDeviationName, 'f8', ('time', 'yc', 'zc'))
    ncfile.createVariable(ensembleExactDeviationName , 'f8', ('time', 'yc', 'zc'))
    ncfile.createVariable(ensembleCovName, 'f8', ('time', 'yc', 'xc'))
    ncfile.createVariable(ensembleKName, 'f8', ('time', 'zc', 'xc'))
    ncfile.createVariable(ensembleMName, 'f8', ('time', 'xc', 'zc'))
    ncfile.createVariable(ensembleLocName, 'f8', ('time', 'zc', 'xc'))
    ncfile.createVariable(thetaName, 'f8', ('time', 'yc', 'xc'))
    ncfile.createVariable(omegaName, 'f8', ('time', 'yc', 'zc'))
    return ncfile

def createCageVariables(ncfile, vars):
    for var in vars:
        ncfile.createVariable(var, np.float64, ('time', 'zc', 'yc', 'xc'))

def createGroupVariables(ncfile, vars):
    for var in vars:
        ncfile.createVariable(var, np.float64, ('time', 'xc'))

def createProfileVariables(ncfile, vars):
    for var in vars:
        ncfile.createVariable(var, np.float64, ('time', 'zc'))

def createScalarVariables(ncfile, vars):
    for var in vars:
        ncfile.createVariable(var, np.float64, ('time'))

def syncFile(ncfile):
    try:
        ncfile.sync()
    except:
        ncfile.close()

def saveCageVariable(ncfile, time, name, values, useMask, mask, writeTime):

    tVar = ncfile.variables['time']
    if writeTime:
        index = tVar.shape[0]
        tVar[index] = time
    else:
        index = tVar.shape[0]-1
    fVar = ncfile.variables[name]
    vv = values.copy()
    if useMask:
        vv[~mask] = -32768
    fVar[index,...] = np.transpose(vv, (2, 1, 0))

def saveGroupVariable(ncfile, time, name, values, writeTime):
    tVar = ncfile.variables['time']
    if writeTime:
        index = tVar.shape[0]
        tVar[index] = time
    else:
        index = tVar.shape[0] - 1
    fVar = ncfile.variables[name]
    vv = values.copy()

    fVar[index, ...] = vv[...]


def saveScalarVariable(ncfile, time, name, value, writeTime):
    tVar = ncfile.variables['time']
    if writeTime:
        index = tVar.shape[0]
        tVar[index] = time
    else:
        index = tVar.shape[0] - 1

    fVar = ncfile.variables[name]
    fVar[index] = value

def saveProfileVariable(ncfile, time, name, values, writeTime):
    tVar = ncfile.variables['time']
    if writeTime:
        index = tVar.shape[0]
        tVar[index] = time
    else:
        index = tVar.shape[0] - 1
    fVar = ncfile.variables[name]
    vv = values.copy()
    fVar[index, ...] = vv[...]

def saveEnsembleState(ncfile, time, mask, X, X_a, X_twin, M, dev, exactDev, K, Xloc, theta, omega, writeTime):
    tVar = ncfile.variables['time']
    if writeTime:
        index = tVar.shape[0]
        tVar[index] = time
    else:
        index = tVar.shape[0] - 1

    fVar = ncfile.variables[maskName]
    fVar[index, ...] = np.transpose(mask)
    fVar = ncfile.variables[ensembleStateName]
    fVar[index,...] = np.transpose(X, (1, 0))
    fVar = ncfile.variables[ensembleAnalysisName]
    fVar[index, ...] = np.transpose(X_a, (1, 0))

    if X_twin is not None:
        fVar = ncfile.variables[twinStateName]
        fVar[index, ...] = np.transpose(X_twin)#np.reshape(X_t, (X_t.shape[0], ))

    fVar = ncfile.variables[ensembleMName]
    fVar[index, ...] = np.transpose(M, (1,0))
    fVar = ncfile.variables[ensembleDeviationName]
    fVar[index, ...] = np.transpose(dev, (1,0))
    fVar = ncfile.variables[ensembleExactDeviationName]
    fVar[index, ...] = np.transpose(exactDev, (1, 0))
    fVar = ncfile.variables[ensembleKName]
    #print(K.shape)
    fVar[index, 0:K.shape[1], 0:K.shape[0]] = np.transpose(K, (1, 0))
    fVar = ncfile.variables[ensembleLocName]
    fVar[index, 0:Xloc.shape[1], 0:Xloc.shape[0]] = np.transpose(Xloc, (1, 0))

    fVar = ncfile.variables[thetaName]
    fVar[index, ...] = np.transpose(theta, (1, 0))
    fVar = ncfile.variables[omegaName]
    fVar[index, ...] = np.transpose(omega, (1, 0))

def saveEnsembleCov(ncfile, time, value, cov):
    tVar = ncfile.variables['time']
    index = tVar.shape[0] - 1
    fVar = ncfile.variables[ensembleCovName]
    fVar[index, ...] = np.transpose(value, (1, 0))