import time
import numpy as np
import math
import fishingestionTempProfile
import mpi
import mpiInternal
import cagemasking
import advect
import storage
import simplefish
import behaviour
import gaussmarkov
import ensembleKalmanFilter
import measurements
import currentfields
import currentMagic
import simInputsNetcdf
import scipy.interpolate

import datetime
#from datetime import datetime

def current_milli_time():
    return round(time.time() * 1000)

if __name__ == '__main__':
    # Save information:
    saveDir = "./bjoroya/"
    #simNamePrefix = "test_internalmpi2_"
    simNamePrefix = "shrt2_"
    #simNamePrefix = "bjoroya_23_magic_"

    # Duration of simulation (seconds)
    t_end = 10 + 5*600 #1800 +50 #4*3600

    # Start time:
    initYear = 2022
    initMonth = 6
    initDate = 27
    initHour = 0
    initMin = 0
    initSec = 0

    # Common settings:
    rad = 25.#55.0
    depth = 25.0
    totDepth = 25.0
    dxy = 4.0 # Horizontal resolution
    dz = dxy # Vertical resolution
    modelDim = 2*(rad+4*dxy) #60.#120.0
    fishMaxDepth = 20. # Lowest depth where non-feeding fish will consume oxygen
    dt = 0.25*dxy # time step (seconds)


    # Storage intervals (seconds):
    storeIntervalScalars = 60 # 60
    storeIntervalFields = 600 #600
    txtUpdateInterval = 200



    # Path to input data:
    #inputFile = r"C:/Users/alver/OneDrive - NTNU/prosjekt/O2_Bjørøya/bjoroya_data2.nc"
    inputFile = r"bjoroya_data.nc"
    #inputFile = r"C:\Users\alver\OneDrive - NTNU\prosjekt\pyFeedOxygen\data\of1_data.nc"
    inputsN = simInputsNetcdf.SimInputsNetcdf(inputFile)

    # Parallelization: do internal MPI parallelization where the model domain is split into
    # parts handled by different processes. This mode rules out MPI-based EnKF.
    internalMPI = True

    # Input and data assimilation settings:
    dryRun = False # If true, no EnKF updates will be applied
    assimCutoff = True # If true, stop EnKF updates at the time given by assimCutoffTime
    assimCutoffTime = 2*3600
    advectFeed = True # If false, feed advection will be skipped
    feedAdvectLimit = 0.1 # Minimum total amount of feed to advect. If sum is below, skip advection of feed
    useNetcdfInputs = True # Read and update input data from NetCDF file (OF1)
    perturbations = False #not dryRun # not dryRun # Add random perturbations to process (needed for EnKF):
    enKFInterval = 60 # Seconds (measurements come at 60 s interval)
    localization = True # Limit spatial "reach" of corrections based on measurements
    ensembleInflation = True # Ensemble inflation applied at each analysis time
    ensembleInflationFactor = 1.05
    localizationDistM = 20 # Distance where covariances are scaled down by 50%
    localizationZMultiplier = 3 # Multiplier for vertical distance in localization calculation
    measurementsToUse = measurements.getSensorsToAssimilateBjoroya()
    varyAmbient = True # Reduction in ambient values towards the rest of the farm
    o2AffinityProfile = 'vert'  # flat / vert / halfvert

    useCurrentMagic = False
    currentMagicField = r"currents_bjoroya2.nc"
    if useCurrentMagic:
        cmg = currentMagic.CurrentMagic(currentMagicField)

    includeTwin = False # If true, run last ensemble member as a twin. Twin receives no EnKF updates, and is used to provide measurements

    if dryRun:
        simNamePrefix = simNamePrefix + "dr_"

    # sinkingSpeed_perturb = 0.
    # sinkingSpeed_beta = 0.05
    # sinkingSpeed_sigma = 0.02
    # current_perturb = np.zeros((2,))
    # current_beta = 0.05
    # current_sigma = 0.025
    ambO2_perturb = 0.
    ambO2_beta = 0.05
    ambO2_sigma = 0.5
    # feedingFrac_perturb = 0.
    # feedingFrac_beta = 0.05
    # feedingFrac_sigma = 0.15
    sinkingSpeed_perturb = 0.
    sinkingSpeed_beta = 0.05
    sinkingSpeed_sigma = 0.02
    current_perturb = np.zeros((2,))
    current_beta = 0.05
    current_sigma = 0.01
    feedingFrac_perturb = 0.
    feedingFrac_beta = 0.05
    feedingFrac_sigma = 0.05


    # npar = 1 # Number of parameters to estimate in EnKF
    # paramVec = np.zeros((npar,)) # perturbation to ambO2
    # paramNoiseStd = np.zeros((npar,))
    # paramNoiseStd[0] = 0.001 # perturbation to amb02
    npar = 1 # Number of parameters to estimate in EnKF
    paramVec = np.zeros((npar,)) # perturbation to ambO2
    paramNoiseStd = np.zeros((npar,))
    paramNoiseStd[0] = 0*0.00025 # 0.001 # perturbation to amb02
    #paramNoiseStd[1] = 0.001 # perturbation to current speed

    if mpiInternal:
        comm, rank, N = mpiInternal.mpiSettings()
        print("Internal MPI rank="+str(rank)+", processes="+str(N))
        doInternalMpi = N > 1
        if doInternalMpi:
            simNamePrefix = simNamePrefix+"mpi_"
        print("doInternalMpi = " + str(doInternalMpi))
        doMpi = False
    else:
        comm, rank, N = mpi.mpiSettings()
        print("MPI rank="+str(rank)+", processes="+str(N))
        doMpi = N > 1
        print("doMpi = "+str(doMpi))

    if doMpi:
        if includeTwin:
            N = N-1 # The last process is the twin, not part of the ensemble
            if rank==N:
                perturbations = False # Disable perturbations for the twin model

    simName = simNamePrefix+str(rank).zfill(2)


    # Set up grid:
    cageDims = (math.ceil(modelDim/dxy), math.ceil(modelDim/dxy), math.ceil(depth/dz)+1)
    mask = cagemasking.circularMasking(cageDims, dxy, rad, False)
    maskAllTrue = np.array(np.zeros(cageDims), dtype=np.bool8)
    maskAllTrue[:] = True
    nstates = cageDims[0]*cageDims[1]*cageDims[2]

    print("Domain dimensions: "+str(cageDims))

    # Bounds of what x coordinates this instance of the model should handle.
    # If not internally parallelizing, it should cover the entire dimension:
    xBounds = (0, cageDims[0])
    if doInternalMpi:
        splits = mpiInternal.getSplits(cageDims[0], N)
        xBounds = splits[rank]
        print("Rank "+str(rank)+" xBounds="+str(xBounds))

    # Current:
    currentReductionFactor = 0.8 # Multiplier for inside current as function of outside
    if useCurrentMagic:
        currentReductionFactor = 1.0 # Reduction is handled by the current magic fields
    currentOffset = np.zeros((3)) # Used to globally change the current speed
    currentOffset_r = np.zeros((3))
    extCurrentSpeed = 0.12
    currentSpeed = currentReductionFactor*extCurrentSpeed
    currentDir = 3.14/2.
    currentOffset[0] = 0. #currentSpeed*math.cos(currentDir)
    currentOffset[1] = 0. #currentSpeed*math.sin(currentDir)
    currentField = np.zeros((cageDims[0]+1, cageDims[1]+1, cageDims[2]+1, 3))

    # Temperature
    T_w = 14



    # Set up vertical depths per layer (for interpolation of environmental values):
    zValues = np.zeros((cageDims[2],))
    for i in range(0,cageDims[2]):
        zValues[i] = dz*(0.5 + i)

    # Feeding setup
    # Feeder positions OF1:
    feedAngles = (300, 30, 120, 210) # Angles for feeding and O2 positions. Angle measured in degrees clockwise from north
    feedRads = (0.35, 0.53, 0.7, 0.87)
    feedDepth = 6.5 # According to OF1 report, feeding depth at 6-7 m
    nFeedPos = len(feedAngles)*len(feedRads)
    divis = float(nFeedPos)
    feedingPos = np.zeros((nFeedPos, 3), dtype=int)
    piv = 0
    for i in range(0, len(feedAngles)):
        for j in range(0, len(feedRads)):
            xDist = int(feedRads[j] * rad * math.sin(math.pi * feedAngles[i] / 180.) / dxy + cageDims[0] / 2)
            yDist = int(feedRads[j] * rad * math.cos(math.pi * feedAngles[i] / 180.) / dxy + cageDims[1] / 2)
            zDist = int(feedDepth/dz)
            #print("Feed pos " + str(piv + 1) + ": " + str(xDist) + " , " + str(yDist)+" , "+str(zDist))
            feedingPos[piv,0] = xDist
            feedingPos[piv,1] = yDist
            feedingPos[piv,2] = zDist
            piv = piv+1

    # Feeding periods:
    feedingPeriods = [10, 7200] # [27000, 63000]
    #nominalFeedingRate = 2900.*1000/(10*3600)
    nominalFeedingRate = 10.*2900. * 1000 / (10 * 3600) # TEST TEST TEST
    isFeeding = False
    feedingRateMult =0


    # Oxygen sensor positions:
    #o2Names, o2Pos, centerPos = measurements.setupSensorPositions(cageDims, dxy, dz, rad)
    o2Names, o2Pos, centerPos = measurements.setupSensorPositionsBjoroya(cageDims, dxy, dz, rad)

    # # TEST TEST TEST TEST
    # numel = cageDims[0]*cageDims[1]*cageDims[2]
    # npar = 0
    # M_all, measStd = measurements.getFieldMeasurementModel(numel, npar, cageDims, o2Pos)

    # Fish setup (N, mean weight and std.dev weight):
    # OF1:
    # nFish = 0.2e6 #1.0e6 # 1038439 smolt opprinnelig mottatt ifølge rapport til FDIR
    # meanWeight = 2540
    # Bjørøya:
    nFish = 169821
    meanWeight = 2869.5
    wFish = (meanWeight, 0.2*meanWeight) # Mean and standard deviation

    #if doMpi and rank == N:
    #    meanWeight = 4000

    # Pellet setup:
    pelletSizes = (3, 6, 9, 12)
    pelletSpeeds = (0.0773, 0.0815, 0.1284, 0.1421)
    pelletDi = 2
    pelletSize = pelletSizes[pelletDi]
    sinkingSpeed = pelletSpeeds[pelletDi]
    pelletWeight=0.2 # Pellet weight (g)

    # Diffusion:
    kappaRef = 0.00012
    kappaAdd = 0.2
    refSize = 9
    kappaZMult = 25
    diffKappa = kappaRef*(kappaAdd + math.pow(pelletSizes[pelletDi]/refSize,2))
    diffKappaZ = diffKappa*kappaZMult

    speed = math.sqrt(currentOffset[0]*currentOffset[0] + currentOffset[1]*currentOffset[1])
    diffKappaO2 = min(0.5, 10*math.pow(speed,2))
    diffKappaO2Z = diffKappaO2


    # Feed and oxygen affinity:
    #O2 affinity: Vertical distribution data based on telemetry (8 individuals):
    affProfiles = {}
    affProfiles['vert'] = [0.0110, 0.0913, 0.8601, 2.1406, 2.7774, 2.6903, 2.5195, 2.2987, 2.0137,
                    1.7448, 1.5883, 1.3667, 1.2348, 1.0724, 0.9379, 0.7764, 0.7104, 0.5895, 0.5607, 0.4668, 0.3933,
                    0.4009, 0.2935, 0.1801, 0.1260, 0.0787, 0.0457, 0.0304]
    affProfiles['halfvert'] = [0.5055, 0.5457, 0.9301, 1.5703, 1.8887, 1.8452, 1.7597, 1.6494, 1.5069,
                    1.3724, 1.2942, 1.1834, 1.1174, 1.0362, 0.9690, 0.8882, 0.8552, 0.7947, 0.7804, 0.7334, 0.6966, 0.7004,
                    0.6467, 0.5901, 0.5630, 0.5393, 0.5228, 0.5152]
    affProfiles['flat'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    affDepths = [0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 5.5000, 6.5000, 7.5000, 8.5000,
                    9.5000, 10.5000, 11.5000, 12.5000, 13.5000, 14.5000, 15.5000, 16.5000, 17.5000, 18.5000, 19.5000,
                    20.5000, 21.5000, 22.5000, 23.5000, 24.5000, 25.5000, 26.5000, 27.5000]
    affProfile = affProfiles[o2AffinityProfile]
    f = scipy.interpolate.interp1d(affDepths, affProfile, kind='linear', bounds_error=False,
                                   fill_value=(affProfile[0], affProfile[-1]))
    affinityProfile = f(zValues)
    affinity = np.zeros(cageDims)
    o2Affinity = np.zeros(cageDims)
    o2AffSum = behaviour.setO2AffinityWithVerticalProfile(cageDims, dz, fishMaxDepth, mask, affinityProfile,
                                                          affinity, o2Affinity)

    # Initialize advection routine:
    fcAdvect = advect.Advect(cageDims)
    o2Advect = advect.Advect(cageDims)
    if (varyAmbient):
        o2Advect.setVaryAmbient(True, affinityProfile)


    startTime = datetime.datetime(initYear, initMonth, initDate, initHour, initMin, initSec, tzinfo=datetime.timezone.utc)
    # Initialize inputs:
    inputsN.setStartTime(startTime)

    # Time string for storage:
    timeUnitString = "seconds since "+startTime.strftime("%Y-%m-%d %H:%M:%S")


    # Ambient values:
    ambientValueFeed = np.zeros((cageDims[2]))
    ambientValueO2 = inputsN.getO2Ambient10()*np.ones((cageDims[2]))
    ambientValueO2_orig = ambientValueO2

    # Initialize 3D fields:
    fc = np.zeros(cageDims)
    o2 = np.zeros(cageDims)
    ingDist = np.zeros(cageDims)
    o2ConsDist = np.zeros(cageDims)

    # Set initial o2 values:
    for i in range(0, cageDims[0]):
        for j in range(0, cageDims[1]):
            for k in range(0, cageDims[2]):
                o2[i,j,k] = ambientValueO2[k]

    # Setup of feeding rate based on feeding positions:
    feedingRate = np.zeros(cageDims)

    #for i in range(0, nFeedPos):
    #    feedingRate[feedingPos[i,0],feedingPos[i,1], feedingPos[i,2]] = 1./divis
    #divis = float(cageDims[0]*cageDims[1])
    center = ((cageDims[0]-1)/2.0, (cageDims[1]-1)/2.0)
    ncell = 0
    for i in range(0, cageDims[0]):
        for j in range(0, cageDims[1]):
            distX = i - center[0]
            distY = j - center[1]
            rpos = dxy * math.sqrt(distX * distX + distY * distY)
            if rpos > 0.3*rad and rpos < 0.65*rad:
                ncell = ncell+1
                feedingRate[i,j,0] = 1.
    feedingRate = feedingRate * (1/float(ncell))

    # Initialize variables:
    t = 0.
    n_steps = int(t_end/dt)
    totFeedAdded = 0.

    # Initialize fish:
    fish = simplefish.SimpleFish(nFish, wFish[0], wFish[1])

    # Initialize save files:
    if (not doInternalMpi) or rank == 0:
        ncfile = storage.init3DFile(saveDir + simName + ".nc", timeUnitString, cageDims, dxy, dz)
        storage.createCageVariables(ncfile, ('feed', 'o2', 'ingDist', 'o2consDist','currentU', 'currentV'))
        ncfile2 = storage.initScalarFile(saveDir + simName + "_fish.nc", timeUnitString, cageDims, fish.getNGroups())
        storage.createScalarVariables(ncfile2, ('totIngested', 'totIngRate', 'waste', 'rho', 'ext_O2',
                                                'center_O2', 'sinkingSpeed', 'diffKappaO2', 'feedingFrac',
                                                'o2ConsumptionRate', 'totFeed', 'feedingRate',
                                                'currentOffsetX', 'currentOffsetY'))
        storage.createScalarVariables(ncfile2, o2Names)
        storage.createGroupVariables(ncfile2, ('appetite', 'ingested'))
        storage.createProfileVariables(ncfile2, ('ext_currentU', 'ext_currentV', 'temperature'))

    if doMpi and rank==0:
        numel = cageDims[0] * cageDims[1] * cageDims[2]
        if includeTwin:
            #M, measStd = measurements.getTwinMeasurementModel(numel, npar, cageDims, mask)
            M, measStd = measurements.getFieldMeasurementModel(numel, npar, cageDims, o2Pos)
        else:
            M, measStd = measurements.getFieldMeasurementModel(numel, npar, cageDims, o2Pos)
            # Get a tuple with indexes of the sensors we should use:
            sensorsToUse = measurements.getSensorsToAssimilateBjoroya()
            # Pick out those rows of M_all corresponding to the sensors we should use:
            M = M[sensorsToUse, :]

        ensNcFile = storage.initEnsembleFile(saveDir+simName+"_ensemble.nc", nstates, npar, N, cageDims, M.shape[0])
    else:
        ensNcFile = None

    stime = current_milli_time()

    # Main simulation loop:
    for i in range(0, n_steps):

        if useNetcdfInputs:
            isNewData = inputsN.advance(t)
            if i==0 or isNewData: # Update environmental values from NetCDF data

                inpDepths = [5., 10., 15.]
                # Temperature:
                tempValues = [inputsN.getTemperature5(), inputsN.getTemperature10(),
                              inputsN.getTemperature15()]
                f = scipy.interpolate.interp1d(inpDepths, tempValues, kind='linear', bounds_error=False,
                                               fill_value=(tempValues[0], tempValues[2]))
                T_w = f(zValues)

                # Ambient O2:
                #ambientValueO2[:] = inputsN.getO2Ambient()
                o2Values = [inputsN.getO2Ambient5(), inputsN.getO2Ambient10(),
                            inputsN.getO2Ambient15()]
                f = scipy.interpolate.interp1d(inpDepths, o2Values, kind='linear', bounds_error=False,
                                               fill_value=(o2Values[0], o2Values[2]))
                ambientValueO2 = f(zValues)
                #print("rank="+str(rank)+": "+str(ambientValueO2))

                # Current:
                extCurrentSpeed = inputsN.getCurrentSpeed()
                currentSpeed0 = currentReductionFactor*extCurrentSpeed
                currentDir0 = inputsN.getCurrentDir()*math.pi/180.
                currentX0 = np.multiply(np.sin(currentDir0), currentSpeed0)
                currentY0 = np.multiply(np.cos(currentDir0), currentSpeed0)
                # Interpolate to our layers:
                f = scipy.interpolate.interp1d(inputsN.getCurrentDepths(), currentX0, kind='linear', bounds_error=False,
                                               fill_value=(currentX0[0], currentX0[-1]))
                currentX = f(zValues)
                f = scipy.interpolate.interp1d(inputsN.getCurrentDepths(), currentY0, kind='linear', bounds_error=False,
                                               fill_value=(currentY0[0], currentY0[-1]))
                currentY = f(zValues)

                if not useCurrentMagic:
                    currentfields.getProfileCurrentField(currentField, currentX, currentY)
                else:
                    currentSpeeds = np.sqrt(np.multiply(currentX, currentX)+np.multiply(currentY, currentY))
                    currentDirs = np.arctan2(currentX, currentY)*180./np.pi
                    cmg.setCurrentField(currentField, currentSpeeds, currentDirs)

                #if doMpi and rank == N:  # If this is the twin model
                #    currentDirNow += (math.pi/180.)*30.  #  degrees offset
                #    currentSpeedNow *= 1.2  # 20% increase
                #    ambientValueO2[:] += 0.1

                # Current speed-dependent O2 diffusion:
                #diffKappaO2 = min(0.5, 10 * math.pow(currentSpeed, 2))
                diffKappaO2 = min(0.5, 10 * math.pow(currentReductionFactor*0.06, 2))
                diffKappaO2Z = 5.0*0.1*diffKappaO2

                # Update feeding rate depending on preset feeding periods:
                if not isFeeding: # Not already feeding. Check if we should start:
                    if t >= feedingPeriods[0] and t<feedingPeriods[1]:
                        isFeeding = True
                        feedingRateMult = nominalFeedingRate
                    else:
                        feedingRateMult = 0
                else: # Already feeding. Check if we should stop:
                    if t >= feedingPeriods[1]:
                        isFeeding = False
                        feedingRateMult = 0
                    else:
                        feedingRateMult = nominalFeedingRate

        else:
            currentDir = currentDir + 0.01*np.random.normal()
            currentDirNow = currentDir
            currentSpeedNow = currentSpeed
            if doMpi and rank == N:  # If this is the twin model
                currentDirNow += 3.14/12. # 15 degrees offset
                currentSpeedNow *= 1.2 # 20% increase
            currentOffset[0] = currentSpeedNow * math.sin(currentDir)
            currentOffset[1] = currentSpeedNow * math.cos(currentDir)
            diffKappaO2 = min(0.5, 10 * math.pow(speed, 2)) # Values ca. 0.5 - 1.6
            diffKappaO2Z = diffKappaO2

        # if doMpi and rank == N:  # If this is the twin model
        #     if t < 3600:
        #         ambientValueO2 = ambientValueO2_orig + 1. #(1. + 0.2 * math.cos(2 * math.pi * t / 7200.))
        #     else:
        #         ambientValueO2 = ambientValueO2_orig - 1.
        #     if t < 3600:
        #         diffKappaO2_offset = 0.3
        #     else:
        #         diffKappaO2_offset = 0.1
        diffKappaO2_offset = 0.

        if perturbations:

            # Add noise to estimated parameters:
            for j in range(0, npar):
                paramVec[j] = paramVec[j] + dt*paramNoiseStd[j]*np.random.normal()

            # Update other random processes:
            if i==0: # Initialize random processes
                sinkingSpeed_perturb = sinkingSpeed_sigma*np.random.normal()
                for jj in range(0, 2):
                    current_perturb[jj] = current_sigma*np.random.normal()
                ambO2_perturb = ambO2_sigma*np.random.normal()
                feedingFrac_perturb = feedingFrac_sigma * np.random.normal()
            else: # Update perturbations
                sinkingSpeed_perturb = gaussmarkov.updateGaussMarkov(sinkingSpeed_perturb, sinkingSpeed_beta,
                                                                     sinkingSpeed_sigma, dt)
                for jj in range(0,2):
                    current_perturb[jj] = gaussmarkov.updateGaussMarkov(current_perturb[jj], current_beta, current_sigma, dt)
                ambO2_perturb = gaussmarkov.updateGaussMarkov(ambO2_perturb, ambO2_beta, ambO2_sigma, dt)
                feedingFrac_perturb = gaussmarkov.updateGaussMarkov(feedingFrac_perturb, feedingFrac_beta,
                                                                    feedingFrac_sigma, dt)

            sinkingSpeed_r = max(0.001, sinkingSpeed + sinkingSpeed_perturb)
            currentOffset_r[0:2] = currentOffset[0:2] + current_perturb[:]
            if npar>1: # Second parameter is current speed adjustment
                currentOffset_r[0:2] = currentOffset_r[0:2]*(1.0 + paramVec[1])
            #ambientValueO2_r = ambientValueO2 + ambO2_perturb
            ambientValueO2_r = ambientValueO2 + ambO2_perturb + paramVec[0]
            diffKappaO2_r = diffKappaO2
            diffKappaO2Z_r = diffKappaO2
            feedingFrac_r = max(-0.2, min(.2, feedingFrac_perturb))

        else:
            sinkingSpeed_r = sinkingSpeed
            currentOffset_r[:] = currentOffset[:]
            ambientValueO2_r = ambientValueO2
            diffKappaO2_r = diffKappaO2
            diffKappaO2Z_r = diffKappaO2Z
            feedingFrac_r = 0.

        if doMpi and includeTwin and rank == N:  # If this is the twin model
            diffKappaO2_r = diffKappaO2 + diffKappaO2_offset
        else:
            diffKappaO2_r = diffKappaO2_r
        diffKappaO2_r = max(1e-3, diffKappaO2_r)

        # Update total feed added:
        totFeedAdded = totFeedAdded + dt*feedingRateMult

        # Advect fields:
        if advectFeed:
            currentTotalFeed = np.sum(fc)
            if feedingRateMult > 0 or currentTotalFeed > feedAdvectLimit:
                lostAll = fcAdvect.advectField(cageDims, xBounds, fc, dt, dxy, dz, mask, sinkingSpeed_r, diffKappa, diffKappaZ,
                                               currentField, currentOffset_r, feedingRate, feedingRateMult, ambientValueFeed)
            else:
                lostAll = 0.
            #print("Rank="+str(rank)+": "+str(lostAll))
        else:
            lostAll = 0.

        o2Advect.advectField(cageDims, xBounds, o2, dt, dxy, dz, mask, 0., diffKappaO2_r, diffKappaO2Z_r,
                                       currentField, currentOffset_r, feedingRate, 0., ambientValueO2_r)

        if doInternalMpi:
            lostFeedTotal = mpiInternal.syncToRank0(comm, rank, N, cageDims, xBounds, fc, o2, lostAll)
            if rank==0:
                lostAll += lostFeedTotal
                #print("lostFeedTotal = "+str(lostFeedTotal))

        if (not doInternalMpi) or rank==0:
            totalIntake, rho, o2ConsumptionRate = fishingestionTempProfile.calculateIngestion\
                (cageDims, dt, fc, o2, affinity, o2Affinity, o2AffSum,
                 ingDist, o2ConsDist, dxy, dz, mask,
                 pelletWeight, T_w, fish, perturbations, feedingFrac_r)

        if doInternalMpi:
            mpiInternal.distFromRank0(comm, rank, N, cageDims, xBounds, fc, o2)

        # Storage:
        if (not doInternalMpi) or rank==0:
            if i>0 and (t/(float(storeIntervalScalars)) - math.floor(t/float(storeIntervalScalars))) < 1e-5:
            #if i>0 and (t % storeIntervalScalars) < dt:
                storage.saveScalarVariable(ncfile2, t, 'totIngRate', totalIntake, True)
                storage.saveScalarVariable(ncfile2, t, 'waste', lostAll, False)
                storage.saveScalarVariable(ncfile2, t, 'rho', rho, False)
                totI = 0.
                ingested = np.zeros((fish.getNGroups()))
                appetite = np.zeros((fish.getNGroups()))
                for j in range(0, fish.getNGroups()):
                    ingested[j] = fish.getIngested(j)
                    totI += fish.getN(j)*ingested[j]
                    appetite[j] = fish.getAppetite(j)

                storage.saveScalarVariable(ncfile2, t, 'totIngested', totI, False)
                storage.saveGroupVariable(ncfile2, t, 'ingested', ingested, False)
                storage.saveGroupVariable(ncfile2, t, 'appetite', appetite, False)

                totalFeed = 0.
                for i in range(0, cageDims[0]):
                    for j in range(0, cageDims[1]):
                        for k in range(0, cageDims[2]):
                            totalFeed += fc[i,j,k]
                storage.saveScalarVariable(ncfile2, t, 'totFeed', totalFeed, False)
                storage.saveScalarVariable(ncfile2, t, 'feedingRate', feedingRateMult, False)
                storage.saveScalarVariable(ncfile2, t, 'ext_O2', ambientValueO2_r, False)
                storage.saveScalarVariable(ncfile2, t, 'o2ConsumptionRate', o2ConsumptionRate, False)
                #storage.saveScalarVariable(ncfile2, t, 'ext_currentSpeed', math.sqrt(currentOffset_r.T @ currentOffset_r), False)
                #storage.saveScalarVariable(ncfile2, t, 'ext_currentDir', (180/math.pi)*math.atan2(currentOffset_r[0], currentOffset_r[1]), False)
                storage.saveProfileVariable(ncfile2, t, 'ext_currentU', currentX, False)
                storage.saveProfileVariable(ncfile2, t, 'ext_currentV', currentY, False)
                storage.saveScalarVariable(ncfile2, t, 'currentOffsetX', currentOffset_r[0], False)
                storage.saveScalarVariable(ncfile2, t, 'currentOffsetY', currentOffset_r[1], False)
                storage.saveScalarVariable(ncfile2, t, 'sinkingSpeed', sinkingSpeed_r, False)
                storage.saveScalarVariable(ncfile2, t, 'diffKappaO2', diffKappaO2_r, False)
                storage.saveScalarVariable(ncfile2, t, 'feedingFrac', feedingFrac_r, False)
                storage.saveProfileVariable(ncfile2, t, 'temperature', T_w, False)

                storage.saveScalarVariable(ncfile2, t, 'center_O2',
                                           o2[int(centerPos[0]), int(centerPos[1]), int(centerPos[2])], False)
                for j in range(0, len(o2Names)):
                    storage.saveScalarVariable(ncfile2, t, o2Names[j], o2[int(o2Pos[j,0]), int(o2Pos[j,1]), int(o2Pos[j,2])], False)

                storage.syncFile(ncfile2)

        if (not doInternalMpi) or rank == 0:
            if i>0 and (t/(float(storeIntervalFields)) - math.floor(t/float(storeIntervalFields))) < 1e-5:
            #if i>0 and (math.floor(t % storeIntervalFields)) < dt:
                print("Saving 3D fields. t="+str(t))
                storage.saveCageVariable(ncfile, t, 'feed', fc, True, maskAllTrue, True)
                storage.saveCageVariable(ncfile, t, 'o2', o2, True, maskAllTrue, False)
                storage.saveCageVariable(ncfile, t, 'ingDist', ingDist, True, mask, False)
                storage.saveCageVariable(ncfile, t, 'o2consDist', o2ConsDist, True, mask, False)
                storage.saveCageVariable(ncfile, t, 'currentU', currentField[:-1,:-1,:-1,0], True, mask, False)
                storage.saveCageVariable(ncfile, t, 'currentV', currentField[:-1, :-1, :-1, 1], True, mask, False)
                storage.syncFile(ncfile)

        # Ensemble Kalman Filter
        if doMpi:
            if i > 0 and (t % enKFInterval) < dt:
                print("Calling ens.: t = "+str(t))

                updO2, updParamVec = ensembleKalmanFilter.doAnalysis(comm, rank, N, cageDims, dxy, dz, rad,
                                                                     o2, mask, t, ensNcFile, localization,
                                                                     includeTwin, paramVec, simInputs=inputsN,
                                                                     currentOffset=currentOffset, localizationDist=localizationDistM/dxy,
                                                                     localizationZMultiplier=localizationZMultiplier,
                                                                     ensembleInflation=ensembleInflation, ensembleInflationFactor=ensembleInflationFactor)
                print(str(rank)+": before: "+str(paramVec[0])+"   after: "+str(updParamVec[0]))
                # Check if this is a dry run:
                doUpdate = not dryRun
                # Check if we have passed a cutoff time:
                if assimCutoff:
                    if t >= assimCutoffTime:
                        doUpdate = False
                if doUpdate:
                    o2[...] = updO2[...]
                    # Update estimated parameters:
                    for j in range(0, npar):
                        paramVec[j] = updParamVec[j]

        #else:
            #ensembleKalmanFilter.doTest(cageDims, o2, t)


        if rank==0 and i>0 and i % txtUpdateInterval == 0:
            elapsed = (current_milli_time() - stime) / 60000.
            fractionCompleted = float(i)/float(n_steps)
            remaining = (elapsed / fractionCompleted) - elapsed
            print("t = " + str(t) + " - Estimated time to complete: " + format(remaining, ".2f") + " minutes")

        t = t + dt

    if rank==0:
        totI = 0.
        for j in range(0, fish.getNGroups()):
            totI += fish.getN(j) * fish.getIngested(j)
        print("TotFeedAdded = "+format(totFeedAdded))
        print("Total ingestion: "+format(totI, ".1f"))
        print("Feed wastage: "+format(100.*(totFeedAdded-totI)/totFeedAdded, ".2f")+"%")