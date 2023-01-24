import numpy as np
import numpy.linalg
import math
import mpi
import storage
import measurements


def doAnalysis(comm, rank, N, cageDims, dxy, dz, rad, field, mask, t, ensNcFile, localization, includeTwin, paramVec,
               simInputs=None, currentOffset=0, localizationDist=10., localizationZMultiplier=1.,
               ensembleInflation=False, ensembleInflationFactor=1.01):


    updParamVec = np.zeros(paramVec.shape)
    updParamVec[:] = paramVec[:]

    numel = cageDims[0]*cageDims[1]*cageDims[2]
    npar = paramVec.shape[0]

    if rank==0: # I should collect all states

        # Initialize ensemble state:
        X = np.zeros((numel+npar, N))
        # Set my own state:
        X[0:numel,0] = np.reshape(field, (numel,))
        X[numel:,0] = paramVec[:]

        # Set up mask vector to store:
        maskVec = np.zeros((numel+npar, 1))
        maskVec[0:numel,0] = np.reshape(mask, (numel,))

        # There are two main modes of operation. If "includeTwin" is True, we are obtaining measurements
        # from a "twin" model running parallel to the ensemble. The set of states to measure is decided
        # by the measurement model provided by the measurements module.
        # If "includeTwin" is False, we are using measurements loaded from file by the simInputsNetcdf module.
        if includeTwin:

            # Get measurement model:
            #M_all, measStd = measurements.getTwinMeasurementModel(numel, npar, cageDims, mask)
            ## Oxygen sensor positions:
            o2Names, o2Pos, centerPos = measurements.setupSensorPositions(cageDims, dxy, dz, rad)
            M_all, measStd = measurements.getFieldMeasurementModel(numel, npar, cageDims, o2Pos)

            # Receive from the twin:
            data = mpi.recvStateVector(comm, (numel+npar,), N, N)
            X_twin = np.reshape(data, (numel+npar,1))
            # Make measurements based on the twin model:
            d = M_all @ X_twin

            # Set up ensemble measurement matrix:
            D_all = np.zeros((M_all.shape[0], N))
            D_exact = np.zeros((M_all.shape[0], N))
            for j in range(0, N):
                D_all[:, j] = d[:, 0] + measStd * np.random.normal(size=(M_all.shape[0],))
                D_exact[:, j] = d[:, 0]



        #print(D)

        else:
            # Measurements are obtained from the simInputsNetcdf module.
            X_twin = None

            # Oxygen sensor positions:
            o2Names, o2Pos, centerPos = measurements.setupSensorPositionsBjoroya(cageDims, dxy, dz, rad)
            M_all, measStd = measurements.getFieldMeasurementModel(numel, npar, cageDims, o2Pos)
            # Get a tuple with indexes of the sensors we should use:
            sensorsToUse = measurements.getSensorsToAssimilateBjoroya()
            # Pick out those rows of M_all corresponding to the sensors we should use:
            M_all = M_all[sensorsToUse,:]

            # Get actual measurements:
            allSensors = simInputs.getO2Measurements()
            d = np.reshape(allSensors, (allSensors.shape[0], 1))
            print(d)
            # Pick out measurements corresponding to the sensors we should use:
            d = d[sensorsToUse,:]
            print(d)
            # Set up ensemble measurement matrix:
            D_all = np.zeros((M_all.shape[0], N))
            D_exact = np.zeros((M_all.shape[0], N))
            for j in range(0, N):
                D_all[:,j] = d[:, 0] + measStd * np.random.normal(size=(M_all.shape[0],))
                D_exact[:, j] = d[:, 0]



        M = M_all
        D = D_all

        if localization:
            #Xloc1, Xloc2 = getLocalizationMatrix(cageDims, M, numel, 4)
            Xloc1, Xloc2 = getLocalizationMatrix(cageDims, M, numel, localizationDist, localizationZMultiplier, currentOffset)

        # Receive from other ensemble members:
        for j in range(1, N):
            data = mpi.recvStateVector(comm, (numel+npar,), j, j)
            X[:, j] = data[:]

        # Compute mean state:
        X_mean = np.mean(X, axis=1)
        X_mean = np.reshape(X_mean, (numel+npar,1))
        E_X = X_mean @ np.ones((1,N))
        theta = X - E_X

        # Measurement error covariance matrix:
        R = np.diag(measStd*measStd*np.ones((M.shape[0],)))

        N1 = max(N-1.,1.)

        # Compute intermediary matrixes:
        MX = M @ X
        omega = M @ theta
        if localization:
            #print(Xloc2)
            #print(omega @ omega.T)
            #print(Xloc2 * (omega @ omega.T))
            phi = (1./N1)*(omega @ omega.T) + R
            #phi = (1 / N1) * (Xloc2 * (omega @ omega.T)) + R
        else:
            phi = (1./N1)*(omega @ omega.T) + R
        phi_inv = np.linalg.inv(phi)

        deviations = D-MX
        dev_exact = D_exact-MX

        if localization:
            #K = np.zeros((numel, M.shape[0]))
            #for j in range(0, numel):
            K = Xloc1 * (1./N1) * (theta @ omega.T @ phi_inv)
            #K = theta @ omega.T @ phi_inv
        else:
            K = (1./N1) * theta @ omega.T @ phi_inv
        X_a = X + K @ deviations

        # Ensemble inflation:
        if ensembleInflation:
            # Compute mean of analysis:
            X_mean = np.reshape(np.mean(X_a, axis=1), (numel + npar, 1))
            E_X = X_mean @ np.ones((1, N))
            X_a = ensembleInflationFactor*(X_a - E_X) + E_X

        # Save ensemble state to file:
        storage.saveEnsembleState(ensNcFile, t, maskVec, X, X_a, X_twin, M_all, deviations, dev_exact, K, Xloc1, theta, omega, True)
        storage.syncFile(ensNcFile)

        # Send updated states to other ensemble members:
        for j in range(1, N):
            data = X_a[:,j].copy()
            #print("Sending updated state to: "+str(j))
            mpi.sendStateVector(comm, data, j, j)

        # Extract own updated state and return:
        updField = np.reshape(X_a[0:numel,0], cageDims)
        updParamVec = X_a[numel:,0]
        return updField, updParamVec
    elif includeTwin and rank==N: # I am the twin
        # Send my state:
        data = np.zeros((numel + npar,))
        data[0:numel] = np.reshape(field, (numel,))
        data[numel:] = paramVec
        mpi.sendStateVector(comm, data, rank, 0)
        #print("Twin sent state")
        return field, paramVec # No updates for the twin

    else: # I am a common ensemble member
        # Send my state:
        data = np.zeros((numel+npar,))
        data[0:numel] = np.reshape(field, (numel,))
        data[numel:] = paramVec
        mpi.sendStateVector(comm, data, rank, 0)
        print("enKF rank "+str(rank)+": "+str(paramVec[0]))
        # Receive updated state:
        data = mpi.recvStateVector(comm, (numel+npar,), rank, 0)
        #print(str(rank)+" received updated state")
        updField = np.reshape(data[0:numel], cageDims)
        updParamVec = data[numel:]
        print("enKF rank " + str(rank) + ": " + str(updParamVec[0]))
        return updField, updParamVec


def computeCovariance(X, nstates, N, byState):
    res = np.zeros((nstates,))
    myVals = X[byState,:].T
    for i in range(0, nstates):
        otherVals = X[i,:].T
        matr = np.concatenate(myVals, otherVals)
        cov = np.cov(matr)


def getLocalizationMatrix(cageDims, M, numel, locDist, locZMultiplier, currentOffset):
    Xloc1 = np.zeros((M.shape[1], M.shape[0]))
    Xloc2 = np.zeros((M.shape[0], M.shape[0]))
    for i in range(M.shape[0]): # Loop over measurements
        # Find index of this measurement:
        mInd = 0
        while (M[i,mInd] == 0):
            mInd = mInd+1
        mCoord = np.unravel_index(mInd, cageDims)

        # Set values for Xloc1:
        for j in range(M.shape[1]): # Loop over state variables
            if j < numel:
                sCoord = np.unravel_index(j, cageDims)
                distance = math.sqrt(math.pow(mCoord[0]-sCoord[0], 2) + math.pow(mCoord[1]-sCoord[1], 2) +
                                     math.pow(locZMultiplier*(mCoord[2] - sCoord[2]), 2))
                Xloc1[j,i] = localizationValue(distance, locDist)

            else:
                ## No localization for parameter values:
                Xloc1[j,i] = 1.

                # # Test localization in x and y direction based on current direction in each component:
                # if currentOffset[0] > 0:
                #     edgeDistX = mCoord[0]
                # else:
                #     edgeDistX = cageDims[0]-1-mCoord[0]
                # if currentOffset[1] > 0:
                #     edgeDistY = mCoord[1]
                # else:
                #     edgeDistY = cageDims[1]-1-mCoord[1]
                # Xloc1[j, i] = localizationValue(min((edgeDistX, edgeDistY)), locDist)

                # Test "close to outer edge" localization for parameter (maybe suitable for ambient o2 value):
                #edgeDist = min((mCoord[0], mCoord[1], cageDims[0]-1-mCoord[0], cageDims[1]-1-mCoord[1]))
                #Xloc1[j, i] = localizationValue(edgeDist, locDist)

        # Set values for Xloc2:
        for j in range(M.shape[0]):  # Loop over measurements
            # Find index of this measurement:
            mInd2 = 0
            while (M[j, mInd2] == 0):
                mInd2 = mInd2 + 1
            mCoord2 = np.unravel_index(mInd2, cageDims)
            distance = math.sqrt(math.pow(mCoord[0] - mCoord2[0], 2) + math.pow(mCoord[1] - mCoord2[1], 2) +
                                 math.pow(locZMultiplier*(mCoord[2] - sCoord[2]), 2))
            Xloc2[j, i] = localizationValue(distance, locDist)


    return Xloc1, Xloc2

def localizationValue(distance, locDist):
    locFac = 0.5 #4.0
    return 1-1/(1+math.exp(-locFac*(distance-locDist)))

def doTest(cageDims, field, t):
    numel = cageDims[0] * cageDims[1] * cageDims[2]
    print(numel)
    #print("field[6,6,0] = "+str(field[6,6,0]))
    #print("field[6,6,2] = " + str(field[6, 6, 2]))
    X = np.reshape(field, (numel,))
    #sub = np.ravel_multi_index((6,6,0), cageDims)
    #print(sub)
    #print("data[314] = " + str(data[sub]))
    #print("data[314] = "+str(data[314]))
    M = measurements.getTwinMeasurementModel(numel, cageDims)
    MX = M @ X
    print(MX)


