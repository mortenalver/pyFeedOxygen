import numpy as np
import math

# Set up measurement model.
def getTwinMeasurementModel(numel, npar, cageDims, mask):
    # Measurement uncertainty:
    measStd = 0.025
    # Measuring lots of cells in all layers:
    coords = []
    step = 3
    #k = int(cageDims[2]/3)
    for i in range(0,cageDims[0], step):
        for j in range(0, cageDims[1], step):
            for ll in range(0, cageDims[2], step):
                if mask[i,j,ll]:
                    coords.append((i,j,ll))
    nmeas = len(coords)
    print(coords)
    M = np.zeros((nmeas, numel+npar))
    for i in range(0, nmeas):
        ind = coords[i]
        sub = np.ravel_multi_index(ind, cageDims)
        M[i,sub] = 1
    return M, measStd

    # Test: measuring in all center cells:
    # nmeas = cageDims[2]
    # center = (int(cageDims[0]/2), int(cageDims[1]/2))
    # M = np.zeros((nmeas, numel))
    # for i in range(0, nmeas):
    #     ind = (center[0], center[1], i)
    #     sub = np.ravel_multi_index(ind, cageDims)
    #     M[i,sub] = 1
    # return M


def getFieldMeasurementModel(numel, npar, cageDims, o2Pos):
    # Measurement uncertainty:
    measStd = 5*0.025
    nmeas = o2Pos.shape[0]
    meas0 = 0
    M = np.zeros((nmeas, numel + npar))
    for i in range(0, nmeas):
        sub = np.ravel_multi_index((int(o2Pos[i+meas0,0]),int(o2Pos[i+meas0,1]),int(o2Pos[i+meas0,2])), cageDims)
        M[i, sub] = 1

    return M, measStd

def getSensorsToAssimilateBjoroya():
    # Return a tuple containing indexes of those oxygen sensors we want to use for assimilation:
    #return (0, 1, 2) # Centre only
    #return (0, 3, 6, 9) # All at 5 m
    #return (1, 4, 7, 10)  # All at 10 m
    #return (4, 7, 10)  # Ring measurements at 10 m
    return (0, 1, 2, 4, 7, 10)  # All at 10 m and all in centre
    #return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) # All sensors

def setupSensorPositionsBjoroya(cageDims, dxy, dz, rad):
    # Oxygen sensor positions:
    o2Names = ("C_5", "C_10", "C_15", "M1_5", "M1_10", "M1_15", "M2_5", "M2_10", "M2_15",
            "M3_5", "M3_10", "M3_15")
    # angles: M1 128.2948, M2 2.8445, M3 246.8427
    angle1 = 128.2948
    angle2 = 2.8445
    angle3 = 246.8427
    sensorAngles = (0, 0, 0, angle1, angle1, angle1, angle2, angle2, angle2,
            angle3, angle3, angle3)
    o2Rad = (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1) # Distance from centre as fraction of cage radius
    o2Depth = (5, 10, 15, 5, 10, 15, 5, 10, 15, 5, 10, 15) # Sensor depth (m)
    o2Pos = np.zeros((len(o2Names), 3))
    for i in range(0, len(o2Names)):
        xDist = (int)(o2Rad[i] * rad * math.sin(math.pi * sensorAngles[i] / 180.) / dxy + cageDims[0] / 2)
        yDist = (int)(o2Rad[i] * rad * math.cos(math.pi * sensorAngles[i] / 180.) / dxy + cageDims[1] / 2)
        zDist = (int)(o2Depth[i] / dz)
        print(o2Names[i] + ": " + str(xDist) + " , " + str(yDist) + " , " + str(zDist))
        o2Pos[i][0] = xDist
        o2Pos[i][1] = yDist
        o2Pos[i][2] = zDist
    centerPos = (int(cageDims[0] / 2), int(cageDims[1] / 2), (int)(12 / dz))
    return o2Names, o2Pos, centerPos

def setupSensorPositions(cageDims, dxy, dz, rad):
    # Oxygen sensor positions:
    o2Names = ("C", "C1", "C4", "C7", "C10", "C1_3", "C4_3", "C7_3", "C10_3")
    sensorAngles = (0, 300, 30, 120, 210, 300, 30, 120, 210)
    o2Rad = (0, 1, 1, 1, 1, 1, 1, 1, 1)  # Distance from centre as fraction of cage radius
    #o2Rad = (0, 0.65, 0.65, 0.65, 0.65, 1, 1, 1, 1)  # Distance from centre as fraction of cage radius
    o2Depth = (12, 12, 12, 12, 12, 22, 22, 22, 22)  # Sensor depth (m)
    o2Pos = np.zeros((len(o2Names), 3))
    for i in range(0, len(o2Names)):
        xDist = (int)(o2Rad[i] * rad * math.sin(math.pi * sensorAngles[i] / 180.) / dxy + cageDims[0] / 2)
        yDist = (int)(o2Rad[i] * rad * math.cos(math.pi * sensorAngles[i] / 180.) / dxy + cageDims[1] / 2)
        zDist = (int)(o2Depth[i] / dz)
        print(o2Names[i] + ": " + str(xDist) + " , " + str(yDist) + " , " + str(zDist))
        o2Pos[i][0] = xDist
        o2Pos[i][1] = yDist
        o2Pos[i][2] = zDist
    centerPos = (int(cageDims[0] / 2), int(cageDims[1] / 2), (int)(12 / dz))

    return o2Names, o2Pos, centerPos