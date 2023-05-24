import netCDF4 as nc
import numpy as np

class CurrentMagic:

    uval = []
    vval = []
    thetas = []

    def __init__(self, filePath):
        ncfile = nc.Dataset(filePath, "r")
        self.uval = np.transpose(ncfile.variables['u'][:],(2,1,0))
        self.vval = np.transpose(ncfile.variables['v'][:],(2,1,0))
        self.thetas = ncfile.variables['theta'][:]

        #print(self.uval.shape)


    def setCurrentField(self, field, speeds, angles):
        for k in range(0, len(angles)):
            fieldI = self.chooseField(angles[k])
            #print(str(angles[k])+": "+str(fieldI)+", "+str(self.thetas[fieldI]))
            field[:,:,k,0] = self.uval[:,:,fieldI]*speeds[k]
            field[:,:,k,1] = self.vval[:,:,fieldI]*speeds[k]




    # Given a theta angle, choose the index for the current field that best matches the angle.
    def chooseField(self, theta):
        #print("theta (internal prior)="+str(theta))
        while (theta < 0):
            theta += 360
        while (theta >= 360):
            theta -= 360
        #print("theta (internal)="+str(theta))
        # Find closest value in thetas:
        best = 9999.
        bestI = -1
        for i in range(0, self.thetas.shape[0]):
            if (abs(theta-self.thetas[i]) < best):
                bestI = i
                best = abs(theta-self.thetas[i])

        return bestI

