import netCDF4 as nc
import datetime
import numpy as np

class SimInputsNetcdf:

    sTime = 0 # Start time of simulation, timestamp in seconds
    times = []
    currentSpeeds = []
    currentDirs = []
    currentDepths = []
    o2ambVal5 = []
    o2ambVal10 = []
    o2ambVal15 = []
    feedingVal = []
    temperatures5 = []
    temperatures10 = []
    temperatures15 = []
    #o2sensorC1 = []
    #o2sensorC4 = []
    #o2sensorC7 = []
    #o2sensorC10 = []
    o2Sensors = []
    nSensors = -1
    piv = 0

    def __init__(self, filename):
        ncfile = nc.Dataset(filename, "r")
        self.times = ncfile.variables['time'][:]
        firstTime = self.getDatetime(self.times[0])
        self.currentSpeeds = ncfile.variables['extCurrentSpeed'][:]
        self.currentDirs = ncfile.variables['extCurrentDir'][:]
        self.currentDepths = ncfile.variables['zc'][:]
        self.o2ambVal5 = ncfile.variables['O2ambient_5'][:]
        self.o2ambVal10 = ncfile.variables['O2ambient_10'][:]
        self.o2ambVal15 = ncfile.variables['O2ambient_15'][:]
        #self.feedingVal = ncfile.variables['feedingBitfield'][:]
        self.temperatures5 = ncfile.variables['temperature_5'][:]
        self.temperatures10 = ncfile.variables['temperature_10'][:]
        self.temperatures15 = ncfile.variables['temperature_15'][:]
        #self.o2sensorC1 = ncfile.variables['O2_C1'][:]
        #self.o2sensorC4 = ncfile.variables['O2_C4'][:]
        #self.o2sensorC7 = ncfile.variables['O2_C7'][:]
        #self.o2sensorC10 = ncfile.variables['O2_C10'][:]
        self.o2Sensors = ncfile.variables['O2allSensors'][:]
        self.nSensors = self.o2Sensors.shape[1]

    def setStartTime(self, startTime):
        # Search for the first relevant time step in the data series, and update piv accordingly.
        self.sTime = startTime.timestamp()
        while self.piv < len(self.times) and self.sTime - 86400*self.times[self.piv] > 1:
            self.piv = self.piv + 1

        print("Current speed: "+str(self.currentSpeeds[self.piv]))
        print("Current dir: " + str(self.currentDirs[self.piv]))
        print("Ambient o2 5m: " + str(self.o2ambVal5[self.piv]))
        #print("Feeding: " + str(self.feedingVal[self.piv]))
        print("Temperature 5m: " + str(self.temperatures5[self.piv]))

    def advance(self, t):
        # Check if it is time to advance to next data point:
        oldPiv = self.piv
        newTime = self.sTime + t
        while self.piv < len(self.times) and newTime - 86400*self.times[self.piv] > 0:
            self.piv = self.piv + 1
        #print("Measurement PIV = "+str(self.piv))
        return self.piv > oldPiv

    #def getTemperature(self):
    #    return self.temperatures[self.piv]

    def getTemperature5(self):
        return self.temperatures5[self.piv]

    def getTemperature10(self):
        return self.temperatures10[self.piv]

    def getTemperature15(self):
        return self.temperatures15[self.piv]


    def getCurrentSpeed(self):
        return self.currentSpeeds[self.piv,:]

    def getCurrentDir(self):
        return self.currentDirs[self.piv,:]

    def getCurrentDepths(self):
        return self.currentDepths[:]

    #def getO2Ambient(self):
    #    return self.o2ambVal[self.piv]

    def getO2Ambient5(self):
        return self.o2ambVal5[self.piv]

    def getO2Ambient10(self):
        return self.o2ambVal10[self.piv]

    def getO2Ambient15(self):
        return self.o2ambVal15[self.piv]

    def getFeedingValue(self):
        return self.feedingVal[self.piv]

    def getO2Measurements(self):
        meas = np.zeros((self.nSensors,))
        meas[:] = self.o2Sensors[self.piv,:]
        return meas

    # def getO2Measurements(self):
    #     meas = np.zeros((4,))
    #     meas[0] = self.o2sensorC1[self.piv]
    #     meas[1] = self.o2sensorC4[self.piv]
    #     meas[2] = self.o2sensorC7[self.piv]
    #     meas[3] = self.o2sensorC10[self.piv]
    #     return meas

    def getDatetime(self, value):
        return datetime.datetime.fromtimestamp(round(86400*value), datetime.timezone.utc)