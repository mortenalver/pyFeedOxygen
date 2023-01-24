import numpy as np

# Super individual representation of salmon (Alver et al., 2004) with weight, numbers and stomach contents.

class SimpleFish:

    # Precomputed grouping parameters for 7 groups with boundaries at +-0.333 STD, +-1.0 STD and +-2.0 STD:
    nGroups = 7
    Ndist = (0.0228, 0.1359, 0.2120, 0.2586, 0.2120, 0.1359, 0.0228)
    wDev = (-2.3577, -1.374, -0.6341, 0, 0.6341, 1.374, 2.3577)

    a_1 = 5.2591e-6
    a_2 = 0.7639 # Stomach evacuation parameters.

    #double[] N, weight, V, ingested, ingRate;

    def __init__(self, Nfish, meanWeight, stdevWeight):
        self.N = np.zeros((self.nGroups))
        self.weight = np.zeros((self.nGroups))
        self.V = np.zeros((self.nGroups))
        self.ingested = np.zeros((self.nGroups))
        self.ingRate = np.zeros((self.nGroups))
        for i in range(0, self.nGroups):
            self.N[i] = Nfish*self.Ndist[i]
            self.weight[i] = meanWeight + self.wDev[i]*stdevWeight


    def getNGroups(self):
        return self.nGroups


    def getTotalN(self):
        sum = 0.
        for i in range(0, len(self.N)):
            sum += self.N[i]

        return sum


    def getTotalW(self):
        sum = 0.
        for i in range(0, len(self.weight)):
            sum += self.N[i]*self.weight[i]

        return sum


    def getMaxW(self):
        maxW = 0.
        for i in range(0, len(self.weight)):
            if self.weight[i] > maxW:
                maxW = self.weight[i]

        return maxW


    def getN(self, group):
        return self.N[group]


    def getW(self, group):
        return self.weight[group]


    def getV(self, group):
        return self.V[group]


    def getIngested(self, group):
        return self.ingested[group]


    def getIngRate(self, group):
        return self.ingRate[group]


    def setN(self, group, N):
        self.N[group] = N


    def setW(self, group, W):
        self.weight[group] = W


    def addIngestion(self, group, ingestion):
        self.V[group] += ingestion
        self.ingested[group] += ingestion


    def stepGutContent(self, group, dt, T_w):
        self.V[group] = self.V[group]*(1 - dt*self.a_1*pow(T_w, self.a_2))



    def setIngRate(self, group, rate):
        self.ingRate[group] = rate


    def getAppetite(self, group):
        mgv = self.getMaxGutVolume(self.weight[group])
        rsv = self.V[group]/mgv
        if rsv > 0.3:
            return max(0, 0.5 - 0.57*(rsv-0.3)/(rsv-0.2))
        else:
            return min(1, 0.5 + 0.67*(0.3-rsv)/(0.4-rsv))


    def getMaxGutVolume(self, weight):
        # Burley and Vigg (1989):
        return 0.0007*pow(weight, 1.3796)


