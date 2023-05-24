import numpy as np
import math

class Advect:

    varyAmbient = False
    affinityProfile = []

    def __init__(self, cageDims):
        self.advect = np.zeros(cageDims)
        self.diffus = np.zeros(cageDims)
        #self.newValues = np.zeros(cageDims)


    def advectField(self, cageDims, xBounds, field, dt, dxy, dz, mask, sinkingSpeed, diffKappa, diffKappaZ,
                    currentSpeed, currentOffset,
                    sourceTerm, feedingRateMult, ambientValue):

        self.calcAdvectAndDiff(cageDims, xBounds, field, dxy, dz, mask, sinkingSpeed, diffKappa, diffKappaZ,
                               currentSpeed, currentOffset, ambientValue, dt)

        presum = 0.0
        postsum = 0.0
        for k in range(0, cageDims[2]):
            #for i in range(0, cageDims[0]):
            for i in range(xBounds[0], xBounds[1]):
                for j in range(0, cageDims[1]):
                    presum = presum + field[i,j,k]
                    field[i,j,k] = field[i,j,k] + dt* (self.advect[i,j,k] + self.diffus[i,j,k])
                    postsum = postsum + field[i,j,k]

                    if feedingRateMult > 0:
                        field[i,j,k] = field[i,j,k] + dt*feedingRateMult*sourceTerm[i,j,k]

        return (-postsum+presum)/dt

    def pickVal(self, cageDims, field, ambientValue, i, j, k):
        if k < 0:
            return 0
        if i < 0 or i >= cageDims[0] or j < 0 or j >= cageDims[1] or k >= cageDims[2]:
            return ambientValue[min(cageDims[2]-1, k)]
        else:
            return field[i,j,k]


    def maxmod(self, a, b):
        if a*b < 0:
            return 0
        else:
            if abs(a) > abs(b):
                return a
            else:
                return b


    def minmod(sefl, a, b):
        if a*b < 0:
            return 0
        else:
            if abs(a) < abs(b):
                return a
            else:
                return b


    def superbeeAdv(self, dt, dx, c_ll, c_l, c_c, c_r, c_rr, v_l, v_r):
        sum = 0.0
        if v_l >= 0:
            sigma_l = self.maxmod(self.minmod((c_c-c_l)/dx, 2.*(c_l-c_ll)/dx),
                                  self.minmod(2.*(c_c-c_l)/dx, (c_l-c_ll)/dx))
            sum += (v_l/dx)*(c_l + (sigma_l/2.0)*(dx-v_l*dt))

        else:
            sigma_l = self.maxmod(self.minmod((c_c-c_l)/dx, 2.*(c_r-c_c)/dx),
                                  self.minmod(2.*(c_c-c_l)/dx, (c_r-c_c)/dx))
            sum += (v_l/dx)*(c_c - (sigma_l/2.0)*(dx+v_l*dt))

        if v_r >= 0:
            sigma_r = self.maxmod(self.minmod((c_c-c_l)/dx, 2.*(c_r-c_c)/dx),
                                  self.minmod(2.*(c_c-c_l)/dx, (c_r-c_c)/dx))
            sum -= (v_r/dx)*(c_c + (sigma_r/2.0)*(dx-v_r*dt))

        else:
            sigma_r = self.maxmod(self.minmod((c_r-c_c)/dx, 2.*(c_rr-c_r)/dx),
                                  self.minmod(2.*(c_r-c_c)/dx, (c_rr-c_r)/dx))
            sum -= (v_r/dx)*(c_r - (sigma_r/2.0)*(dx+v_r*dt))

        return sum

    def setVaryAmbient(self, varyAmbient, affinityProfile):
        self.varyAmbient = varyAmbient
        self.affinityProfile = affinityProfile


    def calcAdvectAndDiff(self, cageDims, xBounds, field, dxy, dz, mask, sinkingSpeed, diffKappa, diffKappaZ,
                          currentSpeed, currentOffset, ambientValue, dt):

        ambientValueHere = np.zeros(ambientValue.shape)

        # Set up neighbourhood variables:
        c_h = 0.
        x_nb = np.zeros((4))
        y_nb = np.zeros((4))
        z_nb = np.zeros((4))
        z_nb_diff = np.zeros((2))
        current = np.zeros((3,2))

        for k in range(0, cageDims[2]):
            #for i in range(0, cageDims[0]):
            for i in range(xBounds[0], xBounds[1]):
                for j in range(0, cageDims[1]):
                    # Find local current. For each dimension we need the current on two edges:
                    # TODO: walls based on mask matrix are not implemented
                    current = np.zeros((3, 2))
                    current[0, 0] = currentSpeed[i,j,k,0] + currentOffset[0]
                    current[1, 0] = currentSpeed[i,j,k,1] + currentOffset[1]
                    current[2, 0] = currentSpeed[i,j,k,2] + currentOffset[2] + sinkingSpeed
                    current[0, 1] = currentSpeed[i+1,j,k,0] + currentOffset[0]
                    current[1, 1] = currentSpeed[i,j+1,k,1] + currentOffset[1]
                    current[2, 1] = currentSpeed[i,j,k+1,2] + currentOffset[2] + sinkingSpeed

                    # If appropriate, apply ambient value reduction:
                    ambientValueHere[:] = ambientValue[:]
                    if self.varyAmbient:
                        # Check if we are near an edge, otherwise the ambient value has no effect:
                        if (i<2 or j<2 or i>=cageDims[0]-2 or j>=cageDims[1]-2 or k<2 or k>=cageDims[2]-2):
                            currHere = 100.*np.sqrt(current[0, 0]*current[0, 0] + current[1, 0]*current[1, 0])
                            omega_red = 0.35
                            if currHere < 6.:
                                omega_red = 1.85 - 0.25*currHere
                            ambRedValue = ((cageDims[0] - i)/cageDims[0])*omega_red*self.affinityProfile[k]
                            ambientValueHere -= ambRedValue

                    # Define cell neighbourhood:
                    c_h = field[i,j,k]
                    x_nb[0] = self.pickVal(cageDims, field, ambientValueHere, i - 2, j, k)
                    x_nb[1] = self.pickVal(cageDims, field, ambientValueHere, i - 1, j, k)
                    x_nb[2] = self.pickVal(cageDims, field, ambientValueHere, i + 1, j, k)
                    x_nb[3] = self.pickVal(cageDims, field, ambientValueHere, i + 2, j, k)

                    y_nb[0] = self.pickVal(cageDims, field, ambientValueHere, i, j - 2, k)
                    y_nb[1] = self.pickVal(cageDims, field, ambientValueHere, i, j - 1, k)
                    y_nb[2] = self.pickVal(cageDims, field, ambientValueHere, i, j + 1, k)
                    y_nb[3] = self.pickVal(cageDims, field, ambientValueHere, i, j + 2, k)

                    z_nb[0] = self.pickVal(cageDims, field, ambientValueHere, i, j, k - 2)
                    z_nb[1] = self.pickVal(cageDims, field, ambientValueHere, i, j, k - 1)
                    z_nb[2] = self.pickVal(cageDims, field, ambientValueHere, i, j, k + 1)
                    z_nb[3] = self.pickVal(cageDims, field, ambientValueHere, i, j, k + 2)
                    if k==0:
                        z_nb_diff[0] = c_h  # No diffusion through surface
                    else:
                        z_nb_diff[0] = z_nb[1]
                    if k==cageDims[2]-1:
                        z_nb_diff[1] = ambientValueHere[k]
                    else:
                        z_nb_diff[1] = z_nb[2]

                    # Collect advection term:
                    self.advect[i,j,k] = \
                        self.superbeeAdv(dt, dxy, x_nb[0], x_nb[1], c_h, x_nb[2], x_nb[3],
                                                          current[0,0], current[0,1]) \
                        + self.superbeeAdv(dt, dxy, y_nb[0], y_nb[1], c_h, y_nb[2], y_nb[3],
                                                          current[1,0], current[1,1]) \
                        + self.superbeeAdv(dt, dxy, z_nb[0], z_nb[1], c_h, z_nb[2], z_nb[3],
                                                          current[2, 0], current[2, 1])

                    # Collect diffusion term:
                    self.diffus[i,j,k] = diffKappa*((x_nb[1] - 2*c_h + x_nb[2])/dxy/dxy
                        + (y_nb[1] - 2*c_h + y_nb[2])/dxy/dxy) \
                        + diffKappaZ*((z_nb_diff[0] - 2*c_h + z_nb_diff[1])/dz/dz)



# # Test mass conservation of superbee advecter:
# advy = Advect((5,5,5))
# dt = 0.1
# mx = 20
# values = np.zeros((mx,))
# for i in range(5,mx-5):
#     values[i] = 1.#2.+math.sin(i*0.5)
# v2 = values
#
# curr = np.zeros((mx+1,))
# for i in range(0, mx+1):
#     curr[i] = 1.5
# curr[10:-1] = 1.
#
# print(values)
# print(np.sum(values))
#
# # Run advection step:
# for j in range(0,4):
#     for i in range(2, mx-2):
#         adv = advy.superbeeAdv(dt, 1, values[i-2], values[i-1], values[i],
#                                values[i+1], values[i+2], curr[i], curr[i+1])
#         v2[i] = values[i] + dt*adv
#     values[:] = v2[:]
#
# print(v2)
# print(np.sum(v2))