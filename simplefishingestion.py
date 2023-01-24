import numpy as np
import gaussmarkov


def calculateIngestion(cageDims, dt, feed, o2, affinity, o2Affinity, o2AffSum,
                    ingDist, o2ConsDist,
                    dxy, dz, mask, pelletWeight, T_w, fish, perturbations, feedingFrac_r):


    if not perturbations:
        rndMult = 1.0

    T_h = 12.  # Handling time
    k_T_s = 1.  # Factor for what feed particle count related to fish count makes search time important
    b = 0.4  # Exponent for confusion factor
    c = 0.5  # Exponent for f_d factor

    o2consumptionMult = 1  # This factor can be used to globally multiply the o2 consumption of the fish.
    U = 1.  # Swimming speed (body lengths/s)

    # We calculate O2 consumption according to a hybrid pattern consisting of two parts:
    # 1. Proportionally to feed distribution. This component is only present when fish is feeding
    # 2. Even distribution over cage volume.
    # When fish is feeding, the factor o2_even_fraction determines the fraction that is determined
    # according to part 2.
    o2_even_fraction = 0.75 + feedingFrac_r

    # O2 consumption (Grøttum and Sigholt, 1998):
    #  VO2 (mg/kg/h) = 61.6 * BW^0.33 * 1.03^T * 1.79^U
    #  BW: body weight (kg)
    #  T: temperature  (C)
    #  U: swimming speed (body lengths/s)

    N = fish.getTotalN()
    WtotKg = 0.001*fish.getTotalW()
    if N == 0:
        return (0, 0)

    totalFeed = 0
    for i in range(0, cageDims[0]):
        for j in range(0, cageDims[1]):
            for k in range(0, cageDims[2]):
                if mask[i,j,k]:
                    totalFeed += affinity[i,j,k]*feed[i,j,k]

    feeding = totalFeed > 1e-3
    if feeding:
        w_0 = 1./(T_h + k_T_s*N*pelletWeight/totalFeed)
    else:
        w_0 = 0.

    # NEW rho calculation:
    muSum = 0
    wsum = 0
    cellVol = dxy*dxy*dz # Cell volume in m3
    maxDensity = 0.
    imx = 0
    jmx = 0
    kmx = 0
    if feeding:

        for i in range(0, cageDims[0]):
            for j in range(0, cageDims[1]):
                for k in range(0, cageDims[2]):
                    if mask[i, j, k]:
                        wHere = WtotKg * affinity[i,j,k] * feed[i,j,k] / totalFeed
                        wsum += wHere
                        density = wHere/cellVol
                        if density > maxDensity:
                            maxDensity = density
                            imx = i
                            jmx = j
                            kmx = k
                        muSum += wHere*mu(density)

    if feeding:
        rho = muSum/WtotKg
    else:
        rho = 0


    # TODO: need to find good way to model effect of o2 level on appetite

    # Confusion factor:
    p_c = pow(rho, b)
    p_a = np.zeros((fish.getNGroups()))
    totalW = fish.getTotalW()
    f_a = 0
    # Calculate appetite factors and the f_a factor:
    for i in range(0, fish.getNGroups()):
        p_a[i] = fish.getAppetite(i)
        f_a += fish.getN(i)*p_a[i]*fish.getW(i)/totalW

    if rho > 0:
        f_d = pow(rho, -c)
    else:
        f_d = 0

    # Calculate p_h factors and feed intake per group:
    maxW = fish.getMaxW()
    w_f = np.zeros((fish.getNGroups()))
    totalIntake = 0.
    for i in range(0, fish.getNGroups()):
        # Hierarchy factor:
        p_h = pow(fish.getW(i)/maxW, f_a*f_d)
        w_f[i] = pelletWeight*w_0*p_c*p_a[i]*p_h
        totalIntake += fish.getN(i)*w_f[i]

    if totalIntake == 0:
        feeding = False


    #print("Total feed: "+str(totalFeed))
    #print("Intake rate: " + str(totalIntake))

    # Make sure the ingestion doesn't exceed the available feed:
    if dt*totalIntake > totalFeed:
        multiplier = totalFeed/(dt*totalIntake)
        totalIntake *= multiplier
        for i in range(0, fish.getNGroups()):
            w_f[i] *= multiplier

    # if totalFeed > 200:
    #     print("2. totalIntake = " + str(totalIntake))
    #     print("p_c  = "+str(p_c))
    #     print("p_a = " + str(p_a))
    #     print("p_h = " + str(p_h))
    #     print("w_f * 60 = " + str(60.*w_f))

    # Calculate total oxygen consumption in mg per second:
    o2Cons = 0
    for i in range(0, fish.getNGroups()):
        # Consumption: N multiplied by weight in kg multiplied by O2 consumption per kg divided by 3600 s:
        o2Cons += o2consumptionMult*fish.getN(i)*0.001*fish.getW(i)*61.6*pow(fish.getW(i)*0.001, -0.33)*pow(1.03, T_w[0])*pow(1.79, U)/3600.0


    # Calculate how much is removed from each cell:
    cellIng = np.zeros(cageDims)
    sumCellIng = 0
    correction = 1
    if feeding:
        for i in range(0, cageDims[0]):
            for j in range(0, cageDims[1]):
                for k in range(0, cageDims[2]):
                    # Remove same relative fraction of feed everywhere:
                    cellIng[i,j,k] = affinity[i,j,k]*totalIntake*feed[i,j,k]/totalFeed
                    sumCellIng += cellIng[i,j,k]

        if sumCellIng > 0:
            correction = totalIntake/sumCellIng
        else:
            correction = 0

    if perturbations:
        rndMult = 1.0 + 0*0.02 * np.random.normal()  # 0.2*np.random.normal()
        o2Cons *= rndMult

    # Find the O2 consumption in the two parts:
    if feeding:
        o2Cons_even = o2_even_fraction*o2Cons
        o2Cons_feeding = (1.-o2_even_fraction)*o2Cons
    else:
        o2Cons_even = o2Cons
        o2Cons_feeding = 0.


    # Remove feed and o2 from cells:
    presum = 0.
    postsum = 0.
    for i in range(0, cageDims[0]):
        for j in range(0, cageDims[1]):
            for k in range(0, cageDims[2]):
                if mask[i, j, k]:
                    feed[i,j,k] -= dt*correction*cellIng[i,j,k]
                    ingDist[i,j,k] = correction*cellIng[i,j,k]
                    presum += o2[i,j,k]
                    # TODO: negative o2 values are simply cut off, no reduction of consumption when o2 is low
                    # Since O2 is given as a concentration (mg/l), we need to divide by the cell volume in l:
                    # Remove part of the O2 consumption according to the same distribution as feed ingestion.
                    o2Before = o2[i,j,k]
                    if feeding:
                        o2[i,j,k] = max(0., o2[i,j,k] - dt*(correction*cellIng[i,j,k]/totalIntake)*o2Cons_feeding/(1000.0*dxy*dxy*dz))

                    # Remove part (or all) of O2 consumption according to affinity:
                    #o2[i,j,k] = max(0., o2[i,j,k] - dt*rndMult*(o2Affinity[i,j,k]/o2AffSum)*o2Cons_even/(1000.0*dxy*dxy*dz))
                    o2[i,j,k] = max(0., o2[i,j,k] - dt*(o2Affinity[i,j,k]/o2AffSum)*o2Cons_even/(1000.0*dxy*dxy*dz))
                    o2ConsDist[i,j,k] = (o2Before - o2[i,j,k])/dt # Cons. rate is # removed per time step divided by step size

                    postsum += o2[i,j,k]


    o2ConsumptionRate = (presum-postsum)*1000.0*dxy*dxy*dz/dt # mg o2 removed from volume per second

    # Add feed to stomachs:
    for i in range(0, fish.getNGroups()):
        fish.stepGutContent(i, dt, T_w)
        fish.addIngestion(i, dt*w_f[i])
        fish.setIngRate(i, w_f[i])


    return totalIntake, rho, o2ConsumptionRate



# Calculate local mu value for a given fish density (kg/m3)
def mu(density):
    thresh = 50 #110
    if density < thresh:
        return 1
    else:
        return max(0, 1-(density-thresh)/50)


