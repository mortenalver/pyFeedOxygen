

def setO2AffinityWithVerticalProfile(cageDims, dz, fishMaxDepth, mask,
                                     affinityProfile, affinity, o2Affinity):
        o2AffSum = 0
        for i in range(0, cageDims[0]):
            for j in range(0, cageDims[1]):
                for k in range(0, cageDims[2]):
                    lDepth = (k+0.5)*dz
                    if (mask[i,j,k] and (lDepth < fishMaxDepth)):
                        affinity[i,j,k] = affinityProfile[k]
                        o2Affinity[i,j,k] = affinityProfile[k]
                    o2AffSum += o2Affinity[i,j,k]

        return o2AffSum
