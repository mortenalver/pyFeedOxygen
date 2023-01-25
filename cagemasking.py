import numpy as np
import math

def circularMasking(dims, dxy, radius, maskBottom):
    mask = np.array(np.zeros(dims), dtype=np.bool8)
    # Find center of grid:
    center = ((dims[0]-1)/2.0, (dims[1]-1)/2.0)

    for i in range(0,dims[0]):
        for j in range(0, dims[1]):
            distX = i-center[0]
            distY = j - center[1]
            rpos = dxy*math.sqrt(distX*distX + distY*distY)
            inside = rpos<=radius
            for k in range(0, dims[2]-1):
                mask[i,j,k] = inside
            if maskBottom:
                mask[i,j,-1] = False
            else:
                mask[i, j, -1] = inside

    return mask



