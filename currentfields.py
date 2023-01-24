import numpy as np
import math

def getProfileCurrentField(field, currentX, currentY):
    dims = field.shape
    lenProf = currentX.shape[0]
    for k in range(0, dims[2]):
        for i in range(0, dims[0]):
            for j in range(0,dims[1]):
                field[i,j,k,0] = currentX[min(k, lenProf-1)]
                field[i,j,k,1] = currentY[min(k, lenProf-1)]
                field[i,j,k,2] = 0.