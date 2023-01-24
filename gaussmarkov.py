import numpy as np


def updateGaussMarkov(prev, beta, sigma, dt):
    f = np.exp(-beta*dt)
    return f*prev + np.sqrt(1-f*f)*np.random.normal()*sigma



# a = 0
# for i in range(0,50):
#     a = updateGaussMarkov(a, 2.5, 1, 1)
#     print(a)