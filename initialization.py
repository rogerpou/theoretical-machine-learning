# file : initialization.py

#This module generates initialization values for the EM Algorithm
#based on the data that it receives

import numpy as np

def init(data, classes, delta):

    dim_data = np.array(data.shape)
    N = dim_data[0]
    D = dim_data[1]
    c = classes
    print('Number of samples =', N)
    print('Dimension =', D)
    print('Number of classes =', c)
    mu_i = np.zeros([c, D])
    sig_i = np.zeros([c, D, D])
    mu_g = np.sum(data, axis = 0)/N
    sig_g = np.cov(data.T)
    id = np.identity(D)

    mu_i = mu_g*np.ones([c, D]) + delta*np.random.randn(c, D)
    pi_i = np.zeros(c)
    pi_i[0] = 0.5 * np.random.rand()
    pi_i[1:] = (1-pi_i[0])/(c-1)
    pi_i = np.around(pi_i, decimals = 2)

    for j in range(c):
        sig_i[j, :, :] = sig_g + id*(delta*np.random.rand(D, D))

    return(mu_i, sig_i, pi_i)
