# file : generative.py

import numpy as np


#This module generates data sets according to different generative models
#taking as inputs: size of data set, model parameters and model to be used.

#The first model in this module is the Gaussian Mixture Model (GMM)

def gmm(N, mu, sigma, pi_c):
    dim = pi_c.shape
    mu_dim = np.array(mu.shape)
    len_x = mu_dim[1]
    k1 = np.arange(100)
    k2 = np.zeros(100)
    x_=np.zeros([N, len_x])
    counter_1 = 0
    counter_2 = 0
    #print('pi_c:', pi_c)
    for c in range(dim[0]):
        counter_2 += np.floor(100*pi_c[c])
        k2[int(counter_1) : int(counter_2)]= c
        counter_1 = counter_2
    decision = np.array([k1, k2])
    for i in range(N):
        cpr =int(np.floor(np.random.uniform(0,100)))
        c_index = int(k2[cpr])
        mu_n = mu[c_index, :]
        sigma_n = sigma[c_index, :]
        x_[i, :] = np.random.multivariate_normal(mu_n, sigma_n)

    return(x_)
