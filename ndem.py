# file : ndem
from scipy.stats import multivariate_normal as mn
import numpy as np
#This module computes the EM algorithm for N dimensional data sets and C classes

def log_lh(x, mu, sigma, pic):
    C_dim = np.array(pic.shape)
    C=C_dim[0]
    N_dim = np.array(x.shape)
    N = N_dim[0]
    logl = 0
    for n in range(N):
        sum_p = 0
        x1 = x[n,:]
        for c in range(C):
            mu_n = mu[c, :]
            sigma_n = sigma[c, :]
            sum_p += (pic[c]*mn.pdf(x1, mu_n, sigma_n))
        logl = logl + np.log(sum_p)
    return logl



def gaussian (x_n, mu, sigma, pi_c, conv):
    #extract number of classes and dimensions
    C_dim = np.array(pi_c.shape)
    C=C_dim[0]
    mu_dim = np.array(mu.shape)
    N_dim = np.array(x_n.shape)
    D = mu_dim[1]
    N = N_dim[0]
    llh1 = log_lh(x_n, mu, sigma, pi_c)
    log_array = np.array([0, llh1])
    diff_log = 1.0
    pic_new = pi_c
    mu_new = mu
    sig_new = sigma
    i=0
    while diff_log > conv:                      # Convergence condition
        #E-step
        y = np.zeros((N, C, D))
        y2 = np.zeros((N, C, D))
        rho = np.zeros((N, C))

        #Calculation of r[n][k] for each data point
        for n in range (N):
            x1 = x_n[n,:]
            den = 0
            den1 = 0
            for c in range(C):
                mu_n = mu[c, :]
                sigma_n = sigma[c, :]
                den1 = pi_c[c]*mn.pdf(x1, mu_n, sigma_n)
                den += den1
            for c in range (C):
                mu_n = mu[c, :]
                sigma_n = sigma[c, :]
                rho[n][c]= (pi_c[c]*mn.pdf(x1, mu_n, sigma_n))/den
                y[n][c][:]= rho[n][c]*x1

        #M-step

        #print("rho = " ,rho)

        #sum over individual values of rho^n for each class
        k1=(np.sum(rho, axis=0))
        k2=(np.sum(y, axis=0))


        #print('k1 = ' ,k1)
        #print('k2 = ' ,k2)

        #Calculation of new Prior Probabilities and new Means
        for c in range (C):
            pic_new[c] = k1[c]/N
            mu_new[c, :] = k2[c, :]/k1[c]
        #print ("pic_new = " ,pic_new)
        #print ("mu_new = " ,mu_new)


        # Calculation of new Variances
        k3 = np.zeros((C, D, D), dtype = float)
        #print (k3.shape)
        for c in range(C):
            mu_n = mu_new[c, :]
            for n in range(N):
                x1 = x_n[n,:]
                k3[c,:,:]+= rho[n][c]*np.outer((x1-mu_n),(x1-mu_n).T)
                #print('k3[c]', k3[c,:,:], rho[n][c], x1, mu_n)
            sig_new[c, :, :]=k3[c]/k1[c]
            #print('sigma c', sig_new[c])
        print(sig_new)
        i += 1
        llh_new = np.array([i, log_lh(x_n, mu_new, sig_new, pic_new)])

        log_array = np.append([log_array],[llh_new])
        diff_log = llh_new[1]-llh1
        llh1 = llh_new[1]


        pi_c = pic_new
        mu = mu_new
        sigma = sig_new
        #theta_old = theta_new
        #print('log_likelihood:', log_array)
    return (log_array,mu_new, sig_new, pic_new)
        #End of EM algorithm
#------------------------------------------------------------------------------
