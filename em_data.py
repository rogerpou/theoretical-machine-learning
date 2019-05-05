#generation of data file used for exercise 4.2

import numpy as np
import generative as gen

#Defnition of parameters of the model

pic = np.array([0.2, 0.3 , 0.2, 0.15, 0.15])
mu1 = np.array([-5, 2])
sig1 = np.array([[2.5, 0],[0, 2.4]])
mu2 = np.array([3.2, -5])
sig2 = np.array([[3, 0], [0, 2]])
mu3 = np.array([10, 7])
sig3 = np.array([[3.5, 0], [0, 1.7]])
mu4 = np.array([4, 3])
sig4 = np.array([[3.2, 1], [1, 1]])
mu5 = np.array([-3, -6])
sig5 = np.array([[2, 0], [0, 4]])
mu = np.array([mu1, mu2, mu3, mu4, mu5])
sig = np.array([sig1, sig2, sig3, sig4, sig5])
N=1000

#generate data using the generative.py module
data = gen.gmm(N, mu, sig, pic)


#save the generated data in a file em_data.txt

em_data = open("em_data", "w")
np.savetxt(em_data, data)
em_data.close()
