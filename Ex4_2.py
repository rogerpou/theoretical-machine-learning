#Import 'standard' modules and functions
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import numpy as np

#Import self-made modules
import generative as gen
import ndem as em
import initialization as ini


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
N = 1000
C = 5


#data = gen.gmm(1000, mu, sig, pic) #This line of code is used to generate data

#Read data from file (em_data.txt)
em_data = open("em_data", "r")
data = np.loadtxt(em_data)
em_data.close()
print('Data shape:', data.shape)

#2D plot of imported data
x, y = data.T
plt.plot(x, y ,'x')
plt.axis('equal')
plt.show()

print('Data shape:', data.shape)


#Initializtion of parameters for EM algorithm
delta = 4
mu_,sig_,pi_c = ini.init(data, C, delta)

#Convergence condition
conv= 0.000001

#Compute parameters using EM algorithm
log_a,mu_new, sig_new, pic_new = em.gaussian(data, mu_, sig_, pi_c, conv)

#Print results
iterations = np.array(log_a.shape)
print('\n Number of iterations:' , iterations[0])
print('\n New Means: \n', mu_new)
print('New covariance matrices: \n', sig_new)
print('New prior probabilities: \n', pic_new)

#Save results
em_results = open("em_results", "w")
np.savetxt(em_results, pic_new)
np.savetxt(em_results, mu_new)
for c in range(C):
    np.savetxt(em_results, sig_new[c,:])
np.savetxt(em_results, log_a)
em_results.close()

#-----------------------------------------------------------------------------
#Plots from generated data

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 2-dimensional distribution over variables X and Y
N = 60
X = np.linspace(-15, 15, N)
Y = np.linspace(-15, 15, N)
X, Y = np.meshgrid(X, Y)
# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

F1 = mn(mu1, sig1)
F2 = mn(mu2, sig2)
F3 = mn(mu3, sig3)
F4 = mn(mu4, sig4)
F5 = mn(mu5, sig5)
Z = F1.pdf(pos)+F2.pdf(pos)+F3.pdf(pos)+F4.pdf(pos)+F5.pdf(pos)
# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()

fig2 = plt.figure()
plt.contour(X,Y, Z)
plt.show()

#------------------------------------------------------------------------------
#plots from em_results
mu_1 = mu_new[0, :]
sig_1 = sig_new[0, :]
mu_2 = mu_new[1, :]
sig_2 = sig_new[1, :]
mu_3 = mu_new[2, :]
sig_3 = sig_new[2, :]
mu_4 = mu_new[3, :]
sig_4 = sig_new[3, :]
mu_5 = mu_new[4, :]
sig_5 = sig_new[4, :]

F1n = mn(mu_1, sig_1)
F2n = mn(mu_2, sig_2)
F3n = mn(mu_3, sig_3)
F4n = mn(mu_4, sig_4)
F5n = mn(mu_5, sig_5)
Zn1 = F1n.pdf(pos)
Zn2 = F2n.pdf(pos)
Zn3 = F3n.pdf(pos)
Zn4 = F4n.pdf(pos)
Zn5 = F5n.pdf(pos)


fig2 = plt.subplots()
plt.plot(x, y ,'x')
plt.axis('equal')
plt.contour(X,Y, Zn1)
plt.contour(X,Y, Zn2)
plt.contour(X,Y, Zn3)
plt.contour(X,Y, Zn4)
plt.contour(X,Y, Zn5)
plt.show()
