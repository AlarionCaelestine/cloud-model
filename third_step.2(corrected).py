from first_step import *
from second_step import *
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import math

x_max = 4.0  # [au]
N = 15
step = x_max / N
G_ast = 4 * np.pi ** 2  # [au**3 y**(-2) m_sun**(-1)]
M_sun = 2e30  # [kg]
au = 1.496e11  # [m]
one_year = 365.2425 * 24 * 3600  # [s]

rho_c = 1.3e-9

rho_c_ast = rho_c * au ** 3 * M_sun ** (-1)  # 3D-density [solar mass / au**3]
R_ast = R * one_year ** 2 * au ** (-2) * M_sun ** (-1)  # universal gas constant in astronomical units
mu_ast = mu * M_sun ** (-1)  # molar mass in astronomical units

# oX[0] and oX[N+1]=oX[-1] -- boundary layers
# oX[1]..oX[N] --- domain
oX = np.arange(N + 2)
oY = np.arange(N + 2)
oZ = np.arange(N + 2)
for n in range(N + 2):
    oX[n] += -step / 2 + n * step
    oY[n] += -step / 2 + n * step
    oZ[n] += -step / 2 + n * step

Phi_0 = [[[0 for i in range(N + 2)] for j in range(N + 2)] for k in range(N + 2)]
Phi = [[[0 for i in range(N + 2)] for j in range(N + 2)] for k in range(N + 2)]
Phi_2d = [[0 for i in range(N + 2)] for j in range(N + 2)]
Rho = [[[0 for i in range(N + 2)] for j in range(N + 2)] for k in range(N + 2)]
f = [[[0 for i in range(N + 2)] for j in range(N + 2)] for k in range(N + 2)]

A = 4.0 * np.pi * G_ast * rho_c_ast
B = mu_ast / R_ast


def boundary_conditions(Phi, Rho, M_0):
    for i in range(N + 2):
        for j in range(N + 2):
            Phi[i][j][0] = Phi[i][j][1]  # bc Phi on z=0
            Rho[i][j][0] = Rho[i][j][1]  # bc Rho on z=0
            Phi[i][j][-1] = -G_ast * M_0 / np.sqrt(oX[i]**2+oY[j]**2+oZ[-1]**2)  # bc on z=z_max

    for i in range(N + 2):
        for k in range(N + 2):
            Phi[i][0][k] = Phi[i][1][k]  # bc Phi on y=0
            Rho[i][0][k] = Rho[i][1][k]  # bc Phi on y=0
            Phi[i][-1][k] = -G_ast * M_0 / np.sqrt(oX[i]**2+oY[-1]**2+oZ[k]**2)  # bc on y=y_max

    for j in range(N + 2):
        for k in range(N + 2):
            Phi[0][j][k] = Phi[1][j][k]  # bc Phi on x=0
            Rho[0][j][k] = Rho[1][j][k]  # bc Phi on x=0
            Phi[-1][j][k] = -G_ast * M_0 / np.sqrt(oX[-1]**2+oY[j]**2+oZ[k]**2)  # bc on x=x_max
    return Phi, Rho


def main__iterations_method(Phi, Rho, M_0):
    M = 0
    Phi, Rho = boundary_conditions(Phi, Rho, M_0)
    for i0 in range(N):
        i = i0 + 1  # shift from boundary
        for j0 in range(N):
            j = j0 + 1
            for k0 in range(N):
                k = k0 + 1
                for _ in range(5):  # iterations to solve transcendental equation phi = W*(L + A*exp(-phi*B/T))
                    #print (i,j,k, ": ", f[i][j][k], Phi[i][j][k])  
                    f[i][j][k] = A * np.exp(-Phi[i][j][k] * B / T)
                    Phi[i][j][k] = (Phi[i + 1][j][k] + Phi[i - 1][j][k]
                                    + Phi[i][j + 1][k] + Phi[i][j - 1][k]
                                    + Phi[i][j][k + 1] + Phi[i][j][k - 1]
                                    - step ** 2 * f[i][j][k]) / 6.0  # FIXED laplacian

                #x = 5
                #input(x)
                Rho[i][j][k] = rho_c_ast * np.exp(-Phi[i][j][k] * B / T)
                M += 8.0 * Rho[i][j][k] * step ** 3
    return Phi, Rho, M


def main__seidel_method(Phi, Rho, M_0):
    M = 0
    Phi, Rho = boundary_conditions(Phi, Rho, M_0)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                f[i][j][k + 1] = A * np.exp(-Phi[i][j][k + 1] * B / T)
                Phi[i][j][k + 1] = (Phi[i + 1][j][k] + Phi[i - 1][j][k + 1] + Phi[i][j + 1][k] + Phi[i][j - 1][
                    k + 1] - step ** 2 * f[i][j][k + 1]) / 6
                Rho[i][j][k + 1] = rho_c_ast * np.exp(-Phi[i][j][k + 1] * B / T)
                M += 8 * Rho[i][j][k + 1] * step ** 2
    return Phi, Rho, M


# initial conditions on Phi
for i in range(N + 2):
    for j in range(N + 2):
        for k in range(N + 2):
            Phi[i][j][k] = -1.0e-5

M = 4 * rho_c_ast * (x_max / 10) ** 2
for q in range(1000):
    Phi, Rho, M = main__iterations_method(Phi, Rho, M)
    if q%100 == 0 : 
    	print(q, "Mass =", M, "M_Sun; Phi[3][4][5]=",Phi[3][4][5], " Rho[3][4][5]=",Rho[3][4][5])

#fig = plt.figure()
#ax = plt.axes(projection='3d')

##ax.plot_surface(oY, oZ, Phi[0])
#plt.plot(oZ, Phi[1][1], color='red', label='potential')
#plt.plot(oZ, -G_ast*M/oZ, color='blue', label='point potential')
#plt.legend(loc='best')
#plt.show()


#Phi_2d = Phi.reshape((Phi.shape[0], Phi.shape[1]))
for i in range(N+2):
    for j in range(N+2):
        Phi_2d[i][j] = Phi[i][j][0]

Phi = np.array(Phi)
Phi_2d = np.array(Phi_2d)

#print('Phi_2d =', Phi_2d)
#print('len0 =', len(Phi_2d[0]))
#print('len1 =', len(Phi_2d[1]))


X, Y = np.meshgrid(oX, oY)
Z = Phi_2d

#print('lenx =', len(X))
#print('leny =', len(Y))

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)

plt.show()
