from first_step import *
from Constants import *
import numpy as np
import matplotlib.pyplot as plt
import math


P_s = rho_0*T_C*R / mu_0
P_0 = (R*T/mu)**4 / (4*math.pi*G**3*M**2)
p_out = P_s/P_0
print(P_s, P_0, p_out)

x_max = 10
N = 10000
step = x_max / N
X = list(range(N+1))
Y = list(range(N+1))
Z = list(range(N+1))
p_s = list(range(N+1))

Y[0] = 0
Z[0] = 0

for i in range(N):
    X[i+1] = X[i] + step
    Y[i+1], Z[i+1] = runge_kutta(step, X[i], Y[i], Z[i])
    p_s[i] = Z[i]**2 * np.exp(-Y[i])


def get_function_value(p_out):
    x_0 = 0
    Y_0 = 0
    Z_0 = 0
    k_s = 0
    for k in range(N-1):
        if abs(p_s[k+1]-p_out) < abs(p_s[k_s]-p_out):
            k_s = k
    x_0 = X[k_s]
    Y_0 = Y[k_s]
    Z_0 = Z[k_s]
    return x_0, Y_0, Z_0


iterations = 100
x_0, Y_0, Z_0 = get_function_value(p_out)
print(x_0, Y_0, Z_0)
mu_s = Z_0
x_j = ((4*math.pi*G**3*M**2*P_s)/(math.exp(-Y_0) * (Z_0/x_0**2)**2))**(1/4) * (mu/(R*T))
# x_j = x_0
# for j in range(iterations):
#     if x_0 != 0 and Y_0 !=0 and Z_0 != 0:
#         x_j = ((4*math.pi*G**3*M**2*P_s)/(math.exp(-Y_0) * (Z_0/x_0**2)**2))**(1/4) * (mu/(R*T))
#         x_0 = x_j

rho_c = 1/(4*math.pi) * (mu_s/M)**2 * ((R*T)/(mu*G))**3
r_0 = ((R*T)/(mu*G*rho_c*4*math.pi))**0.5
R = x_j * r_0

print("zeta =", x_j)
print("P_s [Pa] =", P_s)
print("mu_s =", mu_s)
print("rho_c [kg*m^(-3)]=", rho_c)
print("Radius [m]/[km]/[au] =", R, R/1000, R/(150e9))
