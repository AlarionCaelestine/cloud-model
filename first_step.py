import numpy as np
import matplotlib.pyplot as plt
import math


def F(X, Y, Z):
    if X != 0:
        Fy = Z / X**2
    else:
        Fy = 0
    Fz = X ** 2 * np.exp((-Y))
    return Fy, Fz


def euler_method(step, X, Y, Z):
    Z_new = Z + step * F(X, Y, Z)[1]
    Y_new = Y + step * F(X, Y, Z)[0]
    return Y_new, Z_new


def runge_kutta(step, X, Y, Z):
    k1_y, k1_z = F(X, Y, Z)
    k2_y, k2_z = F(X + step / 2, Y + k1_y * step / 2, Z + k1_z*step/2)
    k3_y, k3_z = F(X + step / 2, Y + k2_y * step / 2, Z + k2_z*step/2)
    k4_y, k4_z = F(X + step, Y + k3_y * step, Z + k3_z*step)

    Z_new = Z + step/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
    Y_new = Y + step/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    return Y_new, Z_new


x_max = 100
N = 1000
step = x_max / N
Y2_10_7 = 0.15882767754579646
X = list(range(N+1))
Z1 = list(range(N+1))
Z2 = list(range(N+1))
Y1 = list(range(N+1))
Y2 = list(range(N+1))

Z1[0] = 0
Z2[0] = 0
Y1[0] = 0
Y2[0] = 0
for i in range(N):
    X[i+1] = X[i] + step
    Y1[i+1], Z1[i+1] = euler_method(step, X[i], Y1[i], Z1[i])
    Y2[i+1], Z2[i+1] =  runge_kutta(step, X[i], Y2[i], Z2[i])


# plt.plot(X, Y1, color='red', label='euler_method')
# plt.plot(X, Y2, color='blue', label='runge_kutta')
# plt.plot(X, Y1, color='green', label='Y1')
# plt.plot(X, Y2, color='cyan', label='Y2')
# plt.legend(loc='best')

# for i in range(len(X)):
#    print(X[i], Y1[i], Y2[i], "", Z1[i], Z2[i])
# print('X_last =', X[-1])
# print('Y1_last =', Y1[-1])
# print('delta_Y1 =', Y2_10_7-Y1[-1])

# plt.xlim([0, 2])
# plt.ylim([0, 1])
# plt.show()
