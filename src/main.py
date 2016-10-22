"""Lab 1.

Approximation of reachable set.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from operable import Operable

# dimension
N = 4

# set up model parameters
# weights
M1 = 2
M2 = 3

# friction forces
B = 4
B1 = 3
B2 = 5

# stiffnesses
K = 2
K1 = 2
K2 = 2

# set up start set M0
A0 = [1, 1, 1, 1]
QV1 = [1, 0, 0, 0]
QV2 = [0, 1, 0, 0]
QV3 = [0, 0, 1, 0]
QV4 = [0, 0, 0, 1]
Q0_SEMI_AXES = [1, 2, 3, 4]
Q0_LAMBDA = [
    [
        (0 if j != i else 1/Q0_SEMI_AXES[i]**2) for j in range(N)
    ]
    for i in range(N)
]

Q0_EIGEN_VECTORS_MATRIX = np.transpose([QV1, QV2, QV3, QV4])
Q0_EIGEN_VECTORS_MATRIX_INV = np.linalg.inv(Q0_EIGEN_VECTORS_MATRIX)

Q0 = np.dot(Q0_EIGEN_VECTORS_MATRIX, Q0_LAMBDA)
Q0 = np.dot(Q0, Q0_EIGEN_VECTORS_MATRIX_INV)

# set up bounding ellipsoid for u(t)

U0 = [0, 0, 0, 0]
G = [
    [1/4, 0, 0, 0],
    [0, 1/4, 0, 0],
    [0, 0, 1/4, 0],
    [0, 0, 0, 1/4]
]

# set up matrix of the system (i. e. matrix A(t))
A = [
    [0, 1, 0, 0],
    [-(K + K1)/M1, -(B + B1)/M1, K/M1, B/M1],
    [0, 0, 0, 1],
    [K/M2, B/M2, -(K + K2)/M2, -(B + B2)/M2]
]

C = [
    [0, 0],
    [1/M1, 0],
    [0, 0],
    [0, -1/M2]
]

def find_center(matrix, initial_condition, t_start, t_end, t_count):
    """Returns center of approximation ellipsoid for reachable set.

    Considered equation: dx/dt = A*x + C*u

    matrix - matrix A(t) which defines system of diff equations of model
    initial_condition - initial condition for system, i. e. - center of ellipsoid,
        which describes initial set.
    t_start - start of time
    t_end - end of time
    t_count - number of timestamps on [t_start, t_end]
    """
    t_array = np.linspace(t_start, t_end, t_count)
    def system(func, time):
        """Describes system of equations."""
        res = np.dot(matrix, func)
        return res
        # theta, omega = func
        # dydt = [omega, -omega - np.sin(theta)]
        # return dydt
    sol = odeint(system, initial_condition, t_array)
    return sol, t_array

def find_ellipsoid_matrix(system, right_part, u_matrix,\
    start_set_ellipsoid, t_start, t_end, t_count):
    """Returns shape matrix of approximation ellipsoid for reachable set.

    Considered equation: dx/dt = A*x + C*u

    system - matrix A(t) which defines system of diff equations of model
    right_part - matrix C
    u_matrix - shape matrix for boundary ellipsoid for u
    start_set_ellipsoid - initial condition for system, i. e. - shape matrix of ellipsoid,
        which describes initial set.
    t_start - start of time
    t_end - end of time
    t_count - number of timestamps on [t_start, t_end]
    """
    pass

CENTER, T = find_center(A, A0, 0, 10, 1000)

plt.plot(T, CENTER[:, 0], 'b', label='y1(t)')
plt.plot(T, CENTER[:, 1], 'g', label='y2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
