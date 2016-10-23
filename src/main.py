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

U0 = [0, 0]
G = [
    [1/4, 0],
    [0, 1/4]
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

def find_center(matrix, initial_condition, t_array):
    """Returns center of approximation ellipsoid for reachable set.

    Considered equation: dx/dt = A*x + C*u

    matrix - matrix A(t) which defines system of diff equations of model
    initial_condition - initial condition for system, i. e. - center of ellipsoid,
        which describes initial set.
    t_array - discrete representation of time interval
    """
    def system(func, time):
        """Describes system of equations."""
        res = np.dot(matrix, func)
        return res
    sol = odeint(system, initial_condition, t_array)
    return sol

def matrix_to_array(matrix):
    """Convert matrix to array representation.

    Used to convert matrix differential equation to system of differential equations.
    Returns array of size n*m where n - number of rows in matrix,
    m - number of columns in matrix."""
    rows, cols = np.shape(matrix)
    return np.reshape(matrix, (rows*cols))

def array_to_matrix(array, rows, cols):
    """Convert array that represents matrix to matrix.

    Used to convert system of differential equations back to matrix form.
    Returns matrix of shape (rows, cols)."""
    return np.reshape(array, (rows, cols))

def solution_to_matrix_form(solution, rows, cols, timestamps_count):
    """Convert numerical solution of system of ODE to matrix form.

    Initially solution is represented in two dimensional form, where
    each row corresponds to certain timestamp and value is array of size
    rows*cols.
    This will be transformed into representation,
    where value for each timestamp will be matrix of shape of ellipsoid
    that corresponds to certain timestamp.
    """
    return np.reshape(solution, (timestamps_count, rows, cols))

def find_ellipsoid_matrix(system, right_part, u_matrix,\
    start_set_ellipsoid, t_array):
    """Returns shape matrix of approximation ellipsoid for reachable set.

    Considered equation: dx/dt = A*x + C*u

    system - matrix A(t) which defines system of diff equations of model
    right_part - matrix C
    u_matrix - shape matrix for boundary ellipsoid for u
    start_set_ellipsoid - initial condition for system, i. e. - shape matrix of ellipsoid,
        which describes initial set.
    t_array - discrete representation of time interval
    """
    parameter_q = 1
    cgc = np.dot(right_part, u_matrix)
    cgc = np.dot(cgc, np.transpose(right_part))
    cgc = np.dot(1/parameter_q, cgc)
    rows, cols = np.shape(system)
    def diff(func, time):
        """Describes system of equations."""
        matrix_representation = array_to_matrix(func, rows, cols)
        a_r = np.dot(system, matrix_representation)
        r_a = np.dot(matrix_representation, np.transpose(system))
        q_r = np.dot(parameter_q, matrix_representation)
        res = np.add(a_r, r_a)
        res = np.add(res, q_r)
        res = np.add(res, cgc)
        res = matrix_to_array(res)
        return res
    initial_condition = matrix_to_array(start_set_ellipsoid)
    sol = odeint(diff, initial_condition, t_array)
    shape_matrix = solution_to_matrix_form(sol, rows, cols, np.shape(t_array)[0])
    return shape_matrix


T_START = 0 # T_START - start of time
T_END = 10  # T_END - end of time
T_COUNT = 1000  # T_COUNT - number of timestamps on [t_start, t_end]
T = np.linspace(T_START, T_END, T_COUNT)
CENTER = find_center(A, A0, T)
SHAPE_MATRIX = find_ellipsoid_matrix(A, C, G, Q0, T)

plt.plot(T, CENTER[:, 0], 'b', label='y1(t)')
plt.plot(T, CENTER[:, 1], 'g', label='y2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
