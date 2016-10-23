"""Lab 1.

Approximation of reachable set.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # for drawing
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

def project_ellipsoid_to_subspace(center, shape_matrix, initial_dimension, projection_coordinates):
    """Projects n-dimesnaional ellipsoid into 2d plane.

    shape_matrix - shape matrix of ellipsoid
    initial_dimension - dimension of ellipsoid
    projection_coordinates - numbers of coordinates that will form projection subspace.
    For e. g. if projection_coordinates=[0, 1] and dimension = 3, than projection
    onto xy plane will be computed"""
    # basis of subspace, ellipsoid is being projected
    projection_coordinates.sort()
    kernel_coordinates = list(set(range(initial_dimension)) - set(projection_coordinates))
    projection_dimension = len(projection_coordinates)
    kernel_dimension = len(kernel_coordinates)
    pr_basis = [
        [1 if i == projection_coordinates[j] else 0 for j in range(projection_dimension)]
        for i in range(initial_dimension)
    ]
    kernel_basis = [
        [1 if i == kernel_coordinates[j] else 0 for j in range(kernel_dimension)]
        for i in range(initial_dimension)
    ]

    new_center = np.dot(np.transpose(pr_basis), center)

    # find new shape matrix
    basis_dot_shape = np.dot(np.transpose(pr_basis), shape_matrix)
    kernel_dot_shape = np.dot(np.transpose(kernel_basis), shape_matrix)
    e11 = np.dot(basis_dot_shape, pr_basis)
    # Yt * E * Z = (Zt * E * Y)t for symmetric E
    e12 = np.dot(basis_dot_shape, kernel_basis)
    e22 = np.dot(kernel_dot_shape, kernel_basis)
    e22 = np.linalg.inv(e22)
    temp = np.dot(e12, e22)
    temp = np.dot(temp, np.transpose(e12))

    new_shape_matrix = np.subtract(e11, temp)

    return new_center, new_shape_matrix


def get_ellipse_points(center, shape):
    theta = np.linspace(0, 2*math.pi, T_COUNT)
    e_vals, e_vecs = np.linalg.eig(shape)

    a = 1/math.sqrt(e_vals[0])
    b = 1/math.sqrt(e_vals[1])

    angle = math.acos(e_vecs[0][0])/math.sqrt(e_vecs[0][0]**2 + e_vecs[1][0]**2)

    if angle < 0:
        angle += 2*math.pi

    x = []
    y = []
    for t in theta:
        x.append(a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle) + center[0])
        y.append(a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle) + center[1])

    return x, y


T_START = 0 # T_START - start of time
T_END = 10  # T_END - end of time
T_COUNT = 50  # T_COUNT - number of timestamps on [t_start, t_end]
T = np.linspace(T_START, T_END, T_COUNT)
CENTER = find_center(A, A0, T)
SHAPE_MATRIX = find_ellipsoid_matrix(A, C, G, Q0, T)

xx = []
yy = []
for t in range(len(T)):
    xx.append([])
    yy.append([])
    center, shape_matrix = project_ellipsoid_to_subspace(CENTER[t], 
        SHAPE_MATRIX[t], len(CENTER[t]), [2, 3])
    xx[t], yy[t] = get_ellipse_points(center, shape_matrix)

fig = plt.figure()
axes = Axes3D(fig)
axes.set_xlabel('T')
axes.set_ylabel('Y1')
axes.set_zlabel('Y2')
for i in range(T_COUNT):
    axes.plot([T[i] for _ in range(T_COUNT)], xx[i], yy[i])

plt.show()
