"""Lab 1.

Approximation of reachable set.
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for drawing
import numpy as np
from scipy.integrate import odeint

from operable import Operable

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
        res = [Operable(res[i])(time) for i in range(len(res))]
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

def get_parameter_q_function(dimension, matrix, cgc):
    def result(time):
        R = np.linalg.inv(matrix)

        CGC = [[cgc[i][j](time) for j in range(dimension)] for i in range(dimension)]
        
        R = np.dot(R, CGC)
        res = np.trace(R)/dimension
        return math.sqrt(res)
    return result


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
    cgc = np.dot(right_part, u_matrix)
    cgc = np.dot(cgc, np.transpose(right_part))
    rows, cols = np.shape(system)
    def diff(func, time):
        """Describes system of equations."""
        matrix_representation = array_to_matrix(func, rows, cols)
        a_r = np.dot(system, matrix_representation)
        r_a = np.dot(matrix_representation, np.transpose(system))

        parameter_q = get_parameter_q_function(rows, matrix_representation, cgc)(time)
        parameter_q = 1 if parameter_q < 0.00001 else parameter_q
        # parameter_q = 1

        q_r = np.dot(parameter_q, matrix_representation)
        res = np.add(a_r, r_a)
        res = np.add(res, q_r)

        tmp = np.dot(1/parameter_q, cgc)
        res = np.add(res, tmp)
        res = matrix_to_array(res)
        res = [Operable(res[i])(time) for i in range(len(res))]
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


def get_ellipse_points(center, shape_matrix, number_of_points):
    """Return two one-dimensional arrays that represent points on ellipse.

    Ellipse described by center and shape matrix ("shape" parameter)"""
    theta = np.linspace(0, 2*math.pi, number_of_points)
    e_vals, e_vecs = np.linalg.eig(shape_matrix)

    ax1 = 1/math.sqrt(e_vals[0])
    ax2 = 1/math.sqrt(e_vals[1])

    angle = math.acos(e_vecs[0][0])/math.sqrt(e_vecs[0][0]**2 + e_vecs[1][0]**2)

    if angle < 0:
        angle += 2*math.pi

    x_coordinates = []
    y_coordinates = []
    for t in theta:
        x_coordinates.append(
            ax1 * np.cos(t) * np.cos(angle) - ax2 * np.sin(t) * np.sin(angle) + center[0])
        y_coordinates.append(
            ax1 * np.cos(t) * np.sin(angle) + ax2 * np.sin(t) * np.cos(angle) + center[1])

    return x_coordinates, y_coordinates


def plot_result(t_array, center_array, shape_matrix_array, coordinates):
    """Plot of result.

    t_array - array of timestamps
    center_array - array containing centers of ellispoids
        for each timestamp
    shape_matrix_array - array containing shape matrices
        of ellipsoids for each timestamp
    coordinates - array that contains two numbers - numbers of coordinates
        in which result will be plotted.
    For e. g. if coordinates=[0, 1] and dimension = 3, than projection
    onto xy plane will be computed
    """

    t_len = len(t_array)
    initial_dimension = np.shape(center_array)[1]

    x_array = []
    y_array = []
    for t in range(t_len):
        x_array.append([])
        y_array.append([])
        center, shape_matrix = project_ellipsoid_to_subspace(center_array[t],\
            shape_matrix_array[t], initial_dimension, coordinates)
        x_array[t], y_array[t] = get_ellipse_points(center, shape_matrix, t_len)

    fig = plt.figure()
    axes = Axes3D(fig)
    axes.set_xlabel('T')
    axes.set_ylabel('Y1')
    axes.set_zlabel('Y2')
    for i in range(t_len):
        axes.plot([t_array[i] for _ in range(t_len)], x_array[i], y_array[i])

    plt.show()


def solve(system, center_of_start_set, start_set_shape_matrix,\
        right_part, u_shape_matrix, t_start, t_end, t_count):
    """Solve approximation problem.

    Assume n - dimension of the problem.

    Returns
    t_array - array of timestamps of length t_count
    center - array of shape (t_count, n)
    shape_matrix - array of shpe(t_count, n, n)"""
    t_array = np.linspace(t_start, t_end, t_count)
    center = find_center(system, center_of_start_set, t_array)
    shape_matrix =\
        find_ellipsoid_matrix(system, right_part, u_shape_matrix, start_set_shape_matrix, t_array)

    return t_array, center, shape_matrix


def main():
    """Entry point for the app."""
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
        [Operable(lambda t: t**2+t*16), Operable(lambda t: t**2+t*8)],
        [Operable(lambda t: t**2+t*8), Operable(lambda t: 4*t**2 + t)]
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

    T_START = 0 # T_START - start of time
    T_END = 10  # T_END - end of time
    T_COUNT = 50  # T_COUNT - number of timestamps on [t_start, t_end]

    t_array, center, shape_matrix = solve(A, A0, Q0, C, G, T_START, T_END, T_COUNT)
    plot_result(t_array, center, shape_matrix, [0, 2])


main()