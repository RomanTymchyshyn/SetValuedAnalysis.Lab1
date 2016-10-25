"""Lab 1.

Approximation of reachable set.
Considered model is x'(t) = A(t)*x(t) + C(t)u(t).
t belongs to [t0, t1]
x(t0) belongs to start set M0, which is ellipsoid
u(t) - control function, which belongs to U(t)
    which is also ellipsoid for any non-negative t
"""

import numpy as np

from approximation import solve
from operable import Operable
from plot_utils import plot_approximation_result


def main():
# pylint: disable=C0103
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

    # set up shape matrix for bounding ellipsoid for u(t)
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
    plot_approximation_result(t_array, center, shape_matrix, [0, 1], 'T', 'Y1', 'Y2')


main()
