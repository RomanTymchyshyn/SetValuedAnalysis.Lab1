"""Lab 1.

Approximation of reachable set.
"""

import numpy as np
from operable import Operable

N = 4

# set up start set M0
A0 = np.array([0, 0, 0, 0])
QV1 = np.array([1, 0, 0, 0])
QV2 = np.array([0, 1, 0, 0])
QV3 = np.array([0, 0, 1, 0])
QV4 = np.array([0, 0, 0, 1])
Q0_SEMI_AXES = np.array([1, 2, 3, 4])
Q0_LAMBDA = np.array([
    [
        (0 if j != i else 1/Q0_SEMI_AXES[i]**2) for j in range(N)
    ]
    for i in range(N)
])

Q0_EIGEN_VECTORS_MATRIX = np.transpose([QV1, QV2, QV3, QV4])
Q0_EIGEN_VECTORS_MATRIX_INV = np.linalg.inv(Q0_EIGEN_VECTORS_MATRIX)

Q0 = np.dot(Q0_EIGEN_VECTORS_MATRIX, Q0_LAMBDA)
Q0 = np.dot(Q0, Q0_EIGEN_VECTORS_MATRIX_INV)

# set up bounding ellipsoid for u(t)

U0 = np.array([0, 0, 0, 0])
G = np.array([
    [1/4, 0, 0, 0],
    [0, 1/4, 0, 0],
    [0, 0, 1/4, 0],
    [0, 0, 0, 1/4]
])




@Operable
def _function1(arg1, arg2):
    return arg1 + arg2

def _function2(arg1, arg2):
    return arg1 * arg2

ADD = Operable(lambda x, y: x**2 + y**2) + _function2
MUL = _function1 * _function2
DIV = _function1 / _function2
print(ADD(1, 2))
print(MUL(1, 2))
print(DIV(1, 2))


A = [Operable(lambda x: x), Operable(lambda x: x)]
B = [Operable(lambda x: x**2), 2]
C = np.dot(A, B)
print(C(2))
