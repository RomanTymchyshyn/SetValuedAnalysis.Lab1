"""This module provides utilities for plotting."""

import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from utils import project_ellipsoid_to_subspace


def get_ellipse_points(center, shape_matrix, number_of_points):
    """Return two one-dimensional arrays that represent points on ellipse.

    Ellipse described by center and shape matrix.
    number_of_points - number of discrete points on ellipse."""
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


# def _plot_2d_ellipse_in_3d_space(center, shape, time_point):
#     """Plot ellipse given by center and shape matrix in some time point."""



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
