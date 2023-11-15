import numpy as np


def get_grid_coords(num_grids_x, num_grids_y):
    """
    get coordinates of the grids, the first grid is (0, 0), then (1, 0) and so on
    :param num_grids_x: num of blocks in axis x
    :param num_grids_y: num of blocks in axis y
    :return: x coords, y coords
    """
    x_grid_coords = np.array([range(num_grids_x)] * num_grids_y).flatten()  # grid coordinates X
    y_grid_coords = np.repeat(range(num_grids_y), num_grids_x)  # grid coordinates Y
    return x_grid_coords, y_grid_coords


def project(points, H):
    """
    project 3d points using projection matrix
    :param points: 3d points
    :param H: projection matrix
    :return: projected points
    """
    project_points = H.dot(points)
    # project_points = np.matmul(H, points)
    new_w = project_points[2, :]
    new_x = (project_points[0, :] / new_w)  # current centers location in the previous frame center X
    new_y = (project_points[1, :] / new_w)  # current centers location in the previous frame center Y
    return new_x, new_y


def reshape(arr: np.ndarray, new_shape):
    return arr.reshape(*new_shape)
