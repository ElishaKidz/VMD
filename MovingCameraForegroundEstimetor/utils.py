import numpy as np


def get_grid_coords(num_grids_x, num_grids_y):
    x_grid_coords = np.array([range(num_grids_y)] * num_grids_x).flatten()  # grid coordinates X
    y_grid_coords = np.repeat(range(num_grids_y), num_grids_x)  # grid coordinates Y
    return x_grid_coords, y_grid_coords


def project(points, H, ):
    project_points = H.dot(points)
    new_w = project_points[2, :]
    new_x = (project_points[0, :] / new_w)  # current centers location in the previous frame center X
    new_y = (project_points[1, :] / new_w)  # current centers location in the previous frame center Y
    return new_x, new_y

