import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
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


class PlotFramesDecorator:
    def __init__(self, save_dir):
        self.counter = 0
        self.save_dir = Path(save_dir)
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            klt_obj = args[0]
            previous_gray_frame = klt_obj.imgPrevGray.copy()
        
            result = func(*args, **kwargs)
            self.counter+=1
            if previous_gray_frame is not None:
            # Access frames and homography
                current_gray_frame = klt_obj.imgPrevGray
                homography = result

                # Apply homography and prepare plotting
                cv.warpPerspective(current_gray_frame, homography, (previous_gray_frame.shape[1], previous_gray_frame.shape[0]))
                fig, axes = plt.subplots(1, 2, figsize=(10, 6))

                # Plot frames
                axes[0].imshow(previous_gray_frame)
                axes[0].set_title("Previous")
                axes[1].imshow(current_gray_frame)
                axes[1].set_title("Rotated Current")

                fig.savefig(self.save_dir/f'{self.counter}_{self.counter+1}.png')
                
            return result

        return wrapper
