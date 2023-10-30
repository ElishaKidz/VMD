import numpy as np
import cv2
from VMD.MovingCameraForegroundEstimetor import utils


class BaseModel:
    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim):
        self.prev_means = None
        self.vars = None
        self.ages = None
        self.model_height = model_width
        self.model_width = model_width
        self.block_size = block_size
        self.num_models = num_models
        self.var_init = var_init
        self.var_trim = var_trim

    def init(self, gray_frame):
        self.prev_means = np.zeros((self.num_models, self.modelHeight, self.modelWidth))
        self.vars = np.zeros((self.num_models, self.modelHeight, self.modelWidth))
        self.ages = np.zeros((self.num_models, self.modelHeight, self.modelWidth))

    def get_models(self):
        return self.prev_means, self.vars, self.ages


class CompensationModel(BaseModel):
    def __init__(self, num_models, model_height, model_width, block_size, lam, theta_v):
        super(CompensationModel, self).__init__(num_models, model_height, model_width, block_size)
        self.lam = lam
        self.theta_v = theta_v

    def init(self, gray_frame, means, vars, ages):
        super(CompensationModel, self).init(gray_frame)
        H = np.identity(3)
        return self.compensate(H)

    def get_weights_for_directions(self, abs_offset_x, abs_offset_y):
        """
        get weight of horizontal offset, vertical offset, and diagonal offset
        :return: horizontal, vertical, and diagonal weights
        """
        W_H = (abs_offset_x * (1 - abs_offset_y)).reshape(self.model_height, self.model_width)
        W_V = (abs_offset_y * (1 - abs_offset_x)).reshape(self.model_height, self.model_width)
        W_HV = (abs_offset_x * abs_offset_y).reshape(self.model_height, self.model_width)
        return W_H, W_V, W_HV

    @staticmethod
    def update_by_condition(cond, temp, prev, W, x_grid_coords, y_grid_coords, grid_overlap):
        """
        weighted mean for directional overlapping
        :param cond: directional cond
        :param temp: store weighted mean in
        :param prev: prev values
        :param W: directional wights
        :param x_grid_coords: x_grid_coords
        :param y_grid_coords: y_grid_coords
        :param grid_overlap: directional grid overlap
        """
        temp[:, y_grid_coords[cond], x_grid_coords[cond]] += \
            W[y_grid_coords[cond], x_grid_coords[cond]] * \
            prev[:, prev_y_grid_coords[cond],
            grid_overlap[cond]]  # weighted mean for the direction overlaping


    def compensate(self, H, prev_means, prev_vars, prev_ages):
        x_grid_coords, y_grid_coords = utils.get_grid_coords(self.model_width, self.model_height)

        points = np.asarray([x_grid_coords * self.block_size + self.block_size / 2, y_grid_coords * self.block_size +
                             self.block_size / 2, np.ones(len(x_grid_coords))])  # current frame grid centers 3D

        # find prev centers
        prev_center_x, prev_center_y = utils.project(points, H)

        prev_x_grid_coords_temp = prev_center_x / self.block_size
        prev_y_grid_coords_temp = prev_center_y / self.block_size

        prev_x_grid_coords = np.floor(prev_x_grid_coords_temp).astype(int)  # grid location index X
        prev_y_grid_coords = np.floor(prev_y_grid_coords_temp).astype(int)  # grid location index Y

        offset_x = prev_x_grid_coords_temp - prev_x_grid_coords - 0.5  # offset of grid's center from the prev to curr frame X
        offset_y = prev_y_grid_coords_temp - prev_y_grid_coords - 0.5  # offset of grid's center from the prev to curr frame Y

        abs_offset_x = abs(offset_x)  # offset of grid's center from the prev to curr frame X
        abs_offset_y = abs(offset_y)  # offset of grid's center from the prev to curr frame Y

        # calculate the weight of the crop which the prev center is in
        W_H, W_V, W_HV = self.get_weights_for_directions(abs_offset_x, abs_offset_y)

        W_self = ((1 - abs_offset_x) * (1 - abs_offset_y)).reshape(self.model_height,
                                                 self.model_width)  # the normalized weight of this crop

        W = np.zeros((self.num_models, self.model_height, self.model_width))

        temp_means = np.zeros(prev_means.shape)
        temp_ages = np.zeros(prev_means.shape)

        x_overlap_idx = prev_x_grid_coords + np.sign(offset_x).astype(int)  # figure out X direction of jumping, taking the overlaping center index in X
        cond_horizontal = (prev_y_grid_coords >= 0) & (prev_y_grid_coords < self.model_height) & (x_overlap_idx >= 0) & (
                    x_overlap_idx < self.model_width)  # check if this crop is in image

        # horizontal overlapping
        CompensationModel.update_by_condition(cond_horizontal, temp_means, prev_means, W_H, x_grid_coords, y_grid_coords,
                                              x_overlap_idx)
        CompensationModel.update_by_condition(cond_horizontal, temp_ages, prev_ages, W_H, x_grid_coords, y_grid_coords,
                                              x_overlap_idx)
        W[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] += W_H[y_grid_coords[cond_horizontal],
                                                                                    x_grid_coords[cond_horizontal]]

        # same for vertical
        y_overlap_idx = prev_y_grid_coords + np.sign(offset_y).astype(int)
        cond_vertical = (y_overlap_idx >= 0) & (y_overlap_idx < self.modelHeight) & (prev_x_grid_coords >= 0) & (prev_x_grid_coords < self.modelWidth)

        CompensationModel.update_by_condition(cond_vertical, temp_means, prev_means, W_V, x_grid_coords, y_grid_coords,
                                              y_overlap_idx)
        CompensationModel.update_by_condition(cond_vertical, temp_ages, prev_ages, W_V, x_grid_coords, y_grid_coords,
                                              y_overlap_idx)
        W[:, y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] += W_V[y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]]

        # same for diagonal
        x_overlap_idx = prev_x_grid_coords + np.sign(offset_x).astype(int)
        y_overlap_idx = prev_y_grid_coords + np.sign(offset_y).astype(int)
        condHV = (y_overlap_idx >= 0) & (y_overlap_idx < self.modelHeight) & (x_overlap_idx >= 0) & (x_overlap_idx < self.modelWidth)
        temp_means[:, y_grid_coords[condHV], x_grid_coords[condHV]] += W_HV[y_grid_coords[condHV], x_grid_coords[condHV]] * prev_means[:, y_overlap_idx[condHV], x_overlap_idx[condHV]]
        temp_ages[:, y_grid_coords[condHV], x_grid_coords[condHV]] += W_HV[y_grid_coords[condHV], x_grid_coords[condHV]] * prev_ages[:, y_overlap_idx[condHV], x_overlap_idx[condHV]]
        W[:, y_grid_coords[condHV], x_grid_coords[condHV]] += W_HV[y_grid_coords[condHV], x_grid_coords[condHV]]

        # same for closest center
        condSelf = (prev_y_grid_coords >= 0) & (prev_y_grid_coords < self.modelHeight) & (prev_x_grid_coords >= 0) & (prev_x_grid_coords < self.modelWidth)
        temp_means[:, y_grid_coords[condSelf], x_grid_coords[condSelf]] += W_self[y_grid_coords[condSelf], x_grid_coords[condSelf]] * prev_means[:, prev_y_grid_coords[condSelf],
                                                                                    prev_x_grid_coords[condSelf]]
        temp_ages[:, y_grid_coords[condSelf], x_grid_coords[condSelf]] += W_self[y_grid_coords[condSelf], x_grid_coords[condSelf]] * prev_ages[:, prev_y_grid_coords[condSelf],
                                                                                    prev_x_grid_coords[condSelf]]
        W[:, y_grid_coords[condSelf], x_grid_coords[condSelf]] += W_self[y_grid_coords[condSelf], x_grid_coords[condSelf]]

        # save the copmpensated means and ages and normalized according to the weights
        self.temp_means[W != 0] = 0

        self.temp_ages[:] = 0
        W[W == 0] = 1
        self.temp_means += temp_means / W
        self.temp_ages += temp_ages / W

        #  same shit for variance
        temp_var = np.zeros(self.prev_means.shape)
        temp_var[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] += W_H[y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] * (prev_vars[:, prev_y_grid_coords[cond_horizontal], x_overlap_idx[cond_horizontal]] +
                                                                      np.power(self.temp_means[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] -
                                                                               self.prev_means[:, prev_y_grid_coords[cond_horizontal],
                                                                               x_overlap_idx[cond_horizontal]],
                                                                               2))  # TODO: not the same calculation. think deeply

        temp_var[:, y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] += W_V[y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] * (prev_vars[:, y_overlap_idx[cond_vertical], prev_x_grid_coords[cond_vertical]] +
                                                                      np.power(self.temp_means[:, y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] -
                                                                               self.prev_means[:,
                                                                               y_overlap_idx[cond_vertical], prev_x_grid_coords[cond_vertical]],
                                                                               2))

        temp_var[:, y_grid_coords[condHV], x_grid_coords[condHV]] += W_HV[y_grid_coords[condHV], x_grid_coords[condHV]] * (prev_vars[:, y_overlap_idx[condHV], x_overlap_idx[condHV]] +
                                                                           np.power(self.temp_means[:, y_grid_coords[condHV],
                                                                                    x_grid_coords[condHV]] -
                                                                                    self.prev_means[:, y_overlap_idx[condHV],
                                                                                    x_overlap_idx[condHV]],
                                                                                    2))

        temp_var[:, y_grid_coords[condSelf], x_grid_coords[condSelf]] += W_self[y_grid_coords[condSelf], x_grid_coords[condSelf]] * (
                    prev_vars[:, prev_y_grid_coords[condSelf], prev_x_grid_coords[condSelf]] +
                    np.power(self.temp_means[:, y_grid_coords[condSelf], x_grid_coords[condSelf]] -
                             self.prev_means[:, prev_y_grid_coords[condSelf], prev_x_grid_coords[condSelf]],
                             2))

        self.temp_vars = temp_var / W  # same normalization
        cond = (prev_y_grid_coords < 1) | (prev_y_grid_coords >= self.modelHeight - 1) | (prev_x_grid_coords < 1) | (
                    prev_x_grid_coords >= self.modelWidth - 1)  # if new grid that was not intreduce before
        self.temp_vars[:, y_grid_coords[cond], x_grid_coords[cond]] = self.INIT_BG_VAR  # update the new grid with the init variance
        self.temp_ages[:, y_grid_coords[cond], x_grid_coords[cond]] = 0
        self.temp_vars[self.temp_vars < self.MIN_BG_VAR] = self.MIN_BG_VAR  # triming the variance from below


class StatisticalModel(BaseModel):
    def __init__(self, num_models, model_height, model_width, block_size, age_trim, theta_s, theta_d=4,
                 sensetivity=False):
        super(StatisticalModel, self).__init__()
        self.theta_s = theta_s
        self.theta_d = theta_d
        self.sensetivity = sensetivity

    def init(self, gray_frame):
        super(StatisticalModel, self).init(gray_frame)

    def get_foreground(self, gray, com_means, com_vars, com_ages):
        pass


