import numpy as np
from VMD.MovingCameraForegroundEstimetor import utils, utils_numba
from numba import jit, prange


class BaseModel:
    """
    parent class for the compensation model and the statistics model
    """
    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim):
        """
        see foreground model init for documentation
        """
        self.prev_means = None
        self.vars = None
        self.ages = None
        self.means = None
        self.model_height = model_height
        self.model_width = model_width
        self.block_size = block_size
        self.num_models = num_models
        self.var_init = var_init
        self.var_trim = var_trim

        self.func_time = 0
        self.num_frames = 0

    def init(self):
        """
        create zero params of size of the models and the grids
        """
        self.means = np.zeros((self.num_models, self.model_height, self.model_width), dtype=np.float32)
        self.vars = np.zeros((self.num_models, self.model_height, self.model_width), dtype=np.float32)
        self.ages = np.zeros((self.num_models, self.model_height, self.model_width), dtype=np.float32)

    def get_models(self):
        """
        :return: means vars and ages of models
        """
        return self.means, self.vars, self.ages

    def update(self, var_init, var_trim):
        self.var_init = var_init
        self.var_trim = var_trim


class CompensationModel(BaseModel):
    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim, lam, theta_v):
        super(CompensationModel, self).__init__(num_models, model_height, model_width, block_size, var_init, var_trim)
        self.lam = lam
        self.theta_v = theta_v

        self.x_grid_coords = None
        self.y_grid_coords = None
        self.points = None

    def init(self, means, vars, ages):
        super(CompensationModel, self).init()
        H = np.identity(3, dtype=np.float32)
        self.get_grid_coords_and_points()
        return self.compensate(H, means, vars, ages)

    def update(self, var_init, var_trim, lam, theta_v):
        super(CompensationModel, self).update(var_init, var_trim)
        self.lam = lam
        self.theta_v = theta_v

    def get_grid_coords_and_points(self):
        self.x_grid_coords, self.y_grid_coords = utils.get_grid_coords(self.model_width, self.model_height)
        self.points = np.asarray([self.x_grid_coords * self.block_size + self.block_size / 2, self.y_grid_coords * self.block_size +
                             self.block_size / 2, np.ones(len(self.x_grid_coords))], dtype=np.float32)  # current frame grid centers 3D

    def get_weights_for_directions(self, abs_offset_x, abs_offset_y):
        """
        get weight of horizontal offset, vertical offset, and diagonal offset
        :return: horizontal, vertical, and diagonal weights
        """
        W_H = (abs_offset_x * (1 - abs_offset_y)).reshape(self.model_height, self.model_width)
        W_V = (abs_offset_y * (1 - abs_offset_x)).reshape(self.model_height, self.model_width)
        W_HV = (abs_offset_x * abs_offset_y).reshape(self.model_height, self.model_width)
        W_self = ((1 - abs_offset_x) * (1 - abs_offset_y)).reshape(self.model_height,
                                                                   self.model_width)
        return W_H, W_V, W_HV, W_self

    @staticmethod
    def update_by_condition(cond, temp, prev, W, x_grid_coords, y_grid_coords, grid_overlap_x, grid_overlap_y):
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
            prev[:, grid_overlap_y[cond],
            grid_overlap_x[cond]]  # weighted mean for the direction overlaping

    def compensate_mean_and_age(self, temp_means, temp_ages, prev_means, prev_ages, W, W_H, W_V, W_HV, W_self,
                                x_grid_coords, y_grid_coords, prev_grid_coords_x, prev_grid_coords_y, offset_x,
                                offset_y):
        """
        compensate the means and the ages of the model
        :param temp_means: where to temporarly store the compenated means
        :param temp_ages: same for ages
        :param prev_means: means of models in previous iterations
        :param prev_ages: same for ages
        :param W: sum of all weights
        :param W_H: horizontal weight
        :param W_V: vertical weight
        :param W_HV: diagonal weight
        :param W_self: weight of closest grid
        :param x_grid_coords: x grid coords from utils.get_grid_coords
        :param y_grid_coords: same for y
        :param prev_grid_coords_x: previouse coordinates of the grid
        :param prev_grid_coords_y: same for y
        :param offset_x: offset of grid's center from the prev to curr frame X
        :param offset_y: same for y
        :return: updated and compensated W, temp_means, temp_ages and conditions for update cond_horizontal,
        cond_vertical, cond_diagonal, cond_self for updating variance so not calc them again
        """
        x_overlap_idx = prev_grid_coords_x + np.sign(offset_x).astype(
            int)  # figure out X direction of jumping, taking the overlaping center index in X
        cond_horizontal = (prev_grid_coords_y >= 0) & (prev_grid_coords_y < self.model_height) & (
                x_overlap_idx >= 0) & (
                                  x_overlap_idx < self.model_width)  # check if this crop is in image

        # horizontal overlapping
        CompensationModel.update_by_condition(cond_horizontal, temp_means, prev_means, W_H, x_grid_coords,
                                              y_grid_coords,
                                              x_overlap_idx, prev_grid_coords_y)
        CompensationModel.update_by_condition(cond_horizontal, temp_ages, prev_ages, W_H, x_grid_coords, y_grid_coords,
                                              x_overlap_idx, prev_grid_coords_y)
        W[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] += W_H[y_grid_coords[cond_horizontal],
                                                                                    x_grid_coords[cond_horizontal]]

        # same for vertical
        y_overlap_idx = prev_grid_coords_y + np.sign(offset_y).astype(int)
        cond_vertical = (y_overlap_idx >= 0) & (y_overlap_idx < self.model_height) & (prev_grid_coords_x >= 0) & \
                        (prev_grid_coords_x < self.model_width)

        CompensationModel.update_by_condition(cond_vertical, temp_means, prev_means, W_V, x_grid_coords, y_grid_coords,
                                              prev_grid_coords_x, y_overlap_idx)
        CompensationModel.update_by_condition(cond_vertical, temp_ages, prev_ages, W_V, x_grid_coords, y_grid_coords,
                                              prev_grid_coords_x, y_overlap_idx)
        W[:, y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] += W_V[
            y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]]

        # same for diagonal
        x_overlap_idx = prev_grid_coords_x + np.sign(offset_x).astype(int)
        y_overlap_idx = prev_grid_coords_y + np.sign(offset_y).astype(int)
        cond_diagonal = (y_overlap_idx >= 0) & (y_overlap_idx < self.model_height) & (x_overlap_idx >= 0) & (
                x_overlap_idx < self.model_width)
        CompensationModel.update_by_condition(cond_diagonal, temp_means, prev_means, W_HV, x_grid_coords, y_grid_coords,
                                              x_overlap_idx, y_overlap_idx)
        CompensationModel.update_by_condition(cond_diagonal, temp_ages, prev_ages, W_HV, x_grid_coords, y_grid_coords,
                                              x_overlap_idx, y_overlap_idx)
        W[:, y_grid_coords[cond_diagonal], x_grid_coords[cond_diagonal]] += W_HV[
            y_grid_coords[cond_diagonal], x_grid_coords[cond_diagonal]]

        # same for closest center
        cond_self = (prev_grid_coords_y >= 0) & (prev_grid_coords_y < self.model_height) & (prev_grid_coords_x >= 0) & (
                prev_grid_coords_x < self.model_width)
        CompensationModel.update_by_condition(cond_self, temp_means, prev_means, W_self, x_grid_coords, y_grid_coords,
                                              prev_grid_coords_x, prev_grid_coords_y)
        CompensationModel.update_by_condition(cond_self, temp_ages, prev_ages, W_self, x_grid_coords, y_grid_coords,
                                              prev_grid_coords_x, prev_grid_coords_y)

        W[:, y_grid_coords[cond_self], x_grid_coords[cond_self]] += W_self[
            y_grid_coords[cond_self], x_grid_coords[cond_self]]

        return W, temp_means, temp_ages, cond_horizontal, cond_vertical, cond_diagonal, cond_self

    def compensate(self, H, prev_means, prev_vars, prev_ages):
        """
        compensate the models
        :param H: homography matrix
        :param prev_means: the final means from prev iteration
        :param prev_vars: same for variance
        :param prev_ages: same for ages
        :return: compensated model
        """
        self.num_frames += 1
        # find prev centers
        prev_center_x, prev_center_y = utils_numba.project(self.points, H)
        prev_x_grid_coords, prev_y_grid_coords, offset_x, offset_y, abs_offset_x, abs_offset_y = \
            utils_numba.calculate_all_coords(prev_center_x, prev_center_y, self.block_size)

        # calculate the weight of the crop which the prev center is in
        W_H, W_V, W_HV, W_self = utils_numba.get_weights_for_directions(abs_offset_x, abs_offset_y, self.model_height,
                                                                self.model_width)

          # the normalized weight of this crop

        W = np.zeros((self.num_models, self.model_height, self.model_width), dtype=np.float32)

        temp_means = np.zeros_like(prev_means)
        temp_ages = np.zeros_like(prev_means)
        W, temp_means, temp_ages, cond_horizontal, cond_vertical, cond_diagonal, cond_self = \
            utils_numba.compensate_mean_and_age(temp_means, temp_ages, prev_means, prev_ages, W, W_H, W_V, W_HV, W_self,
                                             self.x_grid_coords, self.y_grid_coords, prev_x_grid_coords, prev_y_grid_coords,
                                             offset_x, offset_y, self.model_height, self.model_width)

        # save the copmpensated means and ages and normalized according to the weights
        self.means[W != 0] = 0

        self.ages[:] = 0
        W[W == 0] = 1
        self.means += temp_means / W
        self.ages += temp_ages / W

        #  same shit for variance
        temp_var = utils_numba.compensate_var(prev_vars, prev_means, self.means, W_H, W_V, W_HV, W_self,
                                              self.x_grid_coords, self.y_grid_coords, prev_x_grid_coords,
                                              prev_y_grid_coords, offset_x, offset_y, cond_horizontal, cond_vertical,
                                              cond_diagonal, cond_self)

        self.vars = temp_var / W  # same normalization
        cond = (prev_y_grid_coords < 1) | (prev_y_grid_coords >= self.model_height - 1) | (prev_x_grid_coords < 1) | (
                prev_x_grid_coords >= self.model_width - 1)  # if new grid that was not intreduce before
        self.vars[:, self.y_grid_coords[cond],
        self.x_grid_coords[cond]] = self.var_init  # update the new grid with the init variance
        self.ages[:, self.y_grid_coords[cond], self.x_grid_coords[cond]] = 0
        corrected_age = self.ages * np.exp(-self.lam * (self.vars - self.theta_v))
        self.ages[self.vars > self.theta_v] = corrected_age[self.vars > self.theta_v]  # added age reduction
        self.vars[self.vars < self.var_trim] = self.var_trim  # triming the variance from below

        return self.get_models()


class StatisticalModel(BaseModel):
    """
    statistical model that is responsible to decide each models to update according to eq (1) (2) (3) and
    choose foreground
    """

    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim, age_trim, theta_s,
                 theta_d=4,
                 dynamic=False, probs_params=None, sensitivity=False, suppress=False):
        super(StatisticalModel, self).__init__(num_models, model_height, model_width, block_size, var_init, var_trim)
        self.age_trim = age_trim
        self.theta_s = theta_s
        self.theta_d = theta_d
        self.matrix_theta_d = None
        self.dynamic = dynamic
        self.probs_params = probs_params
        self.sensitivity = sensitivity
        self.suppress = suppress

        self.temporal_property = None
        self.spatial_property = None

    def init(self):
        super(StatisticalModel, self).init()
        self.temporal_property = np.zeros((self.model_height * self.block_size, self.model_width * self.block_size),
                                          dtype=np.float32)
        self.spatial_property = np.zeros((self.model_height * self.block_size, self.model_width * self.block_size),
                                         dtype=np.float32)

    def update(self, var_init, var_trim, age_trim, theta_s, theta_d, dynamic, probs_params, sensitivity, suppress):
        super(StatisticalModel, self).update(var_init, var_trim)
        self.age_trim = age_trim
        self.theta_s = theta_s
        self.theta_d = theta_d
        self.dynamic = dynamic
        self.probs_params = probs_params
        self.sensitivity = sensitivity
        self.suppress = suppress

    def choose_models_for_update(self, cur_mean, com_means, com_vars, com_ages):
        """
        chose models for updating
        :param cur_mean:   each pixel contains the current (observable) mean of its grid
        :param com_means:   compensated means
        :param com_vars:    compensated vars
        :param com_ages:    compensated ages
        :return: the chosen models for updating
        """
        models_with_max_age = self.num_models - np.argmax(com_ages[::-1], axis=0).reshape(-1) - 1  # find maximums of ages
        maxes = np.max(com_ages, axis=0)
        h, w = self.model_height, self.model_width
        jj, ii = np.arange(h * w) // w, np.arange(h * w) % w  # indices of grids

        ii, jj = ii[models_with_max_age != 0], jj[
            models_with_max_age != 0]  # indices of grids with max age in index larger then zero, probably because chosen model (apperent model) need to be in first index
        models_with_max_age = models_with_max_age[models_with_max_age != 0]  # same
        com_ages[
            models_with_max_age, jj, ii] = 0  # initiate models in those indices since there should be new candidate models there
        com_ages[0] = maxes  # store the max ages in the first entry of the 3d array

        # same for means and vars
        com_means[0, jj, ii] = com_means[models_with_max_age, jj, ii]
        com_means[models_with_max_age, jj, ii] = cur_mean[jj, ii]

        com_vars[0, jj, ii] = com_vars[models_with_max_age, jj, ii]
        com_vars[models_with_max_age, jj, ii] = self.var_init

        # choose model to update according to e equations 7,8
        model_index = np.ones(cur_mean.shape).astype(int)
        cond1 = np.power(cur_mean - com_means[0], 2) < self.theta_s * com_vars[0]

        cond2 = np.power(cur_mean - com_means[1], 2) < self.theta_s * com_vars[1]
        model_index[cond1] = 0
        model_index[cond2 & ~cond1] = 1
        com_ages[1][(~cond1) & (~cond2)] = 0

        models_to_update = np.arange(self.means.shape[0]).reshape(-1, 1, 1) == model_index  # which model to take as a 3d matrix with trues and falses in the entries
        return models_to_update, model_index

    def update_vars(self, com_vars, alpha, gray_image, models_to_update, model_index):
        """
        update the variance according to eq (2)
        :param com_vars: compensated vars
        :param alpha: the coefficient of mu in eq (1): a_com(t-1) / [a_com(t-1) + 1]
        :param gray_image: the image
        :param models_to_update: first output of "choose_models_to_update"
        :param model_index: second output of "choose_models_to_update"
        """
        h, w = self.model_height, self.model_width
        jj, ii = np.arange(h * w) // w, np.arange(h * w) % w
        mns = utils_numba.get_chosen_means(self.means, model_index, jj, ii)
        mns = utils_numba.reshape_to_2d_array_numba(mns, (self.model_height, self.model_width))
        big_mean_index = utils_numba.enlarge_pixels(mns, self.block_size)  # extande the chosen models means upon the whole grid
        maxes = utils_numba.rebinMax(np.power(gray_image - big_mean_index, 2),
                                          (self.block_size, self.block_size))  # calc V for each grid for chosen model
        self.vars = com_vars * alpha + (1 - alpha) * maxes
        self.vars[(self.vars < self.var_init) & models_to_update & (self.ages == 0)] = self.var_init
        self.vars[(self.vars < self.var_trim) & models_to_update] = self.var_trim

    def update_ages(self, com_ages, models_to_update):
        """
        update ages according to eq (3)
        :param com_ages: compensated ages
        :param models_to_update: first output of "choose_models_to_update"
        """
        self.ages = com_ages.copy()
        self.ages[models_to_update] += 1
        self.ages[models_to_update & (self.ages > self.age_trim)] = self.age_trim

    def update_models(self, gray_image, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages):
        """
        update means, vars and ages of models according to eq (1) (2) (3)
        :param gray_image: image
        :param models_to_update: first output of "choose_models_to_update"
        :param model_index: second output of "choose_models_to_update"
        :param cur_mean: each pixel contains the current (observable) mean of its grid
        :param com_means: compensated means
        :param com_vars: compensated vars
        :param com_ages: compensated ages
        """
        # calculate coefficients
        alpha = utils_numba.get_alpha(com_ages, models_to_update)

        self.means = utils_numba.update_means(com_means, alpha, cur_mean)
        self.update_vars(com_vars, alpha, gray_image, models_to_update, model_index)
        self.update_ages(com_ages, models_to_update)

    def mix_updating_and_foreground(self, gray, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages):
        """
        uodate means, then choose foreground, then update vars and ages
        :param gray: image
        :param models_to_update: first output of "choose_models_to_update"
        :param model_index: second output of "choose_models_to_update"
        :param cur_mean: each pixel contains the current (observable) mean of its grid
        :param com_means: compensated means
        :param com_vars: compensated vars
        :param com_ages: compensated ages
        :return: chosen pixels to be foregrounded
        """
        alpha = utils_numba.get_alpha(com_ages, models_to_update)
        self.means = utils_numba.update_means(com_means, alpha, cur_mean)
        out = self.choose_foreground(gray)

        self.update_vars(com_vars, alpha, gray, models_to_update, model_index)

        self.update_ages(com_ages, models_to_update)
        return out

    def choose_foreground(self, gray):
        """
        select foreground pixels
        :param gray: image
        :return: chosen foreground pixels
        """
        big_mean = utils_numba.enlarge_pixels(self.means[0], self.block_size)  # current appearing mean extended as previouse
        big_ages = utils_numba.enlarge_pixels(self.ages[0], self.block_size)  # same for ages
        big_vars = utils_numba.enlarge_pixels(self.vars[0], self.block_size)  # same for vars

        out = utils_numba.calc_by_thresh(gray, big_mean, big_vars, big_ages, self.theta_d)
        if self.probs_params is not None:
            self.probs_params["neighborhood_size"] = tuple(self.probs_params["neighborhood_size"])  # TODO:otherwise have problems with numba, make more beautiful
            out = utils_numba.calc_probability(out, self.temporal_property, self.spatial_property, **self.probs_params)
        if self.suppress:
            out = utils.suppression(gray, out, self.theta_d, big_mean, big_vars)
        return out

    def get_foreground(self, gray, com_means, com_vars, com_ages):
        """
        update models and choosing foreground
        :param gray: image
        :param com_means: compensated means
        :param com_vars: compensated vars
        :param com_ages: compensated ages
        :return: chosen foreground pixels
        """
        self.num_frames += 1
        cur_mean = utils_numba.rebinMean(gray, (self.block_size, self.block_size))

        models_to_update, model_index = self.choose_models_for_update(cur_mean, com_means, com_vars, com_ages)

        if self.sensitivity == "mixed":
            out = self.mix_updating_and_foreground(gray, models_to_update, model_index, cur_mean, com_means, com_vars,
                                                   com_ages)
        elif self.sensitivity:
            out = self.choose_foreground(gray)
            self.update_models(gray, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages)

        else:
            self.update_models(gray, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages)
            out = self.choose_foreground(gray)
        return out
