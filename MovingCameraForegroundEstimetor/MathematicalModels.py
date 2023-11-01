import numpy as np
import cv2
from VMD.MovingCameraForegroundEstimetor import utils
import scipy


class BaseModel:
    """
    prenet class for the compensation model and the statistics model
    """
    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim):
        """
        see foreground model init for documentation
        """
        self.prev_means = None
        self.vars = None
        self.ages = None
        self.model_height = model_height
        self.model_width = model_width
        self.block_size = block_size
        self.num_models = num_models
        self.var_init = var_init
        self.var_trim = var_trim

    def init(self):
        """
        create zero params of size of the models and the grids
        """
        self.means = np.zeros((self.num_models, self.model_height, self.model_width))
        self.vars = np.zeros((self.num_models, self.model_height, self.model_width))
        self.ages = np.zeros((self.num_models, self.model_height, self.model_width))

    def get_models(self):
        """
        :return: means vars and ages of models
        """
        return self.means, self.vars, self.ages


class CompensationModel(BaseModel):
    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim, lam, theta_v):
        super(CompensationModel, self).__init__(num_models, model_height, model_width, block_size, var_init, var_trim)
        self.lam = lam
        self.theta_v = theta_v

    def init(self, means, vars, ages):
        super(CompensationModel, self).init()
        H = np.identity(3)
        return self.compensate(H,  means, vars, ages)

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
                                x_grid_coords, y_grid_coords, prev_grid_coords_x, prev_grid_coords_y, offset_x, offset_y):
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
        x_overlap_idx = prev_grid_coords_x + np.sign(offset_x).astype(int)  # figure out X direction of jumping, taking the overlaping center index in X
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

        W, temp_means, temp_ages, cond_horizontal, cond_vertical, cond_diagonal, cond_self = \
            self.compensate_mean_and_age(temp_means, temp_ages, prev_means, prev_ages, W, W_H, W_V, W_HV, W_self,
                                     x_grid_coords, y_grid_coords, prev_x_grid_coords, prev_y_grid_coords, offset_x, offset_y)

        # save the copmpensated means and ages and normalized according to the weights
        self.means[W != 0] = 0

        self.ages[:] = 0
        W[W == 0] = 1
        self.means += temp_means / W
        self.ages += temp_ages / W

        #  same shit for variance
        x_overlap_idx = prev_x_grid_coords + np.sign(offset_x).astype(int)  # figure out X direction of jumping, taking the overlaping center index in X
        y_overlap_idx = prev_y_grid_coords + np.sign(offset_y).astype(int)
        temp_var = np.zeros(prev_means.shape)
        temp_var[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] += W_H[y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] * (prev_vars[:, prev_y_grid_coords[cond_horizontal], x_overlap_idx[cond_horizontal]] +
                                                                      np.power(self.means[:, y_grid_coords[cond_horizontal], x_grid_coords[cond_horizontal]] -
                                                                               prev_means[:, prev_y_grid_coords[cond_horizontal],
                                                                               x_overlap_idx[cond_horizontal]],
                                                                               2))  # TODO: not the same calculation. think deeply

        temp_var[:, y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] += W_V[y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] * (prev_vars[:, y_overlap_idx[cond_vertical], prev_x_grid_coords[cond_vertical]] +
                                                                      np.power(self.means[:, y_grid_coords[cond_vertical], x_grid_coords[cond_vertical]] -
                                                                               prev_means[:,
                                                                               y_overlap_idx[cond_vertical], prev_x_grid_coords[cond_vertical]],
                                                                               2))

        temp_var[:, y_grid_coords[cond_diagonal], x_grid_coords[cond_diagonal]] += W_HV[y_grid_coords[cond_diagonal], x_grid_coords[cond_diagonal]] * (prev_vars[:, y_overlap_idx[cond_diagonal], x_overlap_idx[cond_diagonal]] +
                                                                           np.power(self.means[:, y_grid_coords[cond_diagonal],
                                                                                    x_grid_coords[cond_diagonal]] -
                                                                                    prev_means[:, y_overlap_idx[cond_diagonal],
                                                                                    x_overlap_idx[cond_diagonal]],
                                                                                    2))

        temp_var[:, y_grid_coords[cond_self], x_grid_coords[cond_self]] += W_self[y_grid_coords[cond_self], x_grid_coords[cond_self]] * (
                    prev_vars[:, prev_y_grid_coords[cond_self], prev_x_grid_coords[cond_self]] +
                    np.power(self.means[:, y_grid_coords[cond_self], x_grid_coords[cond_self]] -
                             prev_means[:, prev_y_grid_coords[cond_self], prev_x_grid_coords[cond_self]], 2))

        self.vars = temp_var / W  # same normalization
        cond = (prev_y_grid_coords < 1) | (prev_y_grid_coords >= self.model_height - 1) | (prev_x_grid_coords < 1) | (
                    prev_x_grid_coords >= self.model_width - 1)  # if new grid that was not intreduce before
        self.vars[:, y_grid_coords[cond], x_grid_coords[cond]] = self.var_init  # update the new grid with the init variance
        self.ages[:, y_grid_coords[cond], x_grid_coords[cond]] = 0
        # corrected_age = self.ages * np.exp(-self.lam * (self.vars - self.theta_v))
        # self.ages[self.vars > self.theta_v] = corrected_age[self.vars > self.theta_v]   # added age reduction #TODO:fix!!!!!!!
        self.vars[self.vars < self.var_trim] = self.var_trim  # triming the variance from below
        return self.get_models()


class StatisticalModel(BaseModel):
    """
    statistical model that is responsible to decide each models to updates according to eq (1) (2) (3) and
    choose foreground
    """
    def __init__(self, num_models, model_height, model_width, block_size, var_init, var_trim, age_trim, theta_s, theta_d=4,
                 calc_probs=False, sensetivity=False):
        super(StatisticalModel, self).__init__(num_models, model_height, model_width, block_size, var_init, var_trim)
        self.age_trim = age_trim
        self.theta_s = theta_s
        self.theta_d = theta_d
        self.calc_probs = calc_probs
        self.sensetivity = sensetivity

    def init(self):
        super(StatisticalModel, self).init()

    @staticmethod
    def rebinMean(arr, factor):
        # averaging each patch
        sh = arr.shape[0] // factor[0], factor[0], -1, factor[1]   # get number of grids in H. W
        res = arr.reshape(sh).mean(-1).mean(1)
        return res

    @staticmethod
    def rebinMax(arr, factor):
        # identicle to rebin + max
        sh = arr.shape[0] // factor[0], factor[0], -1, factor[1]
        res = arr.reshape(sh).max(-1).max(1)
        return res

    @staticmethod
    def get_alpha(com_ages, models_to_update):
        """
        calc couficient of the paper
        :param com_ages: compensated ages
        :param models_to_update:  the indexes of the chosen models to update
        :return: cofficient
        """
        alpha = com_ages / (com_ages + 1)
        alpha[com_ages < 1] = 0
        alpha[~models_to_update] = 1
        return alpha

    @staticmethod
    def calc_probability(gray, big_mean, big_vars, big_ages):
        """
        calc probability of pixels to be foreground
        :param gray: gray image
        :param big_mean: each pixel has the value of the mean of its grid
        :param big_vars: same for vars
        :param big_ages: same for ages
        :return: probabilities
        """
        cdf = scipy.stats.norm(loc=big_mean, scale=np.sqrt(big_vars)).cdf(gray)
        reverse_cdf = 1 - cdf
        out = np.maximum(cdf, reverse_cdf)
        out[big_ages <= 1] = 0.5
        out *= 255
        out = out.astype(np.uint8)
        return out

    @staticmethod
    def calc_by_thresh(gray, big_means, big_vars, big_ages, theta):
        """
        decide each pixels are foreground by thresholding as in eq (16)
        :param theta: the threshold
        :return: foreground-background matrix
        """
        dist_img = np.power(gray - big_means, 2)
        out = np.zeros(gray.shape).astype(np.uint8)
        out[(big_ages > 1) & (dist_img > theta * big_vars)] = 255
        return out

    def choose_models_for_update(self, cur_mean, com_means, com_vars, com_ages):
        """
        chose models for updating
        :param cur_mean:   each pixel containes the current (observable) mean of its grid
        :param com_means:   compensated means
        :param com_vars:    compensated vars
        :param com_ages:    compensated ages
        :return: the chosen models for updating
        """
        models_with_max_age = self.num_models - np.argmax(com_ages[::-1], axis=0).reshape(-1) - 1  # find maximums of ages
        maxes = np.max(com_ages, axis=0)
        h, w = self.model_height, self.model_width
        jj, ii = np.arange(h * w) // w, np.arange(h * w) % w  # indices of grids

        ii, jj = ii[models_with_max_age != 0], jj[models_with_max_age != 0]  # indices of grids with max age in index larger then zero, probably because chosen model (apperent model) need to be in first index
        models_with_max_age = models_with_max_age[models_with_max_age != 0]  # same
        com_ages[models_with_max_age, jj, ii] = 0  # initiate models in those indices since there should be new candidate models there
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

        models_to_update = np.arange(self.means.shape[0]).reshape(-1, 1,
                                                                1) == model_index  # which model to take as a 3d matrix with trues and falses in the entries
        return models_to_update, model_index

    def update_means(self, com_means, alpha, cur_mean):
        """
        update the means according to eq (1)
        :param com_means: compensated means
        :param alpha: the cofficient of mu in eq (1): a_com(t-1) / [a_com(t-1) + 1]
        :param cur_mean: the current mean as explained before
        """
        self.means = com_means * alpha + cur_mean * (1 - alpha)    # update mean

    def update_vars(self, com_vars, alpha, gray_image, models_to_update, model_index):
        """
        update the variance accoring to eq (2)
        :param com_vars: compensated vars
        :param alpha: the cofficient of mu in eq (1): a_com(t-1) / [a_com(t-1) + 1]
        :param gray_image: the image
        :param models_to_update: first output of "choose_models_to_update"
        :param model_index: second output of "choose_models_to_update"
        """
        h, w = self.model_height, self.model_width
        jj, ii = np.arange(h * w) // w, np.arange(h * w) % w
        big_mean_index = np.kron(self.means[model_index.reshape(-1), jj, ii].reshape(self.model_height, -1),
                                 np.ones((self.block_size,
                                          self.block_size)))  # extande the chosen models means upon the whole grid
        maxes = StatisticalModel.rebinMax(np.power(gray_image - big_mean_index, 2),
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
        :param cur_mean: each pixel containes the current (observable) mean of its grid
        :param com_means: compensated means
        :param com_vars: compensated vars
        :param com_ages: compensated ages
        """
        # calculate coefficients
        alpha = StatisticalModel.get_alpha(com_ages, models_to_update)
        self.update_means(com_means, alpha, cur_mean)

        self.update_vars(com_vars, alpha, gray_image, models_to_update, model_index)

        self.update_ages(com_ages, models_to_update)

    def mix_updating_and_foreground(self, gray, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages):
        """
        uodate means, then choose foreground, then update vars and ages
        :param gray: image
        :param models_to_update: first output of "choose_models_to_update"
        :param model_index: second output of "choose_models_to_update"
        :param cur_mean: each pixel containes the current (observable) mean of its grid
        :param com_means: compensated means
        :param com_vars: compensated vars
        :param com_ages: compensated ages
        :return: chosen pixels to be foreground
        """
        alpha = StatisticalModel.get_alpha(com_ages, models_to_update)
        self.update_means(com_means, alpha,cur_mean)
        out = self.choose_foreground(gray, com_means, com_vars, com_ages)
        self.update_vars(com_vars, alpha, gray, models_to_update, model_index)
        self.update_ages(com_ages, models_to_update)
        return out

    def choose_foreground(self, gray, com_means, com_vars, com_ages):
        """
        select foreground pixels
        :param gray: image
        :param com_means: compensated means
        :param com_vars: compensated vars
        :param com_ages: compensated ages
        :return: chosen foreground pixels
        """
        big_mean = np.kron(self.means[0], np.ones((self.block_size, self.block_size)))  # current appearing mean extended as previouse
        big_ages = np.kron(self.ages[0], np.ones((self.block_size, self.block_size)))  # same for ages
        big_vars = np.kron(self.vars[0], np.ones((self.block_size, self.block_size)))  # same for vars
        if self.calc_probs:
            return StatisticalModel.calc_probability(gray, big_mean, big_vars, big_ages)
        else:
            return StatisticalModel.calc_by_thresh(gray, big_mean, big_vars, big_ages, self.theta_d)
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
        cur_mean = StatisticalModel.rebinMean(gray, (self.block_size, self.block_size))  # calc each grid mean to decide which model to update, aka calculate M(t)
        models_to_update, model_index = self.choose_models_for_update(cur_mean, com_means, com_vars, com_ages)
        if self.sensetivity == "mixed":
            out = self.mix_updating_and_foreground(gray, models_to_update, model_index, cur_mean, com_means, com_vars,
                                                   com_ages)
        elif self.sensetivity:
            out = self.choose_foreground(gray, com_means, com_vars, com_ages)
            self.update_models(gray, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages)
        else:
            self.update_models(gray, models_to_update, model_index, cur_mean, com_means, com_vars, com_ages)
            out = self.choose_foreground(gray, com_means, com_vars, com_ages)
        return out


