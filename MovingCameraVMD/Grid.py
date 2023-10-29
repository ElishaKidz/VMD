import numpy as np
from StatisticalModels import CompensationModel, StatisticalModel
import copy
import scipy


class Grid:
    def __init__(self, x0, y0, x1, y1, values: np.array = None, lam=0.001, theta_d=4, theta_v=50 * 50, init_age=1,
                 var_init=20*20, truncate_age=30, theta_s=2):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.center_x = x0 + (x1 - x0) // 2
        self.center_y = y0 + (y1 - y0) // 2
        self.center = [self.center_x, self.center_y, 1]

        self.com_model = CompensationModel(lam, theta_v)

        self.app_model = StatisticalModel(init_age, var_init, truncate_age)
        self.can_model = StatisticalModel(init_age, var_init, truncate_age)
        self.app_model.init_model(values)
        self.can_model.init_model(values)

        self.theta_s = theta_s
        self.theta_d = theta_d

        self.values = None
        self.M = None
        self.update_values(values)

    def update_values(self, values):
        """
        update the values of the grid
        :param values: the pixel values of the grid
        """
        self.values = values
        self.M = 1 / ((self.x1 - self.x0) * (self.y1 - self.y0)) * np.sum(values)

    def get_coords(self):
        return [self.x0, self.y0, self.x1, self.y1]

    def get_prev_params(self):
        return self.app_model.get_prev_params()

    def calc_V(self, model: StatisticalModel):
        """
        calculate V as in eq 5
        :param model: the chosen statistical model
        :return: V
        """
        v_mat = np.power(model.mean - self.values, 2)
        return np.max(v_mat)

    def compensate_model(self, weights, means, vars, ages):
        """
        see com_model.compensate_model docstring
        """
        self.com_model.compensate_model(weights, means, vars, ages)

    def update_model(self, model: StatisticalModel):
        """
        update the chosen model according to equations 1, 2, 3
        :param model: the chosen model to update
        """
        mn, vr, age = self.com_model.get_compensated_params()
        model.update_mean(mn, self.M, age)
        V = self.calc_V(model)
        model.update_var(vr, V, age)
        model.update_age(age)

    def choose_update_and_swap_models(self):
        """
        choose model to update and update it, then swap models if necessary
        """
        if (self.M - self.app_model.mean) ** 2 < self.theta_s * self.app_model.variance:
            self.update_model(self.app_model)
        elif (self.M - self.can_model.mean) ** 2 < self.theta_s * self.can_model.variance:
            self.update_model(self.can_model)
        else:
            self.can_model.init_model(self.values)

        if self.can_model.age > self.app_model.age:
            self.app_model = copy.copy(self.can_model)
            self.can_model.init_model(self.values)
        else:
            self.can_model.age += 1

    def get_threshold_mat(self):
        chosen = np.zeros_like(self.values)
        chosen[(self.values - self.app_model.mean) ** 2 > self.theta_d * self.app_model.variance] = 255
        return chosen

    def get_prob_mat(self):
        probs = scipy.stats.norm(self.app_model.mean, self.app_model.variance).cdf(self.values)
        return probs


