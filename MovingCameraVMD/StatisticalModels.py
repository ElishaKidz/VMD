import numpy as np


class CompensationModel:
    def __init__(self, lam=0.001, theta_v=50*50):
        self.lam = lam
        self.theta_v = theta_v
        self.variance = None
        self.mean = None
        self.age = None

    def compensate_model(self, weights, means, vrs, ages):  # model compensation using pre calculated weights
        """
        model compensation
        compensate model according to calculated movement
        :param weights: the relative part of each grid
        :param means:  the means of overlapping grids
        :param vrs: the vars of overlapping grids
        :param ages: the ages of overlapping grids
        """
        self.mean = sum([w*m for w, m in zip(weights, means)])

        self.variance = sum([w * (v_p + (mm**2) - (self.mean ** 2)) for w, v_p, mm in zip(weights, vrs, means)])

        self.age = sum([w*a for w, a in zip(weights, ages)])

        if self.variance > self.theta_v:
            self.age = self.age * np.exp(-self.lam * (self.variance - self.theta_v))
            x = 5

    def get_compensated_params(self):
        return self.mean, self.variance, self.age


class StatisticalModel:
    def __init__(self, age_init=1, var_init=20*20, truncate_age=30):
        self.age_init = age_init
        self.var_init = var_init
        self.truncate_age = truncate_age

        self.prev_mean = None
        self.prev_variance = None
        # self.prev_age = None
        self.prev_age = 1

        self.mean = None
        self.variance = None
        # self.age = None
        self.age = 1
        self.values = None

    def init_model(self, values: np.array):
        """
        initiate the model when necessary
        :param values: the values of the grid the model belongs to
        """
        self.prev_mean = np.mean(values)
        self.prev_variance = np.var(values)  # TODO: according to paper needs to be var_init
        # self.prev_age = self.age_init

        self.mean = self.prev_mean
        self.variance = self.prev_variance
        self.age = 1    # self.prev_age

    def update_mean(self, mean, M, age):   # eq 1
        """
        final mean update according to equation 1
        :param mean: compensated_mean
        :param M: M parameter
        """
        # if self.mean:
        #     self.prev_mean = self.mean
        left = (age/(age+1)) * mean
        right = (1/(age + 1)) * M
        self.mean = left + right
        self.prev_mean = self.mean

    def update_var(self, vr, V, age):   # eq 2
        """
        final var update according to equation 2
        :param vr: compensated var
        :param V: V parameter
        """
        # if self.variance:
        #     self.prev_variance = self.variance
        left = (age/(age+1)) * vr
        right = (1/(age + 1)) * V
        self.variance = left + right
        self.prev_variance = self.variance

    def update_age(self, age):   # eq 3
        """
        final age update according to equation 3
        :param age: compensated age
        """
        # if self.age:
        #     self.prev_age = self.age
        self.age = age + 1

        if self.age > self.truncate_age:
            self.age = self.truncate_age
        self.prev_age = self.age

    def get_prev_params(self):
        return self.prev_mean, self.prev_variance, self.prev_age

