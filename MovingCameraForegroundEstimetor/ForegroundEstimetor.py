import numpy as np
from VMD.MovingCameraForegroundEstimetor.MathematicalModels import CompensationModel, StatisticalModel
from VMD.MovingCameraForegroundEstimetor.KLTWrapper import KLTWrapper


class ForegroundEstimetor:
    def __init__(self, num_models, block_size, var_init, var_trim, lam, theta_v, age_trim, theta_s, theta_d=4,
                 sensetivity=False, smooth=False):
        self.is_first = True

        self.homography_calculator = KLTWrapper()
        self.compensation_models = None
        self.statistical_models = None
        self.model_height = None
        self.model_width = None

        self.num_models = num_models
        self.block_size = block_size
        self.var_init = var_init
        self.var_trim = var_trim
        self.lam = lam
        self.theta_v = theta_v
        self.age_trim = age_trim
        self.theta_s = theta_s
        self.theta_d = theta_d
        self.sensetivity = sensetivity
        self.smooth = smooth


    def first_pass(self, gray_frame: np.array):
        self.is_first = False
        h, w = gray_frame.shape
        if h % self.block_size or w % self.block_size:
            raise IOError("Image dims most be dividable by block_size")
        self.model_height, self.model_width = h // self.block_size, w // self.block_size   # calc num of grid in each axis

        self.compensation_models = CompensationModel(self.num_models, self.model_height, self.model_width,
                                                     self.block_size, self.lam, self.theta_v)
        self.statistical_models = StatisticalModel(self.num_models, self.model_height, self.model_width,
                                                   self.block_size, self.age_trim, self.theta_s, self.theta_d,
                                                   self.sensetivity)

        self.homography_calculator.init(gray_frame)

        self.statistical_models.init(gray_frame)
        inited_means, inited_vars, inited_ages = self.statistical_models.get_models()

        com_means, com_vars, com_ages = self.compensation_models.init(gray_frame, inited_means, inited_vars,
                                                                      inited_ages)
        return self.statistical_models.get_foreground(gray_frame, com_means, com_vars, com_ages)

    def get_foreground(self, gray_frame, H):
        if self.smooth:
            gray_frame = self.imgGray = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        if self.is_first:
            return self.first_pass(gray_frame)

        prev_means, prev_vars, prev_ages = self.statistical_models.get_models()
        H = self.homography_calculator.RunTrack(gray_frame)
        com_means, com_vars, com_ages = self.compensation_models.compensate(H, prev_means, prev_vars, prev_ages)
        foreground = self.statistical_models.get_foreground(com_means, com_vars, com_ages)
        return foreground
