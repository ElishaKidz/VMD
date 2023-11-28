import time

import numpy as np
from VMD.MovingCameraForegroundEstimetor.MathematicalModels import CompensationModel, StatisticalModel
from VMD.MovingCameraForegroundEstimetor.KLTWrapper import KLTWrapper
import cv2
from SoiUtils.interfaces import Resetable, Updatable


class ForegroundEstimetor(Resetable, Updatable):
    """
    Moving camera foreground estimator class, gets a grayscale frame and returns foreground pixels or
    probabilities for pixels tp be foreground
    """
    def __init__(self, num_models: int = 2, block_size: int = 4, var_init: float = 20.0*20.0, var_trim: float = 5.0*5.0,
                 lam: float = 0.001, theta_v: float = 50.0*50.0, age_trim: float = 30, theta_s=2, theta_d=2,
                 dynamic=False, calc_probs=False, sensitivity="mixed", suppress=False, smooth=True):
        """
        :param num_models: number of models for each pixel to use, minimum possible value is 2
        :param block_size: the size of a square block in the grid the image dims most be able to divide by this param
        :param var_init: initial value for variance of models
        :param var_trim: lower bound on variance
        :param lam: lambda param for age correction after compensation
        :param theta_v: threshold for variance for age correction as in eq (15) in the paper
        :param age_trim: upper bound on age value
        :param theta_s: the threshold for choosing models to update as in eq (7) (8) in the paper
        :param theta_d: the threshold for foreground choosing as in eq (16). Not relevant if calc_prob is False
        :param calc_probs: if true, return the probability for each pixel to be foreground, if false return which pixel
                        is foreground using eq (16)
        :param sensitivity: decide when to update the models and when to predict foreground
        3 possible values:
        True: choose foreground then update - more sensitive to changes but also to noise
        False: update then foreground - this is what the paper stats. less sensetive to changes but also to noise
        'mixed': the implementation of the original code. update mean, the foreground, then update vars and ages
        :param smooth: if True smooth frame using gaussian blur
        """
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
        self.dynamic = dynamic
        self.suppress = suppress

        self.calc_probs = calc_probs
        self.sensitivity = sensitivity
        self.smooth = smooth

        self.num_frames = 0
        self.com_time = 0
        self.stat_time = 0
        self.h_time = 0
        self.total_time = 0

        self.compile()

    def update(self, num_models: int, block_size: int, var_init: float, var_trim: float,
                 lam: float, theta_v: float, age_trim: float, theta_s, theta_d,
                 dynamic, calc_probs, sensitivity, suppress, smooth, **kwargs):
        print("vmd update")
        self.var_init = var_init
        self.var_trim = var_trim
        self.lam = lam
        self.theta_v = theta_v
        self.age_trim = age_trim
        self.theta_s = theta_s
        self.theta_d = theta_d
        self.dynamic = dynamic
        self.suppress = suppress

        self.calc_probs = calc_probs
        self.sensitivity = sensitivity
        self.smooth = smooth

        if self.num_models != num_models or self.block_size != block_size:
            self.num_models = num_models
            self.block_size = block_size
            self.reset()

        if self.compensation_models is not None:
            self.compensation_models.update(var_init, var_trim, lam, theta_v)

        if self.statistical_models is not None:
            self.statistical_models.update(var_init, var_trim, age_trim, theta_s, theta_d, dynamic, calc_probs,
                                           sensitivity, suppress)

    def reset(self):
        print("vmd reset")
        self.is_first = True

        self.homography_calculator = KLTWrapper()
        self.compensation_models = None
        self.statistical_models = None
        self.model_height = None
        self.model_width = None

        self.num_frames = 0
        self.com_time = 0
        self.stat_time = 0
        self.h_time = 0
        self.total_time = 0

    def compile(self):
        gray = np.ones((512, 640), dtype=np.uint8)
        self.first_pass(gray)
        self.reset()

    def first_pass(self, gray_frame: np.array):
        """
        activate when the frame is the first frame, initialize all the modules
        :param gray_frame: a gray frame
        :return: foreground estimation
        """
        self.is_first = False
        h, w = gray_frame.shape
        if h % self.block_size or w % self.block_size:
            raise IOError("Image dims most be dividable by block_size")
        self.model_height, self.model_width = h // self.block_size, w // self.block_size   # calc num of grid in each axis

        # create models
        if self.compensation_models is None or self.statistical_models is None:
            self.compensation_models = CompensationModel(self.num_models, self.model_height, self.model_width,
                                                         self.block_size, self.var_init, self.var_trim, self.lam,
                                                         self.theta_v)
            self.statistical_models = StatisticalModel(self.num_models, self.model_height, self.model_width,
                                                       self.block_size, self.var_init, self.var_trim, self.age_trim,
                                                       self.theta_s, self.theta_d, self.dynamic, self.calc_probs,
                                                       self.sensitivity, self.suppress)

        # initialize homography calculator
        self.homography_calculator.init(gray_frame)

        # initialize models
        self.statistical_models.init()
        inited_means, inited_vars, inited_ages = self.statistical_models.get_models()

        com_means, com_vars, com_ages = self.compensation_models.init(inited_means, inited_vars,
                                                                      inited_ages)
        foreground = self.statistical_models.get_foreground(gray_frame, com_means, com_vars, com_ages)
        return foreground

    def get_foreground(self, gray_frame):
        """
        do the full pipeline, calculating foreground
        :param gray_frame: a gray frame
        :return: foreground estimation
        """
        self.num_frames += 1
        s = time.time()
        new_h, new_w = gray_frame.shape
        if new_w // self.block_size != self.model_width or new_h // self.block_size != self.model_height:
            self.reset()

        if self.is_first:   # if first frame initialize
            x = self.first_pass(gray_frame)
            e = time.time()
            self.total_time += e - s
            return x

        if self.smooth:
            gray_frame = cv2.medianBlur(gray_frame, 5)
            # gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        # compensate
        prev_means, prev_vars, prev_ages = self.statistical_models.get_models()
        s0 = time.time()
        H = self.homography_calculator.RunTrack(gray_frame)
        e0 = time.time()
        self.h_time += e0 - s0

        s1 = time.time()
        com_means, com_vars, com_ages = self.compensation_models.compensate(H, prev_means, prev_vars, prev_ages)
        e1 = time.time()
        self.com_time += e1 - s1

        # estimate foreground
        foreground = self.statistical_models.get_foreground(gray_frame, com_means, com_vars, com_ages)
        e2 = time.time()
        self.stat_time += e2 - e1

        e = time.time()
        self.total_time += e - s

        return foreground

    def __call__(self, gray_frame):
        return self.get_foreground(gray_frame)
