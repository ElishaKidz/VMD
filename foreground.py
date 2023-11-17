import numpy as np
import cv2 as cv
from skimage.util import view_as_windows
from VMD.MovingCameraForegroundEstimetor.ForegroundEstimetor import ForegroundEstimetor
from time import time
from numba import jit, prange

foreground_estimators = {}


def register(name):
    def register_func_fn(cls):
        foreground_estimators[name] = cls
        return cls

    return register_func_fn


@register("MovingCameraForegroundEstimetor")
class MovingCameraForegroundEstimetor(ForegroundEstimetor):
    def __init__(self, num_models=2, block_size=4, var_init=20.0 * 20.0, var_trim=5.0 * 5.0, lam=0.001,
                 theta_v=50.0 * 50.0, age_trim=30, theta_s=2, theta_d=2, dynamic=False, calc_probs=False,
                 sensitivity="mixed", suppress=False, smooth=True):
        super(MovingCameraForegroundEstimetor, self).__init__(num_models, block_size, var_init, var_trim, lam, theta_v,
                                                              age_trim, theta_s, theta_d, dynamic, calc_probs,
                                                              sensitivity, suppress, smooth)


@register("MedianForegroundEstimation")
class MedianForegroundEstimation:
    def __init__(self, num_frames=10) -> None:
        self.frames_history = []
        self.num_frames = num_frames

    def __call__(self, frame):
        if len(self.frames_history) == 0:
            foreground = frame

        else:
            background = np.median(self.frames_history, axis=0).astype(dtype=np.uint8)
            foreground = cv.absdiff(frame, background)

            if len(self.frames_history) >= self.num_frames:
                self.frames_history = list(self.frames_history[-self.num_frames:])

        self.frames_history.append(frame)
        return foreground

    def reset(self):
        self.frames_history = []




@register("MOG2")
class MOG2():
    def __init__(self, **kwargs):
        self.fgbg = cv.createBackgroundSubtractorMOG2(**kwargs)
    
    def __call__(self, frame):
        fgmask = self.fgbg.apply(frame)
        return fgmask


@register("PESMODForegroundEstimation")
class PESMODForegroundEstimation():
    def __init__(self, neighborhood_matrix: tuple = (3, 3), num_frames=10, suppress=False) -> None:
        self.neighborhood_matrix = neighborhood_matrix
        self.frames_history = None
        self.num_frames = num_frames
        self.suppress = suppress
        self.filter_w, self.filter_h = self.neighborhood_matrix
        self.pad_w = int(self.filter_w / 2)
        self.pad_h = int(self.filter_h / 2)

        PESMODForegroundEstimation.compile_difference()

    @staticmethod
    def compile_difference():
        temp1 = np.zeros((400, 400, 3, 3), dtype=np.float64)
        temp2 = np.zeros((3, 3, 400, 400), dtype=np.uint8)
        difference(temp1, temp2)

    def __call__(self, frame):
        if self.frames_history is None:
            self.frames_history = np.expand_dims(frame,axis=0)
            self.window_sum = self.frames_history[-1].astype(np.int32)
            foreground = frame
        else:
            background = self.window_sum / len(self.frames_history)
            padded_background = np.pad(background, ((self.pad_w, self.pad_w), (self.pad_h, self.pad_h)),
                                       constant_values=(np.inf, np.inf))
            background_patches = view_as_windows(padded_background, self.neighborhood_matrix)

            w, h = frame.shape

            broadcasted_frame = np.broadcast_to(frame, (self.filter_w, self.filter_h, w, h))

            foreground = difference(background_patches, broadcasted_frame)

            if self.suppress:
                mn = np.mean(frame)
                std = np.std(frame)
                foreground[frame < mn + std] = 0

            self.frames_history = np.append(self.frames_history,np.expand_dims(frame, axis=0), axis=0)

            if self.frames_history.shape[0]> self.num_frames:
                self.window_sum -= self.frames_history[0]
                self.frames_history = self.frames_history[1:]

            self.window_sum += self.frames_history[-1]
        return foreground

    def reset(self):
        self.frames_history = None



@jit(parallel=True)
def difference(background_patches, broadcasted_frame):
    h, w = broadcasted_frame.shape[2], broadcasted_frame.shape[3]

    frame = np.transpose(broadcasted_frame, (2, 3, 0, 1))

    foreground = np.empty((h, w), dtype=np.uint8)

    for i in prange(h):
        for j in prange(w):
            diff = np.abs(np.subtract(background_patches[i, j], frame[i, j]))
            min_diff = np.min(diff)
            foreground[i, j] = min_diff
    return foreground
