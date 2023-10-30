import numpy as np
import cv2 as cv
from skimage.util import view_as_windows
foreground_estimators = {}


def register(name):
    def register_func_fn(cls):
        foreground_estimators[name] = cls
        return cls
    return register_func_fn


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


@register("MOG2")
class MOG2():
    def __init__(self, **kwargs):
        self.fgbg = cv.createBackgroundSubtractorMOG2(**kwargs)
    
    def __call__(self, frame):
        fgmask = self.fgbg.apply(frame)
        return fgmask


@register("PESMODForegroundEstimation")
class PESMODForegroundEstimation():
    def __init__(self, neighborhood_matrix: tuple = (3, 3), num_frames=10) -> None:
        self.neighborhood_matrix = neighborhood_matrix
        self.frames_history = None
        self.num_frames = num_frames
    
    def __call__(self, frame):
        if self.frames_history is None:
            self.frames_history = np.expand_dims(frame,axis=0)
            self.window_sum = self.frames_history[-1].astype(np.int32)
            return frame

        else:
            filter_w, filter_h = self.neighborhood_matrix
            pad_w = int(filter_w / 2)
            pad_h = int(filter_h / 2)

            background = self.window_sum / len(self.frames_history)
            padded_background = np.pad(background, ((pad_w, pad_w), (pad_h, pad_h)), constant_values=(np.inf, np.inf))
            background_patches = view_as_windows(padded_background, self.neighborhood_matrix)

            w, h = frame.shape
            
            #foreground = np.abs(background_patches.reshape(-1) - np.repeat(frame, filter_w * filter_h)).reshape(w, h, -1).min(axis=2).astype(np.uint8)

            foreground = np.abs(np.transpose(background_patches, (2, 3, 0, 1)) - 
                               np.broadcast_to(frame, (filter_w, filter_h, w, h))).transpose((2, 3, 0 ,1)).reshape(w, h, -1).min(axis=2).astype(np.uint8)


            self.frames_history = np.append(self.frames_history,np.expand_dims(frame, axis=0), axis=0)

            if self.frames_history.shape[0]> self.num_frames:
                self.window_sum -= self.frames_history[0]
                self.frames_history = self.frames_history[1:]

            self.window_sum += self.frames_history[-1]

            return foreground