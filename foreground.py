import numpy as np
import cv2 as cv


foreground_estimators = {}

def register(cls):
    foreground_estimators[cls.__name__] = cls

@register
class MedianForegroundEstimation:
    def __init__(self,num_frames=10) -> None:
        self.frames_history = []
        self.num_frames = num_frames
    def __call__(self,frame):

        if len(self.frames_history) == 0:
            foreground = frame

        else:
            background = np.median(self.frames_history, axis=0).astype(dtype=np.uint8)
            foreground = cv.absdiff(frame, background)

            if len(self.frames_history)>=self.num_frames:
                self.frames_history = list(self.frames_history[-self.num_frames+1:])

        self.frames_history.append(frame)
        return foreground

@register
class MOG2():
    def __init__(self,**kwargs):
        self.fgbg = cv.createBackgroundSubtractorMOG2(**kwargs)
    
    def __call__(self, frame):
        fgmask = self.fgbg.apply(frame)
        return fgmask


@register
class PESMODForegroundEstimation():
    def __init__(self,neighborhood_size:tuple = (3,3)) -> None:
        self.neighborhood_size = neighborhood_size
        self.frames_history = None
    
    def __call__(self, frame):
        pass



