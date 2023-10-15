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
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        if len(self.frames_history) == 0:
            foreground = gray_frame

        else:
            color_background = np.median(self.frames_history, axis=0).astype(dtype=np.uint8)
            gray_background = cv.cvtColor(color_background, cv.COLOR_BGR2GRAY)
            foreground = cv.absdiff(gray_frame, gray_background)

            if len(self.frames_history)>=self.num_frames:
                self.frames_history = list(self.frames_history[-self.num_frames+1:])

        self.frames_history.append(frame)
        return foreground

@register
class MOG2():
    def __init__(self,**kwargs):
        self.fgbg = cv.createBackgroundSubtractorMOG2(**kwargs)
    
    def __call__(self, frame):
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray_frame)
        cv.imshow('fgmask', fgmask) 

        return fgmask


@register
class PESMODForegroundEstimation():
    def __init__(self) -> None:
        pass
    



