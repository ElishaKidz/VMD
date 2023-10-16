import numpy as np
import cv2 as cv
from utils import cell_neighbors

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
    def __init__(self,neighborhood_size:tuple = 1) -> None:
        self.neighborhood_size = neighborhood_size
        self.frames_history = None
    
    def __call__(self, frame):
        if self.frames_history is None:
            self.frames_history = np.expand_dims(frame,axis=0)
            return frame
        
        else:
            background = np.mean(self.frames_history,axis=0)
            foreground = np.zeros(background.shape,dtype=np.uint8)
            for row in range(frame.shape[0]):
                for col in range(frame.shape[1]):
                    pixel_value = frame[row][col]
                    pixel_nighborhood_background_values = cell_neighbors(background,row,col,self.neighborhood_size)
                    min_distance_of_pixel_to_nighborhood_background = np.min(np.absolute(pixel_value-pixel_nighborhood_background_values))
                    foreground[row,col] = min_distance_of_pixel_to_nighborhood_background
            
            self.frames_history = np.append(self.frames_history,np.expand_dims(frame,axis=0),axis=0)
            return foreground
    


