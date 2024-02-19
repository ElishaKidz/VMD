import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from SoiUtils.interfaces import Updatable
from SoiUtils.output_recorder import global_output_recorder

binarizers = {}

def register(name):
    def register_func_fn(cls):
        binarizers[name] = cls
        return cls
    return register_func_fn


@register("SimpleThresholdBinarizer")
class SimpleThresholdBinarizer(Updatable):
    DEFAULT_ARGS = {'thresh':30,'maxval':255,'type':cv.THRESH_BINARY}
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(SimpleThresholdBinarizer.DEFAULT_ARGS,**kwargs) # give priority to the inserted arguments
        
    def __call__(self, gray_frame):
        # thresh_frame = cv.threshold(src=gray_frame, thresh=self.diff_frame_threshold, maxval=255, type=cv.THRESH_BINARY)[1]
        binary_frame = cv.threshold(src=gray_frame, **self.kwargs)[1]
        return binary_frame
    

    def update(self, **kwargs):
        self.kwargs = dict(SimpleThresholdBinarizer.DEFAULT_ARGS,**kwargs)


@register("DynamicThresholdBinarizer")
class DynamicThresholdBinarizer(SimpleThresholdBinarizer):
    def __init__(self, **kwargs):
        super(DynamicThresholdBinarizer, self).__init__(**kwargs)
        assert 'thresh' in kwargs.keys(), "The constructor excpects thresh to exist in the parameters to the threhsold function"
        self.diff_frame_threshold = kwargs['thresh']

    def __call__(self, gray_frame):
        
        self.diff_frame_threshold = self.calculate_dynamic_threshold(gray_frame,self.diff_frame_threshold)
        self.kwargs['thresh'] = self.diff_frame_threshold
        return super(DynamicThresholdBinarizer,self).__call__(gray_frame)
    
    @staticmethod
    def calculate_dynamic_threshold(gray_frame,current_threshold):
        flat_array = gray_frame.flatten()
        mn, std = np.mean(flat_array), np.std(flat_array)
        statistic_thresh = mn + 5*std
        new_threshold = int(0.5 * statistic_thresh + 0.5 * current_threshold)
        return new_threshold
    

@register("NormalizedBinarizer")
class NormalizedBinarizer(SimpleThresholdBinarizer):
    def __init__(self,**kwargs):
        super(NormalizedBinarizer, self).__init__(**kwargs)

    def __call__(self, gray_frame):
        if not np.any(gray_frame):
            return gray_frame
        
        foreground = gray_frame.astype(np.float32)
        foreground = (foreground - np.min(foreground)) / (np.max(foreground) - np.min(foreground)) * 255
        foreground = self.gammaCorrection(foreground.astype(np.uint8), 2.2)
        return super(NormalizedBinarizer, self).__call__(foreground.astype(np.uint8))

    @staticmethod
    def gammaCorrection(src, gamma):
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv.LUT(src, table)
