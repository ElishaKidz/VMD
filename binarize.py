import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
binarizers = {}


def register(cls):
    binarizers[cls.__name__] = cls


@register
class DilateErodeBinarizer:
    def __init__(self, diff_frame_threshold: int = 100, dilate_kernel_size=(15, 15), erode_kernel_size=(2, 2),
                 dilate_kwargs: dict = None, erode_kwargs: dict = None) -> None:
        self.diff_frame_threshold = diff_frame_threshold
        self.dilate_kernel = np.ones(dilate_kernel_size)
        self.erode_kernel = np.ones(erode_kernel_size)
        self.dilate_kwargs = dilate_kwargs if dilate_kwargs is not None else {}
        self.erode_kwargs = erode_kwargs if erode_kwargs is not None else {}
        
    def __call__(self, gray_frame):
        thresh_frame = cv.threshold(src=gray_frame, thresh=self.diff_frame_threshold, maxval=255, type=cv.THRESH_BINARY)[1]
        thresh_frame = cv.erode(thresh_frame, self.erode_kernel, **self.erode_kwargs)
        thresh_frame = cv.dilate(thresh_frame, self.dilate_kernel, **self.dilate_kwargs)
        return thresh_frame
    