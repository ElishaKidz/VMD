import cv2 as cv
import numpy as np
from SoiUtils.interfaces import Updatable
from SoiUtils.output_recorder import global_output_recorder

morphologies = {}


def register(name):
    def register_func_fn(cls):
        morphologies[name] = cls
        return cls

    return register_func_fn


@register("DilateErodeMorphology")
class DilateErodeMorphology(Updatable):
    def __init__(self, dilate_kernel_size=None, erode_kernel_size=None,
                 dilate_kwargs: dict = None, erode_kwargs: dict = None) -> None:

        self.dilate_kernel = np.ones(dilate_kernel_size) if dilate_kernel_size is not None else None
        self.erode_kernel = np.ones(erode_kernel_size) if erode_kernel_size is not None else None
        self.dilate_kwargs = dilate_kwargs if dilate_kwargs is not None else {}
        self.erode_kwargs = erode_kwargs if erode_kwargs is not None else {}
        
    def __call__(self, binary_frame):
        if self.erode_kernel is not None:
            binary_frame = cv.erode(binary_frame, self.erode_kernel, **self.erode_kwargs)
        if self.dilate_kernel is not None:
            binary_frame = cv.dilate(binary_frame, self.dilate_kernel, **self.dilate_kwargs)
        
        return binary_frame

    def update(self, dilate_kernel_size, erode_kernel_size,
                 dilate_kwargs: dict, erode_kwargs: dict, **kwargs):

        self.dilate_kernel = np.ones(dilate_kernel_size) if dilate_kernel_size is not None else None
        self.erode_kernel = np.ones(erode_kernel_size) if erode_kernel_size is not None else None
        self.dilate_kwargs = dilate_kwargs if dilate_kwargs is not None else {}
        self.erode_kwargs = erode_kwargs if erode_kwargs is not None else {}


@register("FrameSuppressionDilateErodeMorphology")
class FrameSuppressionDilateErodeMorphology(DilateErodeMorphology):
    def __init__(self, dilate_kernel_size=None, erode_kernel_size=None,
                 thickness=10, dilate_kwargs: dict = None, erode_kwargs: dict = None):
        
        super(FrameSuppressionDilateErodeMorphology, self).__init__(dilate_kernel_size, erode_kernel_size,
                                                          dilate_kwargs, erode_kwargs)
        self.thickness = thickness

    def replace_frame_with_zeros(self, binary_frame):
        frame_thickness = int(self.thickness / 100 * max(binary_frame.shape))
        binary_frame[:frame_thickness, :] = 0  # Top frame
        binary_frame[-frame_thickness:, :] = 0  # Bottom frame
        binary_frame[:, :frame_thickness] = 0  # Left frame
        binary_frame[:, -frame_thickness:] = 0  # Right frame
        return binary_frame
    
    @global_output_recorder.record_output
    def __call__(self, gray):
        gray = self.replace_frame_with_zeros(gray)
        return super(FrameSuppressionDilateErodeMorphology, self).__call__(gray)

    def update(self, dilate_kernel_size, erode_kernel_size,
                 thickness, dilate_kwargs: dict, erode_kwargs: dict, **kwargs):
        super(FrameSuppressionDilateErodeMorphology, self).update(dilate_kernel_size,
                                                                 erode_kernel_size, dilate_kwargs, erode_kwargs,**kwargs)
        self.thickness = thickness