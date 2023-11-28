import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
binarizers = {}


def register(name):
    def register_func_fn(cls):
        binarizers[name] = cls
        return cls
    return register_func_fn


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)


@register("DilateErodeBinarizer")
class DilateErodeBinarizer:
    def __init__(self, diff_frame_threshold: int = 30, dilate_kernel_size=(15, 15), erode_kernel_size=(2, 2),
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

    def update(self, diff_frame_threshold: int, dilate_kernel_size, erode_kernel_size,
                 dilate_kwargs: dict, erode_kwargs: dict):
        self.diff_frame_threshold = diff_frame_threshold
        self.dilate_kernel = np.ones(dilate_kernel_size)
        self.erode_kernel = np.ones(erode_kernel_size)
        self.dilate_kwargs = dilate_kwargs if dilate_kwargs is not None else {}
        self.erode_kwargs = erode_kwargs if erode_kwargs is not None else {}


@register("FrameSuppressionDilateErodeBinarizer")
class FrameSuppressionDilateErodeBinarizer(DilateErodeBinarizer):
    def __init__(self, diff_frame_threshold: int = 30, dilate_kernel_size=(15, 15), erode_kernel_size=(2, 2),
                 thickness=10, dilate_kwargs: dict = None, erode_kwargs: dict = None):
        super(FrameSuppressionDilateErodeBinarizer, self).__init__(diff_frame_threshold, dilate_kernel_size, erode_kernel_size,
                                                          dilate_kwargs, erode_kwargs)
        self.thickness = thickness

    def replace_frame_with_zeros(self, image):
        frame_thickness = int(self.thickness / 100 * max(image.shape))
        image[:frame_thickness, :] = 0  # Top frame
        image[-frame_thickness:, :] = 0  # Bottom frame
        image[:, :frame_thickness] = 0  # Left frame
        image[:, -frame_thickness:] = 0  # Right frame
        return image

    def __call__(self, gray):
        gray = self.replace_frame_with_zeros(gray)
        return super(FrameSuppressionDilateErodeBinarizer, self).__call__(gray)

    def update(self, diff_frame_threshold: int, dilate_kernel_size, erode_kernel_size,
                 thickness, dilate_kwargs: dict, erode_kwargs: dict):
        super(FrameSuppressionDilateErodeBinarizer, self).update(diff_frame_threshold, dilate_kernel_size,
                                                                 erode_kernel_size, dilate_kwargs, erode_kwargs)
        self.thickness = thickness


@register("DilateErodeDynamicBinarizer")
class DilateErodeDynamicBinarizer(DilateErodeBinarizer):
    def __init__(self, diff_frame_threshold: int = 150, dilate_kernel_size=(15, 15), erode_kernel_size=(2, 2),
                 dilate_kwargs: dict = None, erode_kwargs: dict = None):
        super(DilateErodeDynamicBinarizer, self).__init__(diff_frame_threshold, dilate_kernel_size, erode_kernel_size,
                                                          dilate_kwargs, erode_kwargs)

    def __call__(self, gray_frame):
        flat_array = gray_frame.flatten()
        mn, std = np.mean(flat_array), np.std(flat_array)
        statistic_thresh = mn + 5*std
        self.diff_frame_threshold = int(0.5 * statistic_thresh + 0.5 * self.diff_frame_threshold)
        thresh_frame = \
            cv.threshold(src=gray_frame, thresh=self.diff_frame_threshold, maxval=255, type=cv.THRESH_BINARY)[1]
        thresh_frame = cv.erode(thresh_frame, self.erode_kernel, **self.erode_kwargs)
        thresh_frame = cv.dilate(thresh_frame, self.dilate_kernel, **self.dilate_kwargs)
        return thresh_frame


@register("NormalizedDilateErodeBinarizer")
class NormalizedDilateErodeBinarizer(DilateErodeBinarizer):
    def __init__(self, diff_frame_threshold: int = 150, dilate_kernel_size=(15, 15), erode_kernel_size=(2, 2),
                 dilate_kwargs: dict = None, erode_kwargs: dict = None):
        super(NormalizedDilateErodeBinarizer, self).__init__(diff_frame_threshold, dilate_kernel_size, erode_kernel_size,
                                                          dilate_kwargs, erode_kwargs)

    def __call__(self, gray_frame):
        if not np.any(gray_frame):
            return gray_frame
        foreground = gray_frame.astype(np.float32)
        foreground = (foreground - np.min(foreground)) / (np.max(foreground) - np.min(foreground)) * 255
        foreground = gammaCorrection(foreground.astype(np.uint8), 2.2)
        return super(NormalizedDilateErodeBinarizer, self).__call__(foreground.astype(np.uint8))
