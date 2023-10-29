import numpy as np
import pandas as pd
import cv2 as cv
import logging
from VMD.binarize import binarizers
from VMD.detections import detectors
from VMD.stabilization import stabilizers
from VMD.foreground import foreground_estimators
from SoiUtils.load import load_yaml

class VMD:
    def __init__(self, yaml_path) -> None:
        logging.basicConfig(level=logging.DEBUG)

        vmd_params = load_yaml(yaml_path)

        self.video_stabilization_obj = stabilizers[vmd_params['stabilizer']['stabilizer_name']](**vmd_params['stabilizer'].get('stabilizer_params',{}))
        self.binary_frame_creator_obj = binarizers[vmd_params['binarizer']['binarizer_name']](**vmd_params['binarizer'].get('binarizer_params',{}))
        self.bbox_creator_obj = detectors[vmd_params['detector']['detector_name']](**vmd_params['detector'].get('detector_params',{}))
        self.foreground_estimation_obj = foreground_estimators[vmd_params['foreground_estimator']['foreground_estimator_name']](**vmd_params['foreground_estimator'].get('foreground_estimator_params',{}))
        self.frame_counter = 0

    def __call__(self, frame):
        # the cv2 caption reads all frames defaultly as bgr therefore they are converted to gray.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        stabilized_frame = self.video_stabilization_obj(frame)
        foreground_estimation = self.foreground_estimation_obj(stabilized_frame)
        binary_foreground_estimation = self.binary_frame_creator_obj(foreground_estimation)
        frame_bboxes = self.bbox_creator_obj(binary_foreground_estimation)
        logging.debug(f'frame number {self.frame_counter}')
        self.frame_counter += 1
        return frame_bboxes
