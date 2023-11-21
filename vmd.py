import numpy as np
import pandas as pd
import cv2 as cv
import logging
from VMD.binarize import binarizers
from VMD.detections import detectors
from VMD.stabilization import stabilizers
from VMD.foreground import foreground_estimators
from SoiUtils.load import load_yaml
import time


class VMD:
    RESET_FN_NAME = 'reset'
    def __init__(self, yaml_path,correct_detections_to_original_frame=True) -> None:
        logging.basicConfig(level=logging.DEBUG)

        vmd_params = load_yaml(yaml_path)

        self.video_stabilization_obj = stabilizers[vmd_params['stabilizer']['stabilizer_name']](**vmd_params['stabilizer'].get('stabilizer_params',{}))
        self.binary_frame_creator_obj = binarizers[vmd_params['binarizer']['binarizer_name']](**vmd_params['binarizer'].get('binarizer_params',{}))
        self.bbox_creator_obj = detectors[vmd_params['detector']['detector_name']](**vmd_params['detector'].get('detector_params',{}))
        self.foreground_estimation_obj = foreground_estimators[vmd_params['foreground_estimator']['foreground_estimator_name']](**vmd_params['foreground_estimator'].get('foreground_estimator_params',{}))
        self.frame_counter = 0
        self.time = 0
        self.correct_detections_to_original_frame = correct_detections_to_original_frame

    def __call__(self, frame):
        # the cv2 caption reads all frames defaultly as bgr therefore they are converted to gray.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        rotation_mat = self.video_stabilization_obj(frame)
        if rotation_mat is not None:
            stabilized_frame = self.rotate_frame(frame,rotation_mat)

        else:
            stabilized_frame = frame
        
        foreground_estimation = self.foreground_estimation_obj(stabilized_frame)
        
        if self.correct_detections_to_original_frame and rotation_mat is not None:
            foreground_estimation = self.rotate_frame(foreground_estimation,rotation_mat,inverse=True)
        
        else:
            pass
        
        binary_foreground_estimation = self.binary_frame_creator_obj(foreground_estimation)
        frame_bboxes = self.bbox_creator_obj(binary_foreground_estimation)
        logging.debug(f'frame number {self.frame_counter}')
        self.frame_counter += 1
        return frame_bboxes
    
    def reset(self):
        self.time = 0
        if hasattr(self.video_stabilization_obj, VMD.RESET_FN_NAME):
            self.video_stabilization_obj.reset()

        if hasattr(self.binary_frame_creator_obj, VMD.RESET_FN_NAME):
            self.binary_frame_creator_obj.reset()
        
        if hasattr(self.bbox_creator_obj, VMD.RESET_FN_NAME):
            self.bbox_creator_obj.reset()
        
        if hasattr(self.foreground_estimation_obj, VMD.RESET_FN_NAME):
            self.foreground_estimation_obj.reset()


    @staticmethod
    def rotate_frame(frame, rot_mat,inverse=False):
        if not inverse:
            rotated_frame = cv.warpPerspective(frame, rot_mat, (frame.shape[1], frame.shape[0]))

        else:
            rotated_frame = cv.warpPerspective(frame, rot_mat, (frame.shape[1], frame.shape[0]),cv.WARP_INVERSE_MAP)

        return rotated_frame
    