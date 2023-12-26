import numpy as np
import pandas as pd
import cv2 as cv
import logging
from VMD.binarize import binarizers
from VMD.detections import detectors
from VMD.stabilization import stabilizers
from VMD.foreground import foreground_estimators
from SoiUtils.load import load_yaml
from SoiUtils.interfaces import Resetable, Updatable, Localizer
import time


class VMD(Resetable,Updatable,Localizer):

    def __init__(self, stabilizer, binarizer, detector, foreground_estimator) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.video_stabilization_obj = stabilizers[stabilizer['stabilizer_name']](
            **stabilizer.get('stabilizer_params', {}))
        self.binary_frame_creator_obj = binarizers[binarizer['binarizer_name']](
            **binarizer.get('binarizer_params', {}))
        self.bbox_creator_obj = detectors[detector['detector_name']](
            **detector.get('detector_params', {}))
        self.foreground_estimation_obj = foreground_estimators[
            foreground_estimator['foreground_estimator_name']](
            **foreground_estimator.get('foreground_estimator_params', {}))
        self.frame_counter = 0
        self.time = 0

    @classmethod
    def from_yaml(cls, yaml_config_path):
        vmd_params = load_yaml(yaml_config_path)
        return cls(**vmd_params)

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

    def reset(self):
        self.time = 0
        if issubclass(type(self.video_stabilization_obj), Resetable):
            self.video_stabilization_obj.reset()

        if issubclass(type(self.binary_frame_creator_obj), Resetable):
            self.binary_frame_creator_obj.reset()

        if issubclass(type(self.bbox_creator_obj), Resetable):
            self.bbox_creator_obj.reset()

        if issubclass(type(self.foreground_estimation_obj), Resetable):
            self.foreground_estimation_obj.reset()

    def update(self, stabilizer, binarizer, detector, foreground_estimator):
        self.time = 0
        if issubclass(type(self.video_stabilization_obj), Updatable):
            self.video_stabilization_obj.update(**stabilizer.get('stabilizer_params', {}))

        if issubclass(type(self.binary_frame_creator_obj), Updatable):
            self.binary_frame_creator_obj.update(**binarizer.get('binarizer_params', {}))

        if issubclass(type(self.bbox_creator_obj), Updatable):
            self.bbox_creator_obj.update(**detector.get('detector_params', {}))

        if issubclass(type(self.foreground_estimation_obj), Updatable):
            self.foreground_estimation_obj.update(**foreground_estimator.get('foreground_estimator_params', {}))

    
    def localize(self, frame):
        return self(frame)