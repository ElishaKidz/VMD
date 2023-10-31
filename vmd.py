import numpy as np
import pandas as pd
import cv2 as cv
import logging
from VMD.binarize import binarizers
from VMD.detections import detectors
from VMD.stabilization import stabilizers
from VMD.foreground import foreground_estimators
from SoiUtils.parallelization import ParallelizedPipeline
from SoiUtils.load import load_yaml

class VMD:
    def __init__(self, yaml_path,run_parallelly) -> None:
        self.run_parallelly =run_parallelly

        vmd_params = load_yaml(yaml_path)
        self.video_stabilization_obj = stabilizers[vmd_params['stabilizer']['stabilizer_name']](**vmd_params['stabilizer'].get('stabilizer_params',{}))
        self.binary_frame_creator_obj = binarizers[vmd_params['binarizer']['binarizer_name']](**vmd_params['binarizer'].get('binarizer_params',{}))
        self.bbox_creator_obj = detectors[vmd_params['detector']['detector_name']](**vmd_params['detector'].get('detector_params',{}))
        self.foreground_estimation_obj = foreground_estimators[vmd_params['foreground_estimator']['foreground_estimator_name']](**vmd_params['foreground_estimator'].get('foreground_estimator_params',{}))
        
        vmd_pipeline = [self.video_stabilization_obj,self.foreground_estimation_obj,self.binary_frame_creator_obj,self.bbox_creator_obj]
        if self.run_parallelly:
            self.vmd_pipeline = ParallelizedPipeline(vmd_pipeline)
        else:
            self.vmd_pipeline = vmd_pipeline
    

    def __call__(self, frame):
        # the cv2 caption reads all frames defaultly as bgr therefore they are converted to gray.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.run_parallelly:
            self.vmd_pipeline.write(frame)
        else:
            for vmd_module in self.vmd_pipeline:
                frame = vmd_module(frame)

            return frame
