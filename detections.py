import cv2 as cv
import pandas as pd
import pybboxes as pbx
detectors = {}


def register(cls):
    detectors[cls.__name__] = cls


@register
class DetectionsFromContours:
    def __init__(self, bbox_format='coco', bbox_col_names=None):
        self.bbox_format = bbox_format
        self.bbox_col_names = bbox_col_names if bbox_col_names is not None else ['x','y','width','height']

    def find_detections_from_contours(self, contours):
        records = []
        for contour in contours:
            bbox = cv.boundingRect(contour)  # x,y,w,h
            # convert the bounding box to the requested format by the user
            bbox = pbx.convert_bbox(bbox, from_type='coco', to_type=self.bbox_format)
            records.append(dict(zip(self.bbox_col_names, bbox)))
        # return a dataframe of all bounding boxes in the given frame with the requested format.
        return pd.DataFrame.from_records(records)

    def __call__(self, binary_map):
        contours, _ = cv.findContours(image=binary_map, mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_SIMPLE)
        return self.find_detections_from_contours(contours)
