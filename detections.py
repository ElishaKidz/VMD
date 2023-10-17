import cv2 as cv
import pandas as pd
detectors = {}

def register(cls):
    detectors[cls.__name__] = cls

@register
class DetectionsFromContours:
    
    def find_detections_from_contours(self, contours):
        bbox_columns = ["x","y","width","height"]
        records = []
        for contour in contours:
            bbox = cv.boundingRect(contour) # x,y,w,h
            records.append(dict(zip(bbox_columns,bbox)))

        return pd.DataFrame.from_records(records)


    def __call__(self, binary_map):

        contours, _= cv.findContours(image=binary_map,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_SIMPLE)
        return self.find_detections_from_contours(contours)