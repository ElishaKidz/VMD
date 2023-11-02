import numpy as np
import cv2
from VMD.fastMCD_master import KLTWrapper
from VMD.fastMCD_master import ProbModel


class MCDWrapper:
    def __init__(self):
        self.imgIpl = None
        self.imgGray = None
        self.imgGrayPrev = None
        self.is_first = True
        self.frm_cnt = 0
        self.lucasKanade = KLTWrapper.KLTWrapper()
        self.model = ProbModel.ProbModel()

    def init(self, image):
        self.imgGray = image
        self.imgGrayPrev = image
        self.lucasKanade.init(self.imgGray)
        self.model.init(self.imgGray)
        self.is_first = False

    def run(self, frame):
        if self.is_first:
            self.init(frame)
            return np.zeros_like(frame)
        self.frm_cnt += 1
        self.imgIpl = frame
        self.imgGray = frame
        # self.imgGray = cv2.medianBlur(self.imgGray, 5)  # TODO: delete
        self.imgGray = cv2.GaussianBlur(self.imgGray, (7, 7), 0)
        if self.imgGrayPrev is None:
            self.imgGrayPrev = self.imgGray.copy()
        
        self.lucasKanade.RunTrack(self.imgGray, self.imgGrayPrev)
        self.model.motionCompensate(self.lucasKanade.H)
        mask = self.model.update(frame)
        self.imgGrayPrev = self.imgGray.copy()
        return mask

    def __call__(self, frame):
        if self.is_first:
            self.init(frame)
            return np.zeros_like(frame)
        self.frm_cnt += 1
        self.imgIpl = frame
        self.imgGray = frame
        # self.imgGray = cv2.medianBlur(self.imgGray, 5)  # TODO: delete
        self.imgGray = cv2.GaussianBlur(self.imgGray, (7, 7), 0)
        if self.imgGrayPrev is None:
            self.imgGrayPrev = self.imgGray.copy()

        self.lucasKanade.RunTrack(self.imgGray, self.imgGrayPrev)
        self.model.motionCompensate(self.lucasKanade.H)
        mask = self.model.update(frame)
        self.imgGrayPrev = self.imgGray.copy()
        return mask
