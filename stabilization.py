import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from vidstab.VidStab import VidStab
from fastMCD_master.KLTWrapper import KLTWrapper
import cv2

stabilizers = {}


def register(cls):
    stabilizers[cls.__name__] = cls


# @register
# class OpticalFlowStabilization:
#     def __init__(self,key_point_kwargs:dict = None,optical_flow_kwargs:dict = None) -> None:
#         self.key_point_kwargs = key_point_kwargs if key_point_kwargs is not None else {} # max_corners: int = 300, quality_level: int =0.01, min_distance: int = 30, block_size: int=3
#         self.optical_flow_kwargs = optical_flow_kwargs if optical_flow_kwargs is not None else {}
#         self.previous_singel_channel_frame = None


#     def __call__(self, singel_channel_frame):

#         rotation_mat = self.get_rotation_mat(frame=singel_channel_frame)
#         if rotation_mat is not None:
#             rotated_frame = self.rotate_frame(singel_channel_frame,rotation_mat)

#         else:
#             rotated_frame = singel_channel_frame

#         return rotated_frame

#     def get_rotation_mat(self,frame):
#         # Assumes that the frame consists onlu of 1 channel i.e (w,h)

#         if self.previous_singel_channel_frame is None:
#             self.previous_singel_channel_frame = frame
#             return None


#         frame_key_points = cv.goodFeaturesToTrack(frame, **self.key_point_kwargs)
#         frame_key_points_previous_position,status,err = cv.calcOpticalFlowPyrLK(frame,self.previous_singel_channel_frame,frame_key_points, **self.optical_flow_kwargs)
#         assert frame_key_points_previous_position.shape ==frame_key_points.shape

#         # find the points where opticalflow sucsseded 
#         success_indices = np.where(status==1)[0]
#         frame_key_points, frame_key_points_previous_position = frame_key_points[success_indices],frame_key_points_previous_position[success_indices]
#         # Estimate the transition matrix from current frame to the previous one
#         rot_mat, mask = cv.findHomography(frame_key_points, frame_key_points_previous_position, cv.RANSAC,5.0)
#         # Update the previous gray frame to the next call.
#         self.previous_singel_channel_frame = frame
#         return rot_mat


#     @staticmethod
#     def rotate_frame(frame,rot_mat):
#         rotated_frame = cv.warpPerspective(frame, rot_mat, (frame.shape[1],frame.shape[0]))
#         return rotated_frame

class OpticalFlowStabilization:
    def __init__(self, rotation_matrix_buffer_size=10, key_point_kwargs: dict = None,
                 optical_flow_kwargs: dict = None) -> None:
        self.key_point_kwargs = key_point_kwargs if key_point_kwargs is not None else {}  # max_corners: int = 300, quality_level: int =0.01, min_distance: int = 30, block_size: int=3
        self.optical_flow_kwargs = optical_flow_kwargs if optical_flow_kwargs is not None else {}
        self.previous_single_channel_frame = None
        self.rotation_matrix_buffer = []
        self.rotation_matrix_buffer_size = rotation_matrix_buffer_size

    def __call__(self, single_channel_frame):

        rotation_mat = self.get_rotation_mat(frame=single_channel_frame)

        # Smooth the rotation matrix using a temporal filter.
        if rotation_mat is not None:
            self.rotation_matrix_buffer.append(rotation_mat)
            if len(self.rotation_matrix_buffer) > self.rotation_matrix_buffer_size:
                self.rotation_matrix_buffer.pop(0)
            rotation_mat = np.mean(self.rotation_matrix_buffer, axis=0)

        # Rotate the frame.
        if rotation_mat is not None:
            rotated_frame = self.rotate_frame(single_channel_frame, rotation_mat)
        else:
            rotated_frame = single_channel_frame

        return rotated_frame

    def get_rotation_mat(self, frame):
        # Assumes that the frame consists onlu of 1 channel i.e (w,h)

        if self.previous_single_channel_frame is None:
            self.previous_single_channel_frame = frame
            return None

        frame_key_points = cv.goodFeaturesToTrack(frame, **self.key_point_kwargs)
        frame_key_points_previous_position, status, err = cv.calcOpticalFlowPyrLK(frame,
                                                                                  self.previous_single_channel_frame,
                                                                                  frame_key_points,
                                                                                  **self.optical_flow_kwargs)
        assert frame_key_points_previous_position.shape == frame_key_points.shape

        # Find the points where optical flow succeeded.
        success_indices = np.where(status == 1)[0]
        frame_key_points, frame_key_points_previous_position = frame_key_points[success_indices], \
                                                               frame_key_points_previous_position[success_indices]

        # Estimate the homography.
        rot_mat, mask = cv.findHomography(frame_key_points, frame_key_points_previous_position, cv.RANSAC, 10.0)

        # Update the previous gray frame to the next call.
        self.previous_single_channel_frame = frame

        return rot_mat

    @staticmethod
    def rotate_frame(frame, rot_mat):
        rotated_frame = cv.warpPerspective(frame, rot_mat, (frame.shape[1], frame.shape[0]))
        return rotated_frame


class KLTStabilization(OpticalFlowStabilization):
    def __init__(self, rotation_matrix_buffer_size=10, key_point_kwargs: dict = None,
                 optical_flow_kwargs: dict = None) -> None:
        super(KLTStabilization, self).__init__(rotation_matrix_buffer_size, key_point_kwargs, optical_flow_kwargs)
        self.klt = KLTWrapper()

    def get_rotation_mat(self, frame):
        if self.previous_single_channel_frame is None:
            self.previous_single_channel_frame = frame
            self.klt.init(frame)
            return None
        frame = cv2.medianBlur(frame, 5)
        self.klt.RunTrack(frame, self.previous_single_channel_frame)
        self.previous_single_channel_frame = frame
        return self.klt.H


@register
class VidStabStabilization:
    def __init__(self, kp_method="GFTT", smoothing_window=1, grow_window=False) -> None:
        self.stabilizer = VidStab(kp_method=kp_method)
        self.smoothing_window = smoothing_window
        self.frame_counter = 0
        self.grow_window = grow_window
        self.stabled_image = False

    def __call__(self, single_channel_frame):
        self.frame_counter += 1

        if self.stabled_image and self.grow_window:
            self.smoothing_window += 1
        stabilized_frame = self.stabilizer.stabilize_frame(input_frame=single_channel_frame,
                                                           smoothing_window=self.smoothing_window)

        if stabilized_frame is None:
            stabilized_frame = single_channel_frame
            self.stabled_image = False
        else:
            self.stabled_image = True

        return stabilized_frame


class NoStability:
    def __init__(self):
        pass

    def __call__(self, single_channel_frame):
        return single_channel_frame
