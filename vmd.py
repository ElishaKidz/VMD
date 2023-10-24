import numpy as np
import pandas as pd
import cv2 as cv
import logging
# def gray_scale_and_blur_frame(frame,ksize: tuple = (5,5), sigmaX: int=0):
#     assert frame is not None, "Empty frame0"
#     prepared_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     prepared_frame = cv.GaussianBlur(src=prepared_frame,ksize=ksize,sigmaX=sigmaX)
#     return prepared_frame

# def crop_margins(frame,width_to_crop: int =6):
#     frame = frame.copy()

#     frame[:width_to_crop,:] = 0
#     frame[-width_to_crop:,:] = 0
#     frame[:,:width_to_crop] = 0
#     frame[:,-width_to_crop] = 0
#     return frame

# def rotate_frame(frame,rot_mat):
#     rotated_frame = cv.warpAffine(frame, rot_mat, (frame.shape[1],frame.shape[0]))
#     return rotated_frame

# def get_rotation_mat(frame_i,frame_f,max_corners: int = 300, quality_level: int =0.01, min_distance: int = 30, block_size: int=3):
#     gray_i = cv.cvtColor(frame_i,cv.COLOR_BGR2GRAY)
#     gray_f = cv.cvtColor(frame_f,cv.COLOR_BGR2GRAY)
#     pts_i = cv.goodFeaturesToTrack(gray_i,maxCorners=max_corners,qualityLevel=quality_level,minDistance=min_distance,blockSize=block_size)
#     ast_pts_f,status,err = cv.calcOpticalFlowPyrLK(gray_i,gray_f,pts_i,None)
#     assert pts_i.shape ==ast_pts_f.shape

#     idx = np.where(status==1)[0]
#     pts_i = pts_i[idx]
#     ast_pts_f = ast_pts_f[idx]
#     rot_mat = cv.estimateAffine2D(pts_i,ast_pts_f)[0]
#     return rot_mat


# def append_to_df(df,current_df,frameNum,frameTime):
#     current_df=current_df.copy()
#     current_df["frame_num"] = frameNum
#     current_df["frame_time"] = frameTime
#     joined_df = pd.concat([df,current_df],axis=0,ignore_index=True)
#     return joined_df


# def create_video_capture(input_video_path):
#     cap = cv.VideoCapture(input_video_path)
#     assert cap.isOpened(), "Could not open video file"
#     return cap

class VMD:
    def __init__(self, video_stabilization_obj, foreground_estimation_obj, binary_frame_creator_obj, bbox_creator_obj):
        
        logging.basicConfig(level=logging.DEBUG)
        self.video_stabilization_obj = video_stabilization_obj
        self.foreground_estimation_obj = foreground_estimation_obj
        self.binary_frame_creator_obj = binary_frame_creator_obj
        self.bbox_creator_obj = bbox_creator_obj
        self.frame_counter = 0

    def __call__(self, frame):
        # the cv2 caption reads all frames defaultly as bgr therefore they are converted to gray.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        stabilized_frame = self.video_stabilization_obj(frame)
        foreground_estimation = self.foreground_estimation_obj(stabilized_frame)
        binary_foreground_estimation = self.binary_frame_creator_obj(foreground_estimation)
        # max_bin_val = foreground_estimation.max()
        frame_bboxes = self.bbox_creator_obj(binary_foreground_estimation)
        logging.debug(f'frame number {self.frame_counter}')
        self.frame_counter += 1
        return frame_bboxes
    


            

    # def calc_diff_frame(self, prev_frame, new_frame, prev_mask,dilate_kernel: tuple= (8,8) ,erode_kernel: tuple = (12,12)):

    #     thresh_frame = self.fgbg.apply(new_frame)
    #     thresh_frame = cv.threshold(src=thresh_frame, thresh=self.diff_frame_threshold,maxval=255, type=cv.THRESH_BINARY)[1]
    #     thresh_frame = cv.dilate(thresh_frame,kernel=dilate_kernel,iterations=1)
    #     thresh_frame = cv.erode(thresh_frame, kernel=erode_kernel, iterations=1)
    #     return thresh_frame
    
    # def find_bools_from_contours(self, contours):
    #     df = pd.DataFrame(columns=["i","j","width","height"])

    #     if len(contours)>self.max_objects:

    #         contours = sorted(contours,key=lambda contour: cv.contourArea(contour))
    #         contours = contours[-self.max_objects:]

    #     for contour in contours:

    #         x,y,w,h = cv.boundingRect(contour)
    #         i = x+w //2
    #         j = y+h //2

    #         temp = pd.DataFrame(columns=["i","j","width","height"], data=[[i,j,w,h]])
    #         df = pd.concat([df,temp], axis=0)

    #     return df

    # def get_bools_between_two_frame(self,prev_frame,new_frame, global_frame):
    #     rot_mat = get_rotation_mat(prev_frame, new_frame)
    #     prev_frame_transformed = rotate_frame(prev_frame, rot_mat)
    #     prev_frame = gray_scale_and_blur_frame(prev_frame_transformed)
    #     new_frame = gray_scale_and_blur_frame(new_frame)
    #     thresh_frame = self.calc_diff_frame(global_frame,new_frame,prev_mask=None)
    #     thresh_frame = crop_margins(thresh_frame)
    #     contours, _ = cv.findContours(image=thresh_frame,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_SIMPLE)

    #     df = self.find_bools_from_contours(contours)
    #     return df, global_frame
    
    # def get_rects_df(self, video_path):

    #     column_names = ["frame_num","frame_time","i","j","width","height"]
    #     df = pd.DataFrame(columns=column_names)
    #     video = create_video_capture(video_path)
    #     ret, new_frame = video.read()
    #     global_frame = gray_scale_and_blur_frame(new_frame)
    #     while True:
    #         frame_num = video.get(cv.CAP_PROP_POS_FRAMES)
    #         frame_time = video.get(cv.CAP_PROP_POS_MSEC)

    #         previous_frame = new_frame
    #         ret, new_frame = video.read()

    #         if not ret:
    #             break
            
    #         assert previous_frame.shape == new_frame.shape, "Problem in loading frame, isn't equal in shape to previous frame"

    #         current_df, global_frame = self.get_bools_between_two_frame(previous_frame, new_frame,global_frame)
    #         df = append_to_df(df, current_df, frame_num, frame_time)
        
    #     return df

