import cv2 as cv
from vmd import VMD
import os
import cv2
import pandas as pd
from utils import load_yaml,create_video_capture,draw_video_from_bool_csv 
from binarize import binarizers
from detections import detectors
from stabilization import stabilizers
from foreground import foreground_estimators
from vmd import VMD

def main(vmd_obj,video_cap,save_detections_file = None,rendered_video_file_path=None,frame_limit=100):
    records = [] 
    while True:
        frame_num = video_cap.get(cv.CAP_PROP_POS_FRAMES)
        success, frame = video_cap.read()
        if not success or frame_num>= frame_limit:
            break
        frame_bboxes = vmd_obj(frame)
        frame_bboxes = frame_bboxes.assign(frame_num=frame_num)
        records.append(frame_bboxes)
    
    video_bboxes_df = pd.concat(records).astype('int32')
    
    if save_detections_file is not None:
        video_bboxes_df.to_csv(save_detections_file)
    
    if rendered_video_file_path is not None:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        draw_video_from_bool_csv(video_cap,video_bboxes_df,rendered_video_file_path,frame_limit)
            

if __name__ == '__main__':
    from pathlib import Path
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--video_path',type=str)
    parser.add_argument('--config_path',type=str,default=Path('configs/default.yaml'))
    parser.add_argument('--bbox_save_path',type=str,default=Path('outputs/bboxes/result.csv'))
    parser.add_argument('--rendered_video_save_path',type=str,default=Path('outputs/videos/result.mp4'))
    parser.add_argument('--frame_limit',type=int,default=500)
    parser.add_argument('--rgb',action='store_true')

    args = parser.parse_args()
    
    vmd_params = load_yaml(args.config_path)    
    video_cap = create_video_capture(args.video_path)
    
    stabilizer = stabilizers[vmd_params['stabilizer']['stabilizer_name']](**vmd_params['stabilizer'].get('stabilizer_params',{}))
    binarizer = binarizers[vmd_params['binarizer']['binarizer_name']](**vmd_params['binarizer'].get('binarizer_params',{}))
    detector = detectors[vmd_params['detector']['detector_name']](**vmd_params['detector'].get('detector_params',{}))
    foreground_estimator = foreground_estimators[vmd_params['foreground_estimator']['foreground_estimator_name']](**vmd_params['foreground_estimator'].get('foreground_estimator_params',{}))
    vmd = VMD(stabilizer,foreground_estimator,binarizer,detector,args.rgb)
    main(vmd,video_cap,args.bbox_save_path,args.rendered_video_save_path,args.frame_limit)


    
    # vmd_model = VMD()
    # example_video_path = Path('data/DJI_20230316103415_0012_W.MP4')
    # output_video_path = Path('outputs/result.mp4')
    # df = vmd_model.get_rects_df(str(example_video_path))
    # draw_video_from_bool_csv(str(example_video_path), df, str(output_video_path))
