import cv2 as cv
from vmd import VMD
import cv2
import pandas as pd
from SoiUtils.video_manipulations import draw_video_from_bool_csv
from SoiUtils.load import create_video_capture
from VMD.vmd import VMD


def main(vmd_obj, video_cap, save_detections_file=None, rendered_video_file_path=None, frame_limit=100):
    records = []
    while True:
        frame_num = video_cap.get(cv.CAP_PROP_POS_FRAMES)
        success, frame = video_cap.read()
        if not success or frame_num >= frame_limit:
            break
        frame_bboxes = vmd_obj(frame)
        frame_bboxes = frame_bboxes.assign(frame_num=frame_num)
        records.append(frame_bboxes)

    video_bboxes_df = pd.concat(records).astype('int32')

    if save_detections_file is not None:
        video_bboxes_df.to_csv(save_detections_file)

    if rendered_video_file_path is not None:
        bbox_col_names = vmd_obj.bbox_creator_obj.bbox_col_names
        bbox_foramt = vmd_obj.bbox_creator_obj.bbox_format
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        draw_video_from_bool_csv(video_cap, video_bboxes_df, bbox_cols_names=bbox_col_names,
                                 output_video_path=rendered_video_file_path, bbox_foramt=bbox_foramt,
                                 frame_limit=frame_limit)


if __name__ == '__main__':
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--config_path', type=str, default=Path('configs/default.yaml'))
    parser.add_argument('--bbox_save_path', type=str, default=Path('outputs/bboxes/result.csv'))
    parser.add_argument('--rendered_video_save_path', type=str, default=Path('outputs/videos/result.mp4'))
    parser.add_argument('--frame_limit', type=int, default=500)

    args = parser.parse_args()

    video_cap = create_video_capture(args.video_path)

    vmd = VMD(args.config_path)
    main(vmd, video_cap, args.bbox_save_path, args.rendered_video_save_path, args.frame_limit)
