import cv2 as cv
from vmd import VMD
import cv2
import pandas as pd
from SoiUtils.video_manipulations import draw_video_from_bool_csv
from SoiUtils.load import create_video_capture
from VMD.vmd import VMD
import asyncio
import logging
finished_reading_video=False
number_of_frames_read = 0
async def execute_vmd(video_cap,vmd_obj,frame_limit):
    global finished_reading_video
    global number_of_frames_read
    while True:
        frame_num = video_cap.get(cv.CAP_PROP_POS_FRAMES)
        success, frame = video_cap.read()
        if not success or frame_num >= frame_limit:
            finished_reading_video =True
            break
        number_of_frames_read +=1
        vmd_obj(frame)

async def read_vmd_results(vmd_obj):
    global finished_reading_video
    global number_of_frames_read
    number_of_frames_finished_processing = 0
    records = []
    while (not finished_reading_video) or (number_of_frames_finished_processing<number_of_frames_read):
        frame_bboxes = vmd_obj.vmd_pipeline.read()
        frame_bboxes = frame_bboxes.assign(frame_num=number_of_frames_finished_processing)
        records.append(frame_bboxes)
        logging.debug(f"finished processing frame number {number_of_frames_finished_processing}")
        number_of_frames_finished_processing+=1
    vmd.vmd_pipeline.stop_all()
    return records





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
        logging.debug(f"finished processing frame number {frame_num}")


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


async def parallel_main(vmd_obj, video_cap, save_detections_file=None, rendered_video_file_path=None, frame_limit=100):
    writing_task = asyncio.create_task(execute_vmd(video_cap,vmd_obj,frame_limit))
    reading_task = asyncio.create_task(read_vmd_results(vmd_obj))    
    await writing_task
    records = await reading_task
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
    import time
    logging.basicConfig(level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--config_path', type=str, default=Path('configs/default.yaml'))
    parser.add_argument('--bbox_save_path', type=str, default=Path('outputs/bboxes/result.csv'))
    parser.add_argument('--rendered_video_save_path', type=str, default=Path('outputs/videos/result.mp4'))
    parser.add_argument('--frame_limit', type=int, default=500)
    parser.add_argument('--run_parallelly', action='store_true')
    args = parser.parse_args()

    video_cap = create_video_capture(args.video_path)

    vmd = VMD(args.config_path,args.run_parallelly)
    if not args.run_parallelly:
        main(vmd, video_cap, args.bbox_save_path, args.rendered_video_save_path, args.frame_limit)

    else:
        asyncio.run(parallel_main(vmd, video_cap, args.bbox_save_path, args.rendered_video_save_path, args.frame_limit))