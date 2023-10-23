import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from utils import load_yaml, create_video_capture, draw_video_from_bool_csv
from binarize import binarizers
from detections import detectors
from stabilization import stabilizers
from foreground import foreground_estimators
from vmd import VMD
from pathlib import Path

VIDEO_LEN_IN_SECONDS = 5
BBOXES_PATH = 'outputs/bboxes'
VIDEOS_PATH = 'outputs/videos'


def run(camera_name, vmd_obj, video_cap, frame_limit, save_detections_file=None, rendered_video_file_path=None):
    records = []
    while True:
        read_frame_time = time.perf_counter()
        ret, frame = video_cap.read()
        print(f"read_frame: {time.perf_counter() - read_frame_time}")
        if not ret:  # or frame_num >= frame_limit:
            break

        vmd_time = time.perf_counter()
        frame_bboxes = vmd_obj(frame)
        print(f"vmd: {time.perf_counter() - vmd_time}")

        rect_time = time.perf_counter()
        for bbox in frame_bboxes.values:
            x, y, width, height = bbox
            frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
        print(f"rect: {time.perf_counter() - rect_time}")

        cv2.imshow(camera_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    video_cap.release()

    video_bboxes_df = pd.concat(records).astype('int32')

    if save_detections_file is not None:
        print(save_detections_file)
        print(video_bboxes_df)
        video_bboxes_df.to_csv(save_detections_file)

    if rendered_video_file_path is not None:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        draw_video_from_bool_csv(video_cap, video_bboxes_df, rendered_video_file_path, frame_limit)


def get_camera_name(camera_path: str) -> str:
    # TODO: fetch rtsp camera id/displayName
    return camera_path.split(".")[0]


def main(camera_path: str, config_path=Path('configs/default.yaml')):
    camera_name = get_camera_name(camera_path)
    #camera_name='RF'
    bbox_save_path = Path(f'{BBOXES_PATH}/{camera_name}.csv')
    rendered_video_save_path = Path(f'{VIDEOS_PATH}/{camera_name}.mp4')
    vmd_params = load_yaml(config_path)
    video_cap = create_video_capture(camera_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_limit = fps * VIDEO_LEN_IN_SECONDS

    stabilizer = stabilizers[vmd_params['stabilizer']['stabilizer_name']](
        **vmd_params['stabilizer'].get('stabilizer_params', {}))
    binarizer = binarizers[vmd_params['binarizer']['binarizer_name']](
        **vmd_params['binarizer'].get('binarizer_params', {}))
    detector = detectors[vmd_params['detector']['detector_name']](**vmd_params['detector'].get('detector_params', {}))
    foreground_estimator = foreground_estimators[vmd_params['foreground_estimator']['foreground_estimator_name']](
        **vmd_params['foreground_estimator'].get('foreground_estimator_params', {}))
    vmd = VMD(stabilizer, foreground_estimator, binarizer, detector)
    run(camera_name, vmd, video_cap, frame_limit, bbox_save_path, rendered_video_save_path)


if __name__ == '__main__':
    Path(BBOXES_PATH).mkdir(exist_ok=True, parents=True)
    Path(VIDEOS_PATH).mkdir(exist_ok=True, parents=True)
    camera_urls = ['rtsp%3A%2F%2Fonvif%3AOnvif%24123%402.55.86.220%3A554%2Flive%2F048d2037-a256-415c-b8a0-2c9a3a89255d']
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = pool.map(main, camera_urls)
        try:
            for _, return_value in futures:
                print(return_value)
        except Exception as e:
            print(e)

    # vmd_model = VMD()
    # example_video_path = Path('data/DJI_20230316103415_0012_W.MP4')
    # output_video_path = Path('outputs/result.mp4')
    # df = vmd_model.get_rects_df(str(example_video_path))
    # draw_video_from_bool_csv(str(example_video_path), df, str(output_video_path))
