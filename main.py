import cv2 as cv
from vmd import VMD
from vmd import create_video_capture
import os
import cv2
import pandas as pd


def create_video_writer_from_capture(video_capture, output_video_path):
    frame_rate = video_capture.get(cv.CAP_PROP_FPS)
    width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width)
    new_height = int(height)
    size = (new_width, new_height)

    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    video_writer = cv.VideoWriter(output_video_path, fourcc, frame_rate, size)
    return video_writer


def calc_corners_of_rect(i, j, width, height):
    p1 = (i - width // 2, j - height // 2)
    p2 = (i + width // 2, j + height // 2)
    return p1, p2


def draw_rect(frame, i, j, width, height):
    frame = frame.copy()
    p1, p2 = calc_corners_of_rect(i, j, width, height)
    img = cv.rectangle(frame, p1, p2, color=(0, 255, 0), thickness=2)
    return img


def draw_video_from_bool_csv(video_path, df, output_video_path):
    video = create_video_capture(video_path)
    writer = create_video_writer_from_capture(video, output_video_path)
    print("Created video capture and writer successfully")

    while True:
        frame_num = video.get(cv.CAP_PROP_POS_FRAMES)
        current_df = df[df["frame_num"] == frame_num][["i", "j", "width", "height"]]

        ret, frame = video.read()
        if not ret:
            break

        for line in current_df.values:
            i, j, width, height = line
            frame = draw_rect(frame, i, j, width, height)

        writer.write(frame)

    video.release()
    writer.release()


if __name__ == '__main__':
    from pathlib import Path

    vmd_model = VMD()
    example_video_path = Path('data/DJI_20230316103415_0012_W.MP4')
    output_video_path = Path('outputs/result.mp4')
    df = vmd_model.get_rects_df(str(example_video_path))
    draw_video_from_bool_csv(str(example_video_path), df, str(output_video_path))
