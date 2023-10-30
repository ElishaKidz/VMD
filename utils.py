import yaml
import cv2 as cv
import numpy as np
from numpy.lib.stride_tricks import as_strided

def load_yaml(path):
    with open(path, "r") as stream:
        try:
            content = yaml.load(stream, Loader=yaml.FullLoader)
            return content
        except yaml.YAMLError as exc:
            print(exc)


def create_video_writer_from_capture(video_capture, output_video_path):
    frame_rate = video_capture.get(cv.CAP_PROP_FPS)
    width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width)
    new_height = int(height)
    size = (new_width, new_height)

    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    video_writer = cv.VideoWriter(str(output_video_path), fourcc, frame_rate, size)
    return video_writer


def calc_corners_of_rect(x, y, width, height):
    p1 = (i - width // 2, j - height // 2)
    p2 = (i + width // 2, j + height // 2)
    return p1, p2


def draw_rect(frame, i, j, width, height):
    frame = frame.copy()
    p1, p2 = calc_corners_of_rect(i, j, width, height)
    img = cv.rectangle(frame, p1, p2, color=(0, 255, 0), thickness=2)
    return img


def draw_video_from_bool_csv(video, df, output_video_path,frame_limit=None):
    
    writer = create_video_writer_from_capture(video, output_video_path)
    limit_flag = False
    while True:
        frame_num = video.get(cv.CAP_PROP_POS_FRAMES)

        current_df = df[df["frame_num"] == frame_num][["x", "y", "width", "height"]]

        ret, frame = video.read()
        if frame_limit is not None:
            limit_flag = frame_num>frame_limit
            
        if not ret or limit_flag:
            break

        for bbox in current_df.values:
            x, y, width, height = bbox
            frame = cv.rectangle(frame, (x,y), (x+width,y+height), color=(0, 255, 0), thickness=2)

        writer.write(frame)

    video.release()
    writer.release()


def create_video_capture(input_video_path):
    cap = cv.VideoCapture(input_video_path)
    assert cap.isOpened(), "Could not open video file"
    return cap


def sliding_window(arr, window_size):
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            raise ValueError("need 2-D input")
        if not (window_size > 0):
            raise ValueError("need a positive window size")
        shape = (arr.shape[0] - window_size + 1,
                arr.shape[1] - window_size + 1,
                window_size, window_size)
        if shape[0] <= 0:
            shape = (1, shape[1], arr.shape[0], shape[3])
        if shape[1] <= 0:
            shape = (shape[0], 1, shape[2], arr.shape[1])
        strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
                arr.shape[1]*arr.itemsize, arr.itemsize)
        return as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()

