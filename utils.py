import yaml
import cv2 as cv
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
