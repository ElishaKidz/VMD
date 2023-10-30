import os

import numpy as np
import cv2
import MCDWrapper
from tqdm import tqdm

np.set_printoptions(precision=2, suppress=True)
cap = cv2.VideoCapture(r'C:\Users\orber\PycharmProjects\VMD\tel_move.mp4')
# cap = cv2.VideoCapture(r'C:\Users\orber\PycharmProjects\VMD\fastMCD-master\data\woman.mp4')
mcd = MCDWrapper.MCDWrapper()
isFirst = True
first_frame = None
os.makedirs("fastMCD_results", exist_ok=True)

for i in tqdm(range(200)):
    ret, frame = cap.read()
    if not ret:
        cap.release()
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if (isFirst):
        mcd.init(gray)
        isFirst = False
        first_frame = frame
        out = cv2.VideoWriter('fastMCD_results/tel_move.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, gray.shape[::-1])
    else:
        mask = mcd.run(gray)
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    bbox_columns = ["x", "y", "width", "height"]
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w*3, y + h*3), (0, 255, 0), 2)

        # rects.append(dict(zip(bbox_columns, bbox)))
        pass
    out.write(frame)

    # frame[mask > 0, 2] = 255
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
out.release()

