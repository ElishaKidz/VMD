import cv2 as cv
from vmd import VMD
import pandas as pd
from SoiUtils.video_manipulations import draw_video_from_bool_csv
from SoiUtils.load import create_video_capture
from vmd import VMD
import gcsfs
from pycocotools.coco import COCO
import numpy as np
import pybboxes as pbx

fs = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")

def get_video_dfs(video_dir, vmd_obj, bbox_col_names, bbox_format):
    pred_bboxes = []
    gt_bboxes = []
    fs.get(f"{video_dir}/annotations.json", "annotations.json")
    frames_bboxes_gt = COCO("annotations.json").imgToAnns
    for frame_path in fs.ls(f"{video_dir}/frames"):
            frame_num = int(frame_path.split("/")[-1].split("_")[1].split(".")[0]) + 1
            ###################### for debug only
            if frame_num > args.frame_limit:
                break
            ######################
            frame = np.asarray(bytearray(fs.open(frame_path, "rb").read()), dtype="uint8")
            frame = cv.imdecode(frame, cv.IMREAD_COLOR)
            
            frame_bboxes = vmd_obj(frame)
            frame_bboxes = frame_bboxes.assign(frame_num=frame_num)
            frame_bboxes = frame_bboxes.assign(video_file=video_dir)
            pred_bboxes.append(frame_bboxes)
            
            if frame_num in frames_bboxes_gt:
                for ann in frames_bboxes_gt[frame_num]: # need to check
                    bbox = list(pbx.convert_bbox(ann["bbox"], from_type="coco", to_type=bbox_format, image_size=frame.shape[:2][::-1]))
                    frame_bboxes_gt = {col_name: value for col_name, value in zip(bbox_col_names, bbox)}
                    frame_bboxes_gt['frame_num'] = frame_num
                    frame_bboxes_gt['video_file'] = video_dir
                    gt_bboxes.append(frame_bboxes_gt)

    return pd.concat(pred_bboxes), pd.DataFrame(gt_bboxes)

def get_precision_recall(pred_bboxes, gt_bboxes, iou_threshold=0.2):
    pred_bboxes = [pbx.BoundingBox.from_coco(*bbox) for bbox in pred_bboxes] if len(pred_bboxes) > 0 else []
    gt_bboxes = [pbx.BoundingBox.from_coco(*bbox) for bbox in gt_bboxes] if len(gt_bboxes) > 0 else []

    ious = []
    for pred_bbox in pred_bboxes:
        for gt_bbox in gt_bboxes:
            ious.append(pred_bbox.iou(gt_bbox))
    
    hits = (np.array(ious) > iou_threshold).sum()
    precision = hits / len(pred_bboxes) if len(pred_bboxes) > 0 else 0
    recall = hits / len(gt_bboxes) if len(gt_bboxes) > 0 else 0

    return precision, recall

def eval_video(pred_bboxes_df, gt_bboxes_df, bbox_col_names):
    frames_with_bboxes = set(pred_bboxes_df["frame_num"].values).union(set(gt_bboxes_df["frame_num"].values))
    sum_precision, sum_recall = 0, 0
    for frame_id in frames_with_bboxes:
        frame_pred_bboxes = pred_bboxes_df.loc[pred_bboxes_df["frame_num"] == frame_id][bbox_col_names].values.tolist()
        frame_gt_bboxes = gt_bboxes_df.loc[gt_bboxes_df["frame_num"] == frame_id][bbox_col_names].values.tolist()
        frame_precision, frame_recall = get_precision_recall(frame_pred_bboxes, frame_gt_bboxes)
        
        sum_precision += frame_precision
        sum_recall += frame_recall

    return sum_precision, sum_recall, len(frames_with_bboxes)

def main(vmd_obj, remote_dir, save_detections_dir=None, rendered_videos_dir_path=None):
    bbox_col_names = vmd_obj.bbox_creator_obj.bbox_col_names
    bbox_format = vmd_obj.bbox_creator_obj.bbox_format
    
    pred_bboxes_df = pd.DataFrame({})
    gt_bboxes_df = pd.DataFrame({})
    
    for video_dir in fs.ls(remote_dir)[1:]:
        if video_dir.split("/")[-1] in ignored_videos:
            continue
        video_pred_bboxes, video_gt_bboxes = get_video_dfs(video_dir, vmd_obj, bbox_col_names, bbox_format)
        video_precision, video_recall, number_of_frames_with_bbox = eval_video(video_pred_bboxes, video_gt_bboxes, bbox_col_names)
        print(f"{video_dir} Precision: {video_precision / number_of_frames_with_bbox}")
        print(f"{video_dir} Recall: {video_recall / number_of_frames_with_bbox}")
        
        video_name = [file_name.split("/")[-1] for file_name in fs.ls(video_dir) if ".MP4" in file_name][0]
        if rendered_videos_dir_path is not None:
            fs.get(f"{video_dir}/{video_name}", "video.mp4")
            video_cap = create_video_capture("video.mp4")
            video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            draw_video_from_bool_csv(video_cap, video_gt_bboxes, bbox_cols_names=bbox_col_names,
                                    output_video_path=f"{rendered_videos_dir_path}/GT_{video_name}", bbox_foramt=bbox_format)
            
            video_cap = create_video_capture("video.mp4")
            video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            draw_video_from_bool_csv(video_cap, video_pred_bboxes, bbox_cols_names=bbox_col_names,
                                    output_video_path=f"{rendered_videos_dir_path}/PRED_{video_name}", bbox_foramt=bbox_format)
       
        if save_detections_dir is not None:
            pred_bboxes_df.to_csv(f"{save_detections_dir}/PRED_{video_name}.csv")
            gt_bboxes_df.to_csv(f"{save_detections_dir}/GT_{video_name}.csv")

if __name__ == '__main__':
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--remote_dir', type=str, default="soi_experiments/annotations-example/")
    parser.add_argument('--config_path', type=str, default=Path('configs/default.yaml'))
    parser.add_argument('--bbox_save_dir', type=str, default=Path('outputs/bboxes'))
    parser.add_argument('--rendered_videos_dir', type=str, default=Path('outputs/videos'))
    parser.add_argument('--ignored_videos', type=str, default="")
    parser.add_argument('--frame_limit', type=int, default=500)
    args = parser.parse_args()

    ignored_videos = args.ignored_videos.split(",")

    vmd = VMD(args.config_path)
    main(vmd, args.remote_dir, args.bbox_save_dir, args.rendered_videos_dir)
