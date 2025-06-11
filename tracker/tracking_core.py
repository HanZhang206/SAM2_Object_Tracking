from models.sam2 import SAM2
from config import WINDOW_SIZE
import json
import os
import numpy as np
import hashlib
from tqdm import tqdm
import cv2
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    """
    Compute IOU between two boxes.
    Each box is a numpy array of shape (2, 2): [[x1, y1], [x2, y2]]
    """
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1])
    boxBArea = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
def hungarian_match(detected_boxes, tracked_boxes, iou_threshold=0.3):
    """
    匈牙利算法匹配检测框和追踪框
    :param detected_boxes: list of np.array([[x1,y1], [x2,y2]])
    :param tracked_boxes: list of np.array([[x1,y1], [x2,y2]])
    :param iou_threshold: float, 低于此阈值的IOU将被视为不匹配
    :return: matched pairs (det_idx, track_idx), unmatched_tracked list, unmatched_detected list
    """
    num_dets = len(detected_boxes)
    num_tracks = len(tracked_boxes)

    if num_dets == 0 or num_tracks == 0:
        return [], list(range(num_tracks)), list(range(num_dets))

    # 构建成本矩阵 (1 - IOU)，越小越好
    cost_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)
    for i, det_box in enumerate(detected_boxes):
        for j, trk_box in enumerate(tracked_boxes):
            iou = compute_iou(det_box, trk_box)
            cost_matrix[i, j] = 1 - iou

    # 匈牙利算法（最小成本匹配）
    det_indices, trk_indices = linear_sum_assignment(cost_matrix)

    matched = []
    unmatched_dets = list(range(num_dets))
    unmatched_trks = list(range(num_tracks))

    for d, t in zip(det_indices, trk_indices):
        if 1 - cost_matrix[d, t] >= iou_threshold:
            matched.append((d, t))
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)

    return matched, unmatched_trks, unmatched_dets
def run_detecton(gt_images=None,gt_annotations = None,image_idx = None ,use_gt = True,image_path=None,detector=None):

    if use_gt:
        filename = next((k for k, v in gt_images.items() if v == image_idx), None)
        if filename is not None:
            detections = [ann for ann in gt_annotations if ann['image_id'] == image_idx]
            detected_person_bboxes = [det['bbox'] for det in detections]  # bbox in COCO format: [x, y, width, height]
        else:
            detected_person_bboxes = []
    else:
        if detector is not None and image_path is not None:
            results = detector.predict(image_path, verbose=False)
            result = results[0]
            detected_person_bboxes = []
            for i in range(len(result.boxes.cls)):
                label = int(result.boxes.cls[i])
                if label == 0:  # class 0 is 'person' in COCO
                    x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                    detected_person_bboxes.append([x1, y1, x2 - x1, y2 - y1])  # COCO format
        else:
            detected_person_bboxes = []
            print(image_path)
            print(detector)
            breakpoint()

    
    return detected_person_bboxes
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay mask on image with transparency"""
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = ((1 - alpha) * image[mask] + alpha * np.array(color)).astype(np.uint8)
    return overlay
def draw_bbox(image, bbox, color=(255, 0, 0)):
    tl, br = bbox[0], bbox[1]
    x1, y1 = int(tl[0]), int(tl[1])
    x2, y2 = int(br[0]), int(br[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image
def id_to_color(obj_id):
    """
    Generate a consistent RGB color for a given object ID.
    """
    np.random.seed(obj_id)  # Ensures same ID gets same color
    return tuple(int(x) for x in np.random.randint(0, 256, size=3))



def vis(frame_data, scale=0.7, out_dir=None):
    """
    Visualize tracking results for the current frame.

    Args:
        frame_data (dict): Dictionary mapping image_path to object tracking data (masks in RLE, bboxes).
        scale (float): Scaling factor for visualization.
        out_dir (str, optional): If provided, saves output images to this directory.
    """
    for image_path, objects in frame_data.items():
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[Warning] Could not read image: {image_path}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for obj_id, data in objects.items():
            bbox = data.get('bbox')
            mask_rle = data.get('mask')

            # Skip invalid bbox
            if not (bbox and isinstance(bbox, list) and len(bbox) == 2):
                print(f"[Warning] Skipping object {obj_id}: invalid bbox format.")
                continue

            # Decode RLE mask
            if not (mask_rle and isinstance(mask_rle, dict) and 'counts' in mask_rle):
                print(f"[Warning] Skipping object {obj_id}: invalid or missing RLE mask.")
                continue

            try:
                mask = mask_utils.decode(mask_rle)
                mask = np.squeeze(mask)  # shape: (H, W)
            except Exception as e:
                print(f"[Error] Decoding RLE for object {obj_id}: {e}")
                continue

            color = id_to_color(obj_id)

            # Overlay mask
            frame_rgb = overlay_mask(frame_rgb, mask, color=color)

            # Draw bounding box and label
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame_rgb, f'ID: {obj_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Resize and display
        frame_rgb_resized = cv2.resize(frame_rgb, (0, 0), fx=scale, fy=scale)
        frame_bgr = cv2.cvtColor(frame_rgb_resized, cv2.COLOR_RGB2BGR)
        if out_dir:
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
            output_path = os.path.join(out_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, frame_bgr)
            # breakpoint()
        else:
            # Display the image
            cv2.imshow("Tracking", frame_bgr)
            cv2.waitKey()

def run_tracking(image_sequence, window_size,GT_Detection_DIR=None,use_gt = False, out_dir = None):
    """
    运行追踪算法
    :param image_sequence: list of images_paths
    :param window_size: int, 窗口大小
    :param GT_Detection_DIR: str, ground truth detection directory if available, it is a json file wth coco format 
    :param OUTPUT_DIR: str, output directory
    :return: dict
    """
    data = {}
    i= 0
    current_track_pool = []
    total_tracked = 0
    weight_path="/home/han/Documents/github/HALO/mmor_human/src/detectron2_training/yolo/runs/detect/4d_or_all_person_w_patient/weights/best.pt"
    # load GT_Detection_DIR json
    if use_gt and GT_Detection_DIR:
        with open(GT_Detection_DIR, 'r') as f:
            gt_data = json.load(f)
        gt_annotations = gt_data['annotations']
        gt_images = {img['file_name']: img['id'] for img in gt_data['images']}
        # breakpoint()
    else:
        detector = YOLO(weight_path) # Load your weights
        gt_annotations = []
        gt_images = {}

    # change to tqdm
    for idx, image_path in tqdm(enumerate(image_sequence), total=len(image_sequence)):
        frame_data = {}
        image_idx = int(image_path.split('/')[-1].split('.')[0])
        # first frame initialization
        if not current_track_pool:
            detected_people_list = run_detecton(
                gt_images=gt_images,
                gt_annotations=gt_annotations,
                image_idx=image_idx,
                use_gt=use_gt,
                image_path=image_path,
                detector=detector
            )
            if not detected_people_list:
                frame_data[image_path] = {}
                data[image_path] = {}
                # vis(frame_data,out_dir=out_dir)
                continue
            boxes = [np.array([[x, y], [x + w, y + h]]) for x, y, w, h in detected_people_list]
            assigned_ids = list(range(total_tracked, total_tracked + len(boxes)))
            total_tracked += len(boxes)
            tracker = SAM2()
            current_track_pool = assigned_ids.copy()
            tracker.load_first_frame(image_path)
            res = tracker.set_bbox(boxes=boxes, ann_obj_id=assigned_ids)
            # frame_data[image_path] = res
            for obj in res.values():
                if isinstance(obj['bbox'], np.ndarray):
                    obj['bbox'] = obj['bbox'].tolist()
            frame_data[image_path] = res
            data[image_path] = res
            # print('first frame initialization')
        # in window 
        # if detected people number is same as current track pool, just let it propogate else reinitialize
        elif idx % window_size != window_size - 1:
            detected_people_list = run_detecton(
                gt_images=gt_images,
                gt_annotations=gt_annotations,
                image_idx=image_idx,
                use_gt=use_gt,
                image_path=image_path,
                detector=detector
            )

            if not detected_people_list:
                print(f"[Warning] No detections in frame {image_idx}")
                frame_data[image_path] = {}
                data[image_path] = {}
                # vis(frame_data,out_dir=out_dir)
                continue

            detected_boxes = [np.array([[x, y], [x + w, y + h]]) for x, y, w, h in detected_people_list]

            if len(detected_boxes) <= len(current_track_pool):
                # Case 1: Same number of detections as current tracks — propagate
                tracker.load_image(image_path)
                res = tracker.propagate_single_frame()

                # Filter valid results
                # valid_results = {
                #     tid: obj for tid, obj in res.items()
                #     if isinstance(obj.get('bbox'), np.ndarray)
                # }
                valid_results = {}
                for tid, obj in res.items():
                    bbox = obj.get('bbox')
                    if bbox is not None and isinstance(bbox, list) and len(bbox) == 2:
                        valid_results[tid] = obj


                current_track_pool = list(valid_results.keys())
                frame_data[image_path] = valid_results
                for obj in valid_results.values():
                    if isinstance(obj['bbox'], np.ndarray):
                        obj['bbox'] = obj['bbox'].tolist()
                data[image_path] = valid_results
                # print("In window (propagation only)")
            else:
                # Case 2: Count mismatch — reinitialize with ID matching
                tracker.load_image(image_path)
                res = tracker.propagate_single_frame()

                # Gather tracked bboxes and their corresponding IDs
                tracked_boxes = []
                tracked_ids = []
                for tid, obj in res.items():
                    bbox = obj.get('bbox')
                    # if isinstance(bbox, np.ndarray) and bbox.shape == (2, 2):
                    if bbox is not None and isinstance(bbox, list) and len(bbox) == 2:

                        tracked_boxes.append(bbox)
                        tracked_ids.append(tid)

                # Match detections with tracked boxes
                matched, unmatched_tracked, unmatched_detected = hungarian_match(
                    detected_boxes, tracked_boxes
                )

                matched_ids = []
                matched_boxes = []

                for det_idx, trk_idx in matched:
                    matched_ids.append(tracked_ids[trk_idx])
                    matched_boxes.append(detected_boxes[det_idx])

                # Assign new IDs to unmatched detections
                for det_idx in unmatched_detected:
                    matched_ids.append(total_tracked)
                    matched_boxes.append(detected_boxes[det_idx])
                    total_tracked += 1

                # Reset tracker with matched results
                tracker = SAM2()
                tracker.load_first_frame(image_path)
                res = tracker.set_bbox(boxes=matched_boxes, ann_obj_id=matched_ids)

                current_track_pool = list(res.keys())
                for obj in res.values():
                    if isinstance(obj['bbox'], np.ndarray):
                        obj['bbox'] = obj['bbox'].tolist()
                frame_data[image_path] = res
                data[image_path] = res
                print("In window (count mismatch — reinitialized with ID matching)")

        else:
            detected_people_list = run_detecton(
                gt_images=gt_images,
                gt_annotations=gt_annotations,
                image_idx=image_idx,
                use_gt=use_gt,
                image_path=image_path,
                detector=detector
            )
            if not detected_people_list:
                print(f"[Warning] No detections in frame {image_idx}")
                frame_data[image_path] = {}
                data[image_path] = {}
                # vis(frame_data,out_dir=out_dir)
                continue

            detected_boxes = [np.array([[x, y], [x + w, y + h]]) for x, y, w, h in detected_people_list]

            tracker.load_image(image_path)
            res = tracker.propagate_single_frame()

            # Filter out any None or invalid bboxes
            valid_ids = []
            current_tracked_bboxes = []

            for tid in res:
                bbox = res[tid].get('bbox')
                if bbox is not None:
                    valid_ids.append(tid)
                    current_tracked_bboxes.append(bbox)

            tracked_ids = valid_ids

            matched, unmatched_tracked, unmatched_detected = hungarian_match(
                detected_boxes, current_tracked_bboxes
            )

            # Reset tracker
            tracker = SAM2()
            tracker.load_first_frame(image_path)

            updated_boxes = []
            updated_ids = []

            # Update matched boxes
            for det_idx, track_idx in matched:
                tid = tracked_ids[track_idx]
                updated_boxes.append(detected_boxes[det_idx])
                updated_ids.append(tid)

            # Initialize new tracks for unmatched detections
            for det_idx in unmatched_detected:
                updated_boxes.append(detected_boxes[det_idx])
                updated_ids.append(total_tracked)
                total_tracked += 1

            if updated_boxes:
                res = tracker.set_bbox(boxes=updated_boxes, ann_obj_id=updated_ids)
            else:
                res = {}

            current_track_pool = list(res.keys())
            for obj in res.values():
                if isinstance(obj['bbox'], np.ndarray):
                    obj['bbox'] = obj['bbox'].tolist()
            frame_data[image_path] = res
            data[image_path] = res
            # print('Reset')
            # printed detected_boxes
            # print(detected_boxes)
        vis(frame_data,out_dir=out_dir)
        # vis(frame_data)
        i+= 1
        # if i > 100:
            # return data
        # print(f"Processing frame {idx + 1}/{len(image_sequence)}")
        # print(f"Current track pool: {current_track_pool}")
    # breakpoint()
    if out_dir is not None:
        mot_lines = []
        for image_path in sorted(data.keys()):
            frame_id = int(os.path.basename(image_path).split('.')[0])
            for track_id, track_data in data[image_path].items():
                if 'bbox' in track_data and track_data['bbox']:
                    bbox = track_data['bbox']  # [[x1,y1],[x2,y2]]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    w = x2 - x1
                    h = y2 - y1
                    line = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1"
                    mot_lines.append(line)

        mot_path = os.path.join(out_dir, "tracking_results_mot.txt")
        with open(mot_path, "w") as f:
            for line in mot_lines:
                f.write(line + "\n")
        print(f"[Info] MOT format results saved to {mot_path}")

    return data
