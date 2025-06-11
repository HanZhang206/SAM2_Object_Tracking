import os
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor

import sys
sam2_checkpoint = "/home/han/Documents/github/HALO/mmor_human/src/SAM2_Tracking/third_party/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sys.path.append("/home/han/Documents/github/HALO/mmor_human/src/SAM2_Tracking/third_party/segment-anything-2-real-time")
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay mask on image with transparency"""
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = ((1 - alpha) * image[mask] + alpha * np.array(color)).astype(np.uint8)
    return overlay

def draw_points(image, coords, labels):
    for (x, y), label in zip(coords, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.drawMarker(image, (int(x), int(y)), color, markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)
    return image

def draw_bbox(image, bbox):
    tl, br = bbox[0], bbox[1]
    x1, y1 = int(tl[0]), int(tl[1])
    x2, y2 = int(br[0]), int(br[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

# Capture video
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

predictor.load_first_frame(frame)

using_point = False
using_box = True
using_mask = False

ann_frame_idx = 0
ann_obj_id = 0

points = np.array([[247, 313]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
bbox = np.array([[173, 160], [600, 300]], dtype=np.float32)

# Initial frame processing
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if using_point:
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    frame_rgb = draw_points(frame_rgb, points, labels)

elif using_box:
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        bbox=bbox,
    )
    frame_rgb = draw_bbox(frame_rgb, bbox)

elif using_mask:
    mask_img_path = "masks/aquarium/aquarium_mask.png"
    mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0

    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=mask,
    )

mask_np = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
frame_rgb = overlay_mask(frame_rgb, mask_np)

cv2.imshow("SAM2 Tracking - Initial Frame", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(1)

# Track and visualize
vis_gap = 5
while True:
    ret, frame = cap.read()
    ann_frame_idx += 1
    if not ret:
        break

    out_obj_ids, out_mask_logits = predictor.track(frame)
    # if ann_frame_idx % vis_gap == 0:
    # print(f"frame {ann_frame_idx}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask_np = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    frame_rgb = overlay_mask(frame_rgb, mask_np)
    cv2.imshow("SAM2 Tracking", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(30)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()