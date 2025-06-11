import os
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor

import sys
sam2_checkpoint = "/home/han/Documents/github/HALO/mmor_human/src/SAM2_Tracking/third_party/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
sys.path.append("/home/han/Documents/github/HALO/mmor_human/src/SAM2_Tracking/third_party/segment-anything-2-real-time")
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

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

# --------------- 视频读取和模型初始化 ---------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

predictor.load_first_frame(frame)

# --------------- 设置多个目标框 ---------------
bboxes = [
    np.array([[150, 320], [330, 400]], dtype=np.float32),
    np.array([[300, 100], [500, 400]], dtype=np.float32)
]
obj_ids = list(range(len(bboxes)))  # [0, 1]
# breakpoint()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --------------- 添加多个目标 ---------------
for obj_id, bbox in zip(obj_ids, bboxes):
    _, _, out_mask_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=obj_id,
        bbox=bbox,
    )
    frame_rgb = draw_bbox(frame_rgb, bbox)

    # 显示当前掩码
    mask_np = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
    color = (0, 255, 0) if obj_id == 0 else (255, 0, 0)
    frame_rgb = overlay_mask(frame_rgb, mask_np, color=color)

cv2.imshow("SAM2 Tracking - Initial Frame", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(1)

# --------------- 开始多目标追踪 ---------------
ann_frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    ann_frame_idx += 1

    out_obj_ids, out_mask_logits = predictor.track(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 可视化所有对象掩码
    for obj_idx, mask_logits in enumerate(out_mask_logits):
        mask_np = (mask_logits > 0.0).cpu().numpy().squeeze()
        color = (0, 255, 0) if obj_idx == 0 else (255, 0, 0)  # 每个对象不同颜色
        frame_rgb = overlay_mask(frame_rgb, mask_np, color=color)

    cv2.imshow("SAM2 Tracking", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(30)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()