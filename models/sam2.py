import torch
import cv2
from sam2.build_sam import build_sam2_camera_predictor
import numpy as np
from pycocotools import mask as mask_utils

class SAM2:
    def __init__(self):
        sam2_checkpoint = "/home/han/Documents/github/HALO/mmor_human/src/SAM2_Tracking/third_party/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.ann_frame_idx = 0
        self.object_id = []
        self.current_mask = []
        self.current_frame = None
    def set_bbox(self, boxes=[], ann_obj_id=[]):
        """
        Initialize the tracker with bounding boxes and object IDs.

        Args:
            boxes (list of np.array): List of bounding boxes, e.g., [np.array([[x1, y1], [x2, y2]])].
            ann_obj_id (list): List of corresponding object IDs.

        Returns:
            dict: Mapping from object ID to their initial mask and bbox.
        """
        result = {}
        self.object_id = [] 
        for obj_id, bbox in zip(ann_obj_id, boxes):
            
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=obj_id,
                bbox=bbox,
            )
        # breakpoint()
        for obj_id, mask in zip(out_obj_ids, out_mask_logits):
            self.object_id.append(obj_id)
            # breakpoint()
            mask_np = (mask > 0.0).cpu().numpy().squeeze()
            mask_np = mask_np.astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                all_points = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                bbox = np.array([[x, y], [x + w, y + h]])
            else:
                bbox = None
            mask_rle = mask_utils.encode(np.asfortranarray(mask_np[:, :, None]))[0]
            mask_rle = {
                'size': [int(s) for s in mask_rle['size']],
                'counts': mask_rle['counts'].decode('utf-8')
            }

            result[obj_id] = {
                'mask': mask_rle,
                'bbox': bbox.tolist() if bbox is not None else None
            }
            # breakpoint()

        self.current_mask = [result[obj_id]['mask'] for obj_id in ann_obj_id]
        return result
    def load_first_frame(self, image_path):
        self.current_frame = cv2.imread(image_path)
        self.predictor.load_first_frame(self.current_frame)
        if self.current_frame is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        

    def load_image(self, image_path):
        """
        Load image from file path.

        Args:
            image_path (str): Path to the image file.
        """
        self.current_frame = cv2.imread(image_path)
        if self.current_frame is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

    def propagate_single_frame(self):
        """
        Propagate the current tracked masks to the next frame.

        Returns:
            dict: Mapping from object ID to updated mask and bbox.
        """
        if self.current_frame is None:
            raise ValueError("No image loaded. Use `load_image()` first.")

        out_obj_ids, out_mask_logits = self.predictor.track(self.current_frame)
        result = {}

        for obj_idx, mask_logits in enumerate(out_mask_logits):
            mask_np = (mask_logits > 0.0).cpu().numpy().squeeze()
            # change the mask to a bbox x y w h
            mask_np = mask_np.astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                all_points = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                bbox = np.array([[x, y], [x + w, y + h]])
            else:
                bbox = None

            obj_id = self.object_id[obj_idx]
            mask_rle = mask_utils.encode(np.asfortranarray(mask_np[:, :, None]))[0]
            mask_rle = {
                'size': [int(s) for s in mask_rle['size']],
                'counts': mask_rle['counts'].decode('utf-8')
            }

            result[obj_id] = {
            'mask': mask_rle,
            'bbox': bbox.tolist() if bbox is not None else None  # Optional: make bbox JSON-serializable
        }

        self.current_mask = [result[obj_id]['mask'] for obj_id in self.object_id]
        self.ann_frame_idx += 1
        return result
