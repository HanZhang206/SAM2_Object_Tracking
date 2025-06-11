from models.sam2 import SAM2
# from models.detectron2_wrapper import Detectron2
from tracker.tracking_core import run_tracking
from utils.io_utils import load_image_sequence
from config import WINDOW_SIZE, IMAGE_SEQUENCE_DIR, GT_Detection_DIR, OUTPUT_DIR
import os
import json

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_sequence = load_image_sequence(IMAGE_SEQUENCE_DIR)
    data = run_tracking(image_sequence, WINDOW_SIZE,GT_Detection_DIR=GT_Detection_DIR, use_gt=False,out_dir=OUTPUT_DIR)

    # breakpoint()
    json_name = OUTPUT_DIR+'/outputs.json'
    with open(json_name, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()