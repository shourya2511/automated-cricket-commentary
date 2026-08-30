"""
debug_crops.py

For a specific clip + frame range, saves the raw YOLO-cropped images
so you can visually inspect exactly who/what is being detected and
fed into MediaPipe. Use this when a pose sequence looks wrong and you
need to confirm whether YOLO is tracking the batsman or someone else
(e.g. the bowler).

Usage:
    python src/preprocessing/debug_crops.py cover3.mp4 400 470

This saves crops for frames 400-470 of cover3.mp4 into
data/debug_crops/cover3_400_470/
"""

import sys
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips" / "cover_drives"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "best.pt"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug_crops"

yolo_model = YOLO(str(YOLO_WEIGHTS))


def get_batsman_box(frame):
    results = yolo_model(frame, verbose=False)[0]
    best_box = None
    best_conf = -1
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf = float(box.conf[0])
        if cls_name == "batsman" and conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0].tolist()
    return best_box, best_conf


def main():
    if len(sys.argv) < 4:
        print("Usage: python debug_crops.py <clip_name> <start_frame> <end_frame>")
        return

    clip_name = sys.argv[1]
    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])

    clip_path = CLIPS_DIR / clip_name
    cap = cv2.VideoCapture(str(clip_path))

    out_dir = DEBUG_DIR / f"{Path(clip_name).stem}_{start_frame}_{end_frame}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        box, conf = get_batsman_box(frame)

        annotated = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"conf={conf:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"frame {frame_idx}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), annotated)

    cap.release()
    print(f"Saved annotated frames to {out_dir}")
    print("Open a few of these images and check: is the green box around the batsman or someone else?")


if __name__ == "__main__":
    main()