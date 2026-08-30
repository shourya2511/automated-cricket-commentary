"""
pose_extraction.py  (v3 - fixes degenerate/empty-frame issues found during
visual verification)

Two fixes over v2:

1. TWO-DIRECTIONAL FILL instead of forward-only fill.
   Previously, if the very first frame of a window failed detection,
   there was no "previous" frame to repeat, so it defaulted to all
   zeros (visually: an empty/collapsed skeleton). Now we first collect
   raw per-frame keypoints (or None if detection failed), then run a
   fill pass: for any None, look forward for the nearest valid frame
   (backward-fill in time), and only fall back to zeros if truly no
   frame in the whole window had a valid detection.

2. CONFIDENCE/VISIBILITY CHECK on MediaPipe output.
   Previously any MediaPipe result was accepted as-is, even if the
   crop was bad (too small, mostly background) and the landmarks came
   out degenerate (all clustered together). Now we check MediaPipe's
   per-landmark visibility score and reject the frame (treat as a
   failed detection, to be filled) if average visibility is too low.

For each clip, data/contact_frames.json gives a list of instances:
    {"start": <frame where this delivery begins, right after the cut>,
     "contact": <frame of bat-ball contact>}

Run:
    python src/preprocessing/pose_extraction.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips" / "cover_drives"
CONTACT_JSON = PROJECT_ROOT / "data" / "contact_frames.json"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pose_sequences"

FRAMES_BEFORE = 10
FRAMES_AFTER = 19   # + contact frame itself = 30 total
WINDOW_SIZE = FRAMES_BEFORE + FRAMES_AFTER + 1

BOX_PADDING = 0.15  # 15% padding around the YOLO box
MIN_AVG_VISIBILITY = 0.5  # reject MediaPipe result if average landmark visibility is below this
MIN_BOX_HEIGHT_FRAC = 0.15  # reject YOLO box if its height is less than 15% of frame height
                             # (a real batsman crop should take up a meaningful chunk of the frame;
                             # tiny boxes are usually spurious/wrong detections that later produce
                             # a degenerate, collapsed-looking pose from MediaPipe)
MIN_YOLO_CONF = 0.45  # reject YOLO detections below this confidence

KEEP_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_heel": 29,     "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}
assert len(KEEP_LANDMARKS) == 17

yolo_model = YOLO(str(YOLO_WEIGHTS))
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)


def get_batsman_box(frame):
    """Returns the best batsman box, or None if nothing plausible was found.
    Applies two sanity filters beyond just 'highest confidence':
      - confidence must clear MIN_YOLO_CONF
      - box height must be a plausible fraction of the frame height
        (rejects tiny/spurious boxes that would otherwise produce a
        degenerate, collapsed-looking MediaPipe pose)
    """
    frame_h = frame.shape[0]
    results = yolo_model(frame, verbose=False)[0]
    best_box = None
    best_conf = -1
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf = float(box.conf[0])
        if cls_name != "batsman" or conf <= best_conf:
            continue
        if conf < MIN_YOLO_CONF:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_h = y2 - y1
        if box_h < MIN_BOX_HEIGHT_FRAC * frame_h:
            continue  # too small to plausibly be a real batsman crop
        best_conf = conf
        best_box = [x1, y1, x2, y2]
    return best_box


def crop_with_padding(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    x1 -= box_w * BOX_PADDING
    x2 += box_w * BOX_PADDING
    y1 -= box_h * BOX_PADDING
    y2 += box_h * BOX_PADDING
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return frame[y1:y2, x1:x2]


def extract_keypoints(crop):
    """Returns a (34,) keypoint array, or None if detection failed or
    was too low-confidence to trust (degenerate/garbage pose)."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if result.pose_landmarks is None:
        return None

    landmarks = result.pose_landmarks.landmark

    visibilities = [landmarks[idx].visibility for idx in KEEP_LANDMARKS.values()]
    avg_visibility = sum(visibilities) / len(visibilities)
    if avg_visibility < MIN_AVG_VISIBILITY:
        return None  # reject low-confidence / likely-degenerate pose

    keypoints = []
    for name, idx in KEEP_LANDMARKS.items():
        lm = landmarks[idx]
        keypoints.extend([lm.x, lm.y])
    return np.array(keypoints, dtype=np.float32)


def get_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def fill_gaps(raw_kpts_list):
    """raw_kpts_list: list of (34,) arrays or None.
    Returns a fully-filled list (no Nones), using nearest-neighbor fill
    in time: for each None, look for the nearest valid frame, preferring
    to look backward first, then forward. Only if NO frame in the whole
    list is valid do we fall back to zeros (logs a warning in that case)."""
    n = len(raw_kpts_list)
    valid_indices = [i for i, k in enumerate(raw_kpts_list) if k is not None]

    if not valid_indices:
        print("    WARNING: no valid detections in entire window - using zeros throughout.")
        return [np.zeros(34, dtype=np.float32) for _ in range(n)]

    filled = list(raw_kpts_list)
    for i in range(n):
        if filled[i] is not None:
            continue
        # find nearest valid index (by absolute distance)
        nearest = min(valid_indices, key=lambda v: abs(v - i))
        filled[i] = raw_kpts_list[nearest]

    return filled


def extract_sequence(cap, contact_frame, seg_start, seg_end):
    """Window clamped within [seg_start, seg_end], padded by repeating edge frame,
    with two-directional gap-filling for failed/rejected detections."""
    start = max(seg_start, contact_frame - FRAMES_BEFORE)
    end = min(seg_end, contact_frame + FRAMES_AFTER)

    raw_indices = list(range(start, end + 1))

    while len(raw_indices) < WINDOW_SIZE and raw_indices[0] > seg_start:
        raw_indices.insert(0, raw_indices[0] - 1)
    while len(raw_indices) < WINDOW_SIZE:
        raw_indices.insert(0, raw_indices[0])

    while len(raw_indices) < WINDOW_SIZE and raw_indices[-1] < seg_end:
        raw_indices.append(raw_indices[-1] + 1)
    while len(raw_indices) < WINDOW_SIZE:
        raw_indices.append(raw_indices[-1])
    raw_indices = raw_indices[:WINDOW_SIZE]

    raw_kpts_list = []
    for idx in raw_indices:
        frame = get_frame(cap, idx)
        kpts = None
        if frame is not None:
            box = get_batsman_box(frame)
            if box is not None:
                crop = crop_with_padding(frame, box)
                if crop.size > 0:
                    kpts = extract_keypoints(crop)
        raw_kpts_list.append(kpts)

    filled = fill_gaps(raw_kpts_list)
    sequence = np.stack(filled, axis=0)
    assert sequence.shape == (WINDOW_SIZE, 34)
    return sequence


def process_clip(clip_name, instances):
    clip_path = CLIPS_DIR / clip_name
    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_paths = []

    for i, inst in enumerate(instances):
        seg_start = inst["start"]
        seg_end = (instances[i + 1]["start"] - 1) if i + 1 < len(instances) else (total_frames - 1)
        contact = inst["contact"]

        print(f"  Instance {i+1}: start={seg_start} contact={contact} end={seg_end}")

        sequence = extract_sequence(cap, contact, seg_start, seg_end)

        base_name = Path(clip_name).stem
        out_name = f"pose_seq_{base_name}_{i+1:02d}.npy"
        out_path = OUTPUT_DIR / out_name
        np.save(out_path, sequence)
        saved_paths.append(out_path)
        print(f"    -> saved {out_path}  shape={sequence.shape}")

    cap.release()
    return saved_paths


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONTACT_JSON, "r", encoding="utf-8") as f:
        contact_data = json.load(f)

    total_sequences = 0
    for clip_name, instances in contact_data.items():
        print(f"Processing {clip_name} ({len(instances)} instances)...")
        saved = process_clip(clip_name, instances)
        total_sequences += len(saved)

    print(f"\nDone. {total_sequences} pose sequences saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()