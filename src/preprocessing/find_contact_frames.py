"""
find_contact_frame.py  (v2 - handles multiple spliced instances per clip)
 
Each raw clip contains 5-6 separate cover drive instances, spliced
together with hard cuts. For each instance we need:
    - start frame (right after the cut, where this instance begins)
    - contact frame (bat hits ball)
 
Controls while stepping through a clip:
    d / next   -> step forward one frame
    a / prev   -> step backward one frame
    b          -> mark BEGIN of a new instance at current frame
    s          -> mark current frame as CONTACT for the current instance
    n          -> finish this clip, move to next clip
    q          -> quit entirely (progress so far is saved)
 
Typical flow per clip:
    step to first delivery's start -> press b
    step to that delivery's contact moment -> press s
    step forward past the cut to next delivery's start -> press b
    step to its contact moment -> press s
    ... repeat for all instances in the clip ...
    press n when done with this clip
 
Output: data/contact_frames.json, structured as:
{
  "cover1.mp4": [
    {"start": 10, "contact": 87},
    {"start": 160, "contact": 210}
  ]
}
("end" of each instance is inferred later as next instance's start - 1,
or last frame of clip for the final instance.)
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips" / "cover_drives"
OUTPUT_JSON = PROJECT_ROOT / "data" / "contact_frames.json"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pose_sequences"


def ensure_runtime_environment():
    try:
        import cv2
    except ModuleNotFoundError:
        venv_python = PROJECT_ROOT / "cric_env" / "Scripts" / "python.exe"
        print("OpenCV is not available in the active Python environment.", file=sys.stderr)
        print("Use the project virtual environment instead:", file=sys.stderr)
        print(f'  "{venv_python}" "{PROJECT_ROOT / "src" / "preprocessing" / "find_contact_frames.py"}"', file=sys.stderr)
        raise SystemExit(1)

    if not CLIPS_DIR.exists():
        print(f"Clip directory not found: {CLIPS_DIR}", file=sys.stderr)
        print("Run this script from the project root or make sure the raw clips exist.", file=sys.stderr)
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return cv2

def load_existing_results():
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def mark_clip(clip_path, cv2):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    instances = []           # list of {"start": int, "contact": int or None}
    current_instance = None
    quit_flag = False
    print(f"Clips : {clip_path} ({total_frames} Frames)")
    print("b=mark start | s=mark contact | d/a=step | p=prev marked start | c=fix last contact | n=next clip | q=quit")
    cv2.namedWindow("Mark Instances", cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            break

        display = frame.copy()
        info = f"Frame {frame_idx}/{total_frames-1} | instances marked : {len(instances)}"
        cv2.putText(display, info, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if current_instance is not None:
            status = f"Current instance start = {current_instance['start']} contact = {current_instance['contact']}"
            cv2.putText(display, status, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Mark Instances", display)
        key = cv2.waitKey(0) & 0xFF
 
        if key == ord('d'):
            frame_idx = min(frame_idx + 1, total_frames - 1)
 
        elif key == ord('a'):
            frame_idx = max(frame_idx - 1, 0)

        elif key == ord('p'):
            if not instances:
                print("  No previous instance exists.")
            else:
                prev_start = instances[-1]["start"]
                frame_idx = prev_start
                current_instance = instances[-1]
                print(f"  Jumped to previous instance start at frame {frame_idx}")

        elif key == ord('c'):
            if not instances:
                print("  No contact to correct yet.")
            else:
                last_instance = instances[-1]
                last_instance["contact"] = frame_idx
                current_instance = last_instance
                print(f"  Last contact corrected to frame {frame_idx}")
 
        elif key == ord('b'):
            if current_instance is not None and current_instance["contact"] is None:
                print("  Warning: previous instance had no contact frame marked, discarding it.")
                instances.pop()
            current_instance = {"start": frame_idx, "contact": None}
            instances.append(current_instance)
            print(f"  New instance started at frame {frame_idx}")
 
        elif key == ord('s'):
            if current_instance is None:
                print("  No instance started yet - press 'b' first.")
            else:
                current_instance["contact"] = frame_idx
                print(f"  Contact marked at frame {frame_idx}")
 
        elif key == ord('n'):
            if current_instance is not None and current_instance["contact"] is None:
                print("  Warning: last instance had no contact frame, discarding it.")
                instances.pop()
            break
 
        elif key == ord('q'):
            if current_instance is not None and current_instance["contact"] is None:
                instances.pop()
            quit_flag = True
            break
 
    cap.release()
    cv2.destroyAllWindows()
    return instances, quit_flag
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Mark contact frames for cricket cover-drive clips.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess clips that already exist in the output JSON file.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="Only process a specific clip filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cv2 = ensure_runtime_environment()
    results = load_existing_results()
    clip_files = sorted([
        f for f in os.listdir(CLIPS_DIR)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    ])

    if args.clip is not None:
        clip_files = [args.clip] if args.clip in clip_files else []

    for clip_name in clip_files:
        if clip_name in results and not args.force:
            print(f"Already marked: {clip_name} ({len(results[clip_name])} instances)")
            continue

        clip_path = CLIPS_DIR / clip_name
        instances, quit_flag = mark_clip(clip_path, cv2)
        if instances:
            results[clip_name] = instances
            save_results(results)
            print(f"Saved {len(instances)} instances for {clip_name}")

        if quit_flag:
            print("Quitting. Progress saved.")
            return

    print(f"\nAll clips done. Results in {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
 
