"""
verify_pose_sequence.py

Quick visual sanity check: loads a saved pose sequence and plots the
skeleton keypoints for a few frames side by side, so you can eyeball
whether the pose tracking looks correct (limbs in sensible positions,
motion progressing through the swing) or broken (points scattered
randomly, frozen/static across frames, etc.)

Usage:
    python src/preprocessing/verify_pose_sequence.py pose_seq_cover1_01.npy
    python src/preprocessing/verify_pose_sequence.py pose_seq_cover2_03.npy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_DIR = PROJECT_ROOT / "data" / "pose_sequences"

# same order as KEEP_LANDMARKS in pose_extraction.py
LANDMARK_NAMES = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# simple skeleton connections for a clearer visual (pairs of landmark indices)
SKELETON_EDGES = [
    (1, 2),    # shoulders
    (1, 3), (3, 5),    # left arm
    (2, 4), (4, 6),    # right arm
    (1, 7), (2, 8),    # shoulder to hip
    (7, 8),    # hips
    (7, 9), (9, 11), (11, 13), (11, 15),   # left leg
    (8, 10), (10, 12), (12, 14), (12, 16), # right leg
]


def plot_sequence(npy_path, frames_to_show=(0, 10, 20, 29)):
    seq = np.load(npy_path)  # shape (30, 34)
    assert seq.shape == (30, 34), f"Unexpected shape: {seq.shape}"

    fig, axes = plt.subplots(1, len(frames_to_show), figsize=(4 * len(frames_to_show), 4))
    fig.suptitle(f"Pose sequence: {npy_path.name}")

    for ax, frame_idx in zip(axes, frames_to_show):
        kpts = seq[frame_idx].reshape(17, 2)  # (17, 2) x,y

        xs, ys = kpts[:, 0], kpts[:, 1]
        ax.scatter(xs, ys, c="red", s=20, zorder=3)

        for i, j in SKELETON_EDGES:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], c="blue", linewidth=1.5, zorder=2)

        ax.set_title(f"Frame {frame_idx}")
        ax.invert_yaxis()  # image coords: y grows downward
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")

    plt.tight_layout()
    out_path = npy_path.with_suffix(".png")
    plt.savefig(out_path, dpi=100)
    print(f"Saved visualization to {out_path}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_pose_sequence.py <filename.npy>")
        print(f"Available files in {POSE_DIR}:")
        for f in sorted(POSE_DIR.glob("*.npy")):
            print(f"  {f.name}")
        return

    filename = sys.argv[1]
    npy_path = POSE_DIR / filename
    if not npy_path.exists():
        print(f"File not found: {npy_path}")
        return

    plot_sequence(npy_path)


if __name__ == "__main__":
    main()