"""
dataset.py

Dataset class for loading pose sequences.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseSequenceDataset(Dataset):
    """
    Loads .npy pose sequence files from a directory.
    Each file should be shape (seq_len, 34) where 34 = 17 joints * 2 (x, y).
    """

    def __init__(self, pose_dir: Path):
        """
        Args:
            pose_dir: Path to directory containing .npy files
        """
        self.pose_dir = Path(pose_dir)
        self.files = sorted(self.pose_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {self.pose_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns (pose_sequence, dummy_label) tuple.
        pose_sequence: torch.Tensor of shape (seq_len, 34)
        dummy_label: 0 (unused, for compatibility with standard PyTorch loaders)
        """
        pose = np.load(self.files[idx])
        pose = torch.tensor(pose, dtype=torch.float32)
        return pose, 0

    def summary(self):
        """
        Returns a summary string about the dataset.
        """
        if self.files:
            sample = np.load(self.files[0])
            return f"Loaded {len(self.files)} sequences, shape per sequence: {sample.shape}"
        return "Empty dataset"
