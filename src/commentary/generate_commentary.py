"""
generate_commentary.py

End-to-end Day 5 demo: takes a trained autoencoder + a pose sequence,
computes reconstruction error, builds context, generates commentary via
Gemini, and saves audio via gTTS.

Usage (from project root):
    python src/commentary/generate_commentary.py pose_seq_cover1_01.npy
"""

import sys
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "lstm_model"))

if TYPE_CHECKING:
    from models.lstm_model.model import LSTMAutoencoder, reconstruction_error
else:
    from model import LSTMAutoencoder, reconstruction_error
from context_builder import build_context
from gemini_commentary import generate_commentary
from text_to_speech import text_to_audio

POSE_SEQUENCES_DIR = PROJECT_ROOT / "data" / "pose_sequences"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model" / "best_autoencoder.pt"
AUDIO_OUT_DIR = PROJECT_ROOT / "data" / "commentary"


def get_reconstruction_error(npy_filename: str) -> float:
    seq = np.load(POSE_SEQUENCES_DIR / npy_filename)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 30, 34)

    model = LSTMAutoencoder(input_size=34, hidden_size=64, seq_len=30)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        recon, _ = model(x)
        error = reconstruction_error(x, recon).item()
    return error


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_commentary.py <pose_sequence_filename.npy>")
        return

    npy_filename = sys.argv[1]
    error = get_reconstruction_error(npy_filename)
    print(f"Reconstruction error for {npy_filename}: {error:.5f}")

    context = build_context(error)
    print("\nContext:")
    print(json.dumps(context, indent=2))

    print("\nGenerating commentary...")
    commentary_text = generate_commentary(context)
    print(f"Commentary: {commentary_text}")

    audio_filename = Path(npy_filename).stem + ".mp3"
    audio_path = AUDIO_OUT_DIR / audio_filename
    text_to_audio(commentary_text, audio_path)


if __name__ == "__main__":
    main()