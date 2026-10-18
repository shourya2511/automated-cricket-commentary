"""
train.py

Trains the LSTM autoencoder on your cover-drive pose sequences.
Run this on Google Colab (per your project's training platform) or
locally - dataset is tiny (14 sequences), so it'll run fine on CPU too,
just slower per epoch than GPU (though with this little data, speed
isn't really the bottleneck either way).

Usage (from project root):
    python src/lstm/train.py

Or on Colab: upload data/pose_sequences/*.npy, then run the equivalent
cells (copy this script's logic into a notebook, or just !python it
if you've cloned the repo into Colab).

Outputs:
    models/lstm_model/best_autoencoder.pt   - best model weights (lowest val loss)
    models/lstm_model/training_curve.png    - loss over epochs
    Prints reconstruction error stats at the end - use this to pick a
    sensible confidence threshold later.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PoseSequenceDataset
from models.lstm_model.model import LSTMAutoencoder, reconstruction_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_SEQUENCES_DIR = PROJECT_ROOT / "data" / "pose_sequences"
MODEL_OUT_DIR = PROJECT_ROOT / "models" / "lstm_model"

EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
VAL_FRACTION = 0.2  # with only 14 samples this is ~2-3 held out - small,
                     # but still useful as a sanity check against overfitting
SEED = 42


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = PoseSequenceDataset(POSE_SEQUENCES_DIR)
    print(dataset.summary())

    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train: {n_train} sequences | Val: {n_val} sequences")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAutoencoder(input_size=34, hidden_size=64, seq_len=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0
        for x, _ in train_loader:  # labels unused - autoencoder is unsupervised
            x = x.to(device)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x.size(0)
        epoch_train_loss /= n_train

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                loss = criterion(recon, x)
                epoch_val_loss += loss.item() * x.size(0)
        epoch_val_loss /= n_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_OUT_DIR / "best_autoencoder.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={epoch_train_loss:.5f} | val_loss={epoch_val_loss:.5f}")

    print(f"\nBest val loss: {best_val_loss:.5f}")
    print(f"Best model saved to {MODEL_OUT_DIR / 'best_autoencoder.pt'}")

    # plot training curve
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE reconstruction loss")
    plt.title("LSTM Autoencoder Training")
    plt.legend()
    plt.tight_layout()
    curve_path = MODEL_OUT_DIR / "training_curve.png"
    plt.savefig(curve_path, dpi=100)
    print(f"Training curve saved to {curve_path}")

    # reconstruction error stats on the FULL dataset - useful for picking
    # a confidence threshold later (e.g. "error > X -> low confidence /
    # unknown shot", matching your project's 60% confidence-threshold idea)
    model.load_state_dict(torch.load(MODEL_OUT_DIR / "best_autoencoder.pt"))
    model.eval()
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for x, _ in full_loader:
            x = x.to(device)
            recon, _ = model(x)
            errors = reconstruction_error(x, recon).cpu().numpy()

    print("\nPer-sample reconstruction error (all sequences, best model):")
    for f, err in zip(dataset.files, errors):
        print(f"  {f.name}: {err:.5f}")
    print(f"\nMean: {errors.mean():.5f} | Std: {errors.std():.5f} | "
          f"Min: {errors.min():.5f} | Max: {errors.max():.5f}")


if __name__ == "__main__":
    main()