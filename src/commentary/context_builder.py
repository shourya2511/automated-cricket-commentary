"""
context_builder.py

Converts the LSTM autoencoder's reconstruction error into a confidence
score, then assembles a structured JSON "context" describing the shot -
this is what gets handed to Gemini to generate commentary from.

CONFIDENCE MAPPING:
Lower reconstruction error = more "typical" cover-drive motion = higher
confidence. We map error to a 0-1 confidence score using the training
set's error statistics (mean, std) as a reference scale, then clip to
[0, 1]. This is a simple, explainable mapping appropriate for a
prototype - not a calibrated probability, just a monotonic confidence
proxy.

Per the project's original design: if confidence < 60%, the shot type
should NOT be named in the generated commentary (event-based commentary
only), to avoid confidently mislabeling an unfamiliar motion pattern.
"""

# Training error statistics (computed from best_autoencoder.pt on full dataset)
# These are used to normalize reconstruction errors into confidence scores.
# To update these: run train.py and copy the printed Mean/Std values here.
TRAIN_ERROR_MEAN = 0.01000
TRAIN_ERROR_STD = 0.00500

CONFIDENCE_THRESHOLD = 0.60  # below this, shot type is withheld


def error_to_confidence(error: float) -> float:
    """Maps reconstruction error to a 0-1 confidence score.
    error == mean -> ~0.5 confidence
    error well below mean -> confidence approaches 1.0
    error well above mean -> confidence approaches 0.0
    """
    z = (error - TRAIN_ERROR_MEAN) / (TRAIN_ERROR_STD + 1e-8)
    # sigmoid-like squashing, flipped so lower error = higher confidence
    confidence = 1 / (1 + pow(2.71828, z))
    return max(0.0, min(1.0, confidence))


def build_context(reconstruction_error: float) -> dict:
    """Builds the JSON context dict to send to the commentary generator.
    Deliberately does NOT include player names, team names, or match score -
    commentary is generated purely from the shot event itself."""
    confidence = error_to_confidence(reconstruction_error)
    shot_known = confidence >= CONFIDENCE_THRESHOLD

    context = {
        "shot_type": "cover_drive" if shot_known else None,
        "shot_confidence": round(confidence, 3),
        "shot_type_known": shot_known,
        "reconstruction_error": round(reconstruction_error, 5),
    }
    return context


if __name__ == "__main__":
    # quick sanity check using a few example errors
    for err in [0.006, 0.015, 0.030, 0.045]:
        ctx = build_context(err)
        print(f"error={err:.4f} -> confidence={ctx['shot_confidence']:.3f}  "
              f"shot_known={ctx['shot_type_known']}")