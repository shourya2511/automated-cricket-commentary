+  -> Gemini commentary
+  -> gTTS MP3
+```
+
+## 2. What is present and implemented
+
+### Video annotation and preprocessing
+
+- `src/preprocessing/find_contact_frames.py` displays videos and lets an operator step through frames and mark each delivery's start and contact frames. The results are saved to `data/contact_frames.json`.
+- `src/preprocessing/pose_extraction.py` reads the marked frames, detects the batsman with `models/yolo/best.pt`, extracts pose landmarks, fills failed detections from nearby frames, and saves one `.npy` file per delivery under `data/pose_sequences/`.
+- `src/preprocessing/extract_frames.py` is a separate helper that samples every fifth frame and saves resized JPEGs under `data/frames/`. The pose-extraction pipeline reads source videos directly and does not consume those JPEGs.
+- `src/preprocessing/debug_crops.py` writes annotated crops/frames for inspecting whether YOLO selected the batsman.
+- `src/preprocessing/verify_pose_sequence.py` plots selected pose frames as skeletons for visual inspection.
+
+### Model training and inference
+
+- `src/lstm/dataset.py` loads pose `.npy` files as PyTorch tensors.
+- `src/lstm/train.py` trains the LSTM autoencoder, records training/validation loss, saves the best checkpoint, and prints reconstruction-error statistics.
+- `models/lstm_model/model.py` defines the encoder/decoder LSTM and per-sequence mean squared reconstruction error.
+- `models/lstm_model/best_autoencoder.pt` is the current LSTM checkpoint.
+- `models/yolo/best.pt` is the custom YOLO checkpoint used during pose extraction.
+- `data/cricket.v1i.yolov8/` contains an exported object-detection dataset with two configured classes, `ball` and `batsman`. There is no YOLO training script in the current repositor
:.
+
+- `src/preprocessing/find_contact_frames.py` displays videos and lets an operator step through frames and mark each delivery's start and contact frames. The results are saved to `data/contact_frames.json`.
+- `src/preprocessing/pose_extraction.py` reads the marked frames, detects the batsman with `models/yolo/best.pt`, extracts pose landmarks, fills failed detections from nearby frames, and saves one `.npy` file per delivery under `data/pose_sequences/`.
+- `src/preprocessing/extract_frames.py` is a separate helper that samples every fifth frame and saves resized JPEGs under `data/frames/`. The pose-extraction pipeline reads source videos directly and does not consume those JPEGs.
+- `src/preprocessing/debug_crops.py` writes annotated crops/frames for inspecting whether YOLO selected the batsman.
+- `src/preprocessing/verify_pose_sequence.py` plots selected pose frames as skeletons for visual inspection.
+
+### Model training and inference
+
+- `src/lstm/dataset.py` loads pose `.npy` files as PyTorch tensors.
+- `src/lstm/train.py` trains the LSTM autoencoder, records training/validation loss, saves the best checkpoint, and prints reconstruction-error statistics.
+- `models/lstm_model/model.py` defines the encoder/decoder LSTM and per-sequence mean squared reconstruction error.
+- `models/lstm_model/best_autoencoder.pt` is the current LSTM checkpoint.
+- `models/yolo/best.pt` is the custom YOLO checkpoint used during pose extraction.
+- `data/cricket.v1i.yolov8/` contains an exported object-detection dataset with two configured classes, `ball` and `batsman`. There is no YOLO training script in the current repository.
:
+
+- `src/preprocessing/find_contact_frames.py` displays videos and lets an operator step through frames and mark each delivery's start and contact frames. The results are saved to `data/contact_frames.json`.
+- `src/preprocessing/pose_extraction.py` reads the marked frames, detects the batsman with `models/yolo/best.pt`, extracts pose landmarks, fills failed detections from nearby frames, and saves one `.npy` file per delivery under `data/pose_sequences/`.
+- `src/preprocessing/extract_frames.py` is a separate helper that samples every fifth frame and saves resized JPEGs under `data/frames/`. The pose-extraction pipeline reads source videos directly and does not consume those JPEGs.
+- `src/preprocessing/debug_crops.py` writes annotated crops/frames for inspecting whether YOLO selected the batsman.
+- `src/preprocessing/verify_pose_sequence.py` plots selected pose frames as skeletons for visual inspection.
+
+### Model training and inference
+
+- `src/lstm/dataset.py` loads pose `.npy` files as PyTorch tensors.
+- `src/lstm/train.py` trains the LSTM autoencoder, records training/validation loss, saves the best checkpoint, and prints reconstruction-error statistics.
+- `models/lstm_model/model.py` defines the encoder/decoder LSTM and per-sequence mean squared reconstruction error.
+- `models/lstm_model/best_autoencoder.pt` is the current LSTM checkpoint.
+- `models/yolo/best.pt` is the custom YOLO checkpoint used during pose extraction.
+- `data/cricket.v1i.yolov8/` contains an exported object-detection dataset with two configured classes, `ball` and `batsman`. There is no YOLO training script in the current repository.
+
:
+
+- `src/preprocessing/find_contact_frames.py` displays videos and lets an operator step through frames and mark each delivery's start and contact frames. The results are saved to `data/contact_frames.json`.
+- `src/preprocessing/pose_extraction.py` reads the marked frames, detects the batsman with `models/yolo/best.pt`, extracts pose landmarks, fills failed detections from nearby frames, and saves one `.npy` file per delivery under `data/pose_sequences/`.
+- `src/preprocessing/extract_frames.py` is a separate helper that samples every fifth frame and saves resized JPEGs under `data/frames/`. The pose-extraction pipeline reads source videos directly and does not consume those JPEGs.
+- `src/preprocessing/debug_crops.py` writes annotated crops/frames for inspecting whether YOLO selected the batsman.
+- `src/preprocessing/verify_pose_sequence.py` plots selected pose frames as skeletons for visual inspection.
+
+### Model training and inference
+
+- `src/lstm/dataset.py` loads pose `.npy` files as PyTorch tensors.
+- `src/lstm/train.py` trains the LSTM autoencoder, records training/validation loss, saves the best checkpoint, and prints reconstruction-error statistics.
+- `models/lstm_model/model.py` defines the encoder/decoder LSTM and per-sequence mean squared reconstruction error.
+- `models/lstm_model/best_autoencoder.pt` is the current LSTM checkpoint.
+- `models/yolo/best.pt` is the custom YOLO checkpoint used during pose extraction.
+- `data/cricket.v1i.yolov8/` contains an exported object-detection dataset with two configured classes, `ball` and `batsman`. There is no YOLO training script in the current repository.
+
+### Commentary generation
:
