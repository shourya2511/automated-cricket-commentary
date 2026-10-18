# CricketVision AI — Project Status and Improvement Plan

## 1. Executive summary

CricketVision AI is a prototype for generating spoken cricket-shot commentary from video. The repository contains scripts and assets for:

1. Manually marking delivery start and bat-ball contact frames.
2. Detecting a batsman in video frames with a custom YOLOv8 model.
3. Extracting body landmarks with MediaPipe Pose and saving fixed-length NumPy sequences.
4. Training an LSTM autoencoder on those sequences.
5. Mapping the autoencoder's reconstruction error to a heuristic confidence score.
6. Asking Gemini to generate short commentary and converting the text to an MP3 with gTTS.

The implementation is a collection of offline scripts, not a live or deployed application. The README mentions Streamlit, but there is no Streamlit app in the current source tree. Likewise, contact-frame annotation is manual; the current code does not automatically detect bat-ball contact.

The complete prototype path is:

```text
Raw clips
  -> manual start/contact annotation
  -> YOLO batsman detection
  -> MediaPipe pose landmarks
  -> 30-frame, 34-feature pose sequence
  -> LSTM autoencoder reconstruction error
  -> heuristic confidence/context
  -> Gemini commentary text
  -> gTTS MP3
```

## 2. What is present and implemented

### Video annotation and preprocessing

- `src/preprocessing/find_contact_frames.py` displays videos and lets an operator step through frames and mark each delivery's start and contact frames. The results are saved to `data/contact_frames.json`.
- `src/preprocessing/pose_extraction.py` reads the marked frames, detects the batsman with `models/yolo/best.pt`, extracts pose landmarks, fills failed detections from nearby frames, and saves one `.npy` file per delivery under `data/pose_sequences/`.
- `src/preprocessing/extract_frames.py` is a separate helper that samples every fifth frame and saves resized JPEGs under `data/frames/`. The pose-extraction pipeline reads source videos directly and does not consume those JPEGs.
- `src/preprocessing/debug_crops.py` writes annotated crops/frames for inspecting whether YOLO selected the batsman.
- `src/preprocessing/verify_pose_sequence.py` plots selected pose frames as skeletons for visual inspection.

### Model training and inference

- `src/lstm/dataset.py` loads pose `.npy` files as PyTorch tensors.
- `src/lstm/train.py` trains the LSTM autoencoder, records training/validation loss, saves the best checkpoint, and prints reconstruction-error statistics.
- `models/lstm_model/model.py` defines the encoder/decoder LSTM and per-sequence mean squared reconstruction error.
- `models/lstm_model/best_autoencoder.pt` is the current LSTM checkpoint.
- `models/yolo/best.pt` is the custom YOLO checkpoint used during pose extraction.
- `data/cricket.v1i.yolov8/` contains an exported object-detection dataset with two configured classes, `ball` and `batsman`. There is no YOLO training script in the current repository.

### Commentary generation

- `src/commentary/generate_commentary.py` loads one pose sequence and the LSTM checkpoint, calculates its reconstruction error, calls the context builder and Gemini generator, then writes an MP3.
- `src/commentary/context_builder.py` maps reconstruction error to a heuristic score and decides whether the context should name the shot as a cover drive.
- `src/commentary/gemini_commentary.py` loads `GEMINI_API_KEY` from the environment or project `.env`, requests one or two sentences from Gemini, and returns the response text.
- `src/commentary/text_to_speech.py` uses gTTS to save that text as an MP3.

### Other project files and assets

- `test_setup.py` is a dependency/import smoke check; it is not a functional or accuracy test.
- `test_yolo.py`, `bus.jpg`, and `yolov8n.pt` form a general YOLO demo using a bus image. They do not test the cricket model.
- `requirements.txt` lists pinned Python dependencies.
- `.gitignore` excludes the local virtual environment, caches, debug crops, editor settings, and `.env`.
- `my.gitignore` is an extra ignore-rules file; Git only automatically reads `.gitignore` (and certain configured excludes), not a file named `my.gitignore`.
- `data/pose_sequences/` contains 14 `.npy` sequences and visualizations. `data/commentary/` contains two MP3 examples.
- `data/debug_crops/` and `data/frames/` contain diagnostic/generated outputs.
- `data/raw_clips/cover_drives/` contains three clips referenced by the annotation and pose scripts.
- `data/cricket.v1i.yolov8.zip` is a compressed copy of the exported detection data.
- `cric_env/` is a local, ignored Python virtual environment, not source code.
- `app/`, `notebooks/`, and `data/annotations/` are currently empty.
- Two root-level MP4 files and `data/video_urls.txt` are source/reference media.

## 3. Main problems to fix for the next prototype

The immediate goal is a small offline prototype that predicts one of five shot types from a pose sequence. The goal is **not** to make the system live, production-ready, or automatically understand every part of a cricket match.

### 3.1 The current model does not learn shot names

The LSTM autoencoder learns to recreate a pose sequence. It is not given shot names during training, so its reconstruction error cannot reliably distinguish five shot types. The current context builder turns a heuristic score into a hardcoded `"cover_drive"` label.

**Prototype change:** keep the existing pose-extraction format, but train a supervised sequence classifier with one output class per selected shot type. The existing autoencoder is not required for this first five-shot classifier; it can be kept separately for later experimentation.

### 3.2 The examples need shot labels

Each delivery needs a human-checked shot name, in addition to its start and contact frames. For example:

```json
{
  "cover1.mp4": [
    {
      "start": 0,
      "contact": 25,
      "shot_type": "cover_drive"
    }
  ]
}
```

The person reviewing the video supplies the label. Pose extraction should carry that label into a small metadata file next to each saved `.npy` sequence (or a single CSV/JSON manifest). The dataset loader will use it as the correct answer during training. The five class names should be written down once and used consistently in annotations, training, and commentary.

### 3.3 There are too few examples to judge accuracy

The current collection has 14 pose sequences from three clips, all currently treated as cover-drive examples. This is not enough to train or fairly evaluate a five-class model.

**Prototype change:** collect reviewed examples for each of the five shot types. Aim for a similar number per type where practical. Vary the clips and players as much as the available material allows. Save some complete clips for evaluation and do not train on those clips.

### 3.4 Pose sequences need a quick quality check

Bad contact-frame annotations, a missed batsman detection, or many failed pose detections can give the classifier bad examples. Existing scripts can visualize detections and pose sequences, so this can be addressed without building a new tracking system.

**Prototype change:** inspect the generated pose plots for each training sequence and correct or exclude obviously bad sequences. Add a simple dataset check for the expected `(30, 34)` shape and finite numeric values. Keep the existing YOLO and MediaPipe steps unless review shows they are a consistent source of errors.

### 3.5 The prediction should include an honest “uncertain” outcome

A classifier will produce a score for each known shot type, but it may still be wrong, especially with a small dataset. It should not always force Gemini to announce one of the five shots.

**Prototype change:** initially use a simple confidence threshold selected by checking predictions on held-out clips. If the best class score is below that threshold, label the result as unknown/uncertain and ask Gemini for generic commentary. Do not call the score a calibrated probability.

## 4. Small, practical implementation plan

Complete these steps in order. They are suitable for adding to the current script-based prototype:

1. **Choose and record the five class names.** Use stable machine-friendly names such as `cover_drive`; do not vary spelling or capitalization between annotations.
2. **Add `shot_type` to each delivery annotation.** Manually review the clip and assign the best label. If the shot is genuinely unclear, mark it for review rather than guessing.
3. **Connect labels to extracted sequences.** Update `pose_extraction.py` to save a small metadata manifest that maps each `.npy` filename to its source clip, delivery number, and `shot_type`.
4. **Validate the data.** Update `dataset.py` to load both the pose and its label, check that the pose has shape `(30, 34)`, and report missing labels or invalid files clearly.
5. **Train a five-class classifier.** Update/add a model and training script that use the labels and a standard classification loss. The output layer must have five classes. The autoencoder checkpoint is not interchangeable with this classifier checkpoint.
6. **Evaluate on unseen clips.** Make a simple train/validation split by source clip, not by randomly mixing sequences from the same clip. Report total accuracy and a per-class count of correct and incorrect predictions; a confusion matrix can be added if convenient.
7. **Pass prediction into commentary.** Change `generate_commentary.py` and `context_builder.py` to use the classifier's predicted class and score instead of converting autoencoder reconstruction error to `"cover_drive"`.
8. **Keep low-confidence commentary generic.** Update the Gemini prompt/context to support all five names and to avoid naming a shot when the classifier is uncertain.
9. **Run a small end-to-end check.** Use one known sequence from each class and one sequence from a held-out clip. Check that prediction, text generation, and MP3 creation either succeed or report a clear error.

The parts that can stay the same initially are manual frame marking, YOLO batsman detection, MediaPipe landmarks, the `(30, 34)` pose format, Gemini text generation, and gTTS. Improve those components only when reviewing examples shows a specific problem.

## 5. Future problems to watch for

| Possible problem | Simple way to reduce the risk in this prototype |
|---|---|
| One shot type has far fewer examples than others. | Count labels before training; collect more for the smallest class and show per-class results rather than only total accuracy. |
| Training results look good because similar examples are in both splits. | Keep whole source clips in either training or evaluation, never split sequences from the same clip across both. |
| Wrong or inconsistent human labels teach the wrong answer. | Use one written definition/example for each class and review uncertain deliveries before using them for training. |
| Poor pose extraction is mistaken for a real shot pattern. | Use the existing visualization tools to inspect training sequences; reject clearly broken sequences and check array shape/finite values. |
| The model confidently predicts a wrong class for an unfamiliar clip. | Add a simple low-confidence/uncertain path and test the threshold on held-out clips; do not present the score as a probability. |
| New training data is added but an old model is still used. | Save the classifier weights to a clearly named file and record which dataset/classes were used; retrain after changing labels or pose data. |
| Commentary generation fails because the key, network, or speech service is unavailable. | Check for `GEMINI_API_KEY`, save/print the generated text before TTS, and report API/TTS errors instead of implying an MP3 was created. |
| Dependencies or media files are difficult to set up on another computer. | Keep the virtual environment out of Git, retain the dependency list, and write exact setup/run commands in the README. |
| README promises features not present in code. | Describe the five-shot classifier and any automatic/live features as planned until they actually exist and have been tested. |

## 6. Current run requirements and testing gap

Run scripts from the project root. The existing later-stage commands are:

```powershell
python src\lstm\train.py
python src\commentary\generate_commentary.py pose_seq_cover1_01.npy
```

After adding the classifier, the training command and inference command will need to be updated to use the classifier and its checkpoint. The current commentary command requires a pose sequence, the existing autoencoder checkpoint, a valid `GEMINI_API_KEY`, and network access for Gemini and gTTS.

`test_setup.py` checks whether packages import, while `test_yolo.py` runs a general YOLO demo on a bus image. Neither checks five-shot accuracy. For the next prototype, the most useful tests are modest: annotation labels are valid, pose arrays have the expected shape, the dataset can load all examples, and the classifier can train and predict all configured classes.

## 7. Optional repository cleanup

These items are not blockers for shot classification:

- Remove empty `app/`, `notebooks/`, and `data/annotations/` folders if they are not being used as placeholders.
- Remove or merge `my.gitignore`; `.gitignore` is the standard active ignore filename.
- Remove `data/debug_crops/` or `data/frames/` only if those debugging/generated files are no longer useful.
- Keep either the YOLO dataset ZIP or its extracted copy if duplicate storage is unnecessary.
- Keep `bus.jpg` and `yolov8n.pt` only if the general YOLO smoke test is useful.
- Keep `cric_env/` as a local environment or recreate it from dependencies; it should not be committed.

Do not delete raw clips, contact annotations, pose sequences, or checkpoints that are needed for the current pipeline or for collecting/reviewing training examples.

## 8. Bottom line

The next useful prototype is a **small supervised five-shot classifier** using reviewed examples. The immediate work is to add shot labels to deliveries, preserve those labels beside the extracted pose sequences, train a classifier that uses them, and evaluate it on clips it did not train on. Keep the existing video-to-pose and commentary components for now. The autoencoder can be set aside for shot recognition; it is not necessary for the first working five-shot prototype.
