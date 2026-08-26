import cv2
from pathlib import Path

# configuration

RAW_CLIPS_DIR = Path("data/raw_clips/cover_drives")
OUTPUT_DIR = Path("data/frames/cover_drive")

FRAME_SKIP = 5
RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224



def extract_frames(video_path : Path):

    save_folder = OUTPUT_DIR/video_path.stem
    save_folder.mkdir(parents=True,exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    frame_number = 0
    saved_frames = 0

    while True:
        success,frame = cap.read()

        if not success:
            break
        
        if frame_number % FRAME_SKIP == 0:

            frame = cv2.resize(frame,(RESIZE_WIDTH,RESIZE_HEIGHT))
            frame_name = save_folder/f"frame_{saved_frames:04d}.jpg"
            cv2.imwrite(str(frame_name),frame)
            saved_frames+=1
        
        frame_number+=1

    cap.release()

    print(f"{video_path.name}")
    print(f"Total Frames : {frame_number}")
    print(f"Saved Frames : {saved_frames}")
    print(f"-"*40)

def main():
    videos = sorted(RAW_CLIPS_DIR.glob("*.mp4"))

    if len(videos)==0:
        print("No videos found!")
        return
    
    print(f"\nFound {len(videos)} videos\n")

    for video in videos:
        extract_frames(video)

    print("\nFrame Extraction Complete\n")

if __name__ == "__main__":
    main()

