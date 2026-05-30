from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} objects")
    for box in boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        print(f"  → {name} ({conf:.0%} confidence)")

print("\n✓ YOLOv8 is working!")