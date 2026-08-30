from ultralytics import YOLO

model = YOLO("yolov8l")

results = model.predict("input_videos/video_1.mp4", save=True)

print(results)
print("==========================")

for box in results[0].boxes:
    print(box)
