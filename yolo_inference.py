#Import All the Required Libraries
from ultralytics import YOLO

#Load the YOLOv12 Model
model = YOLO("yolo12n.pt")

#Predictions
#results = model.predict(source="input_videos/input_video.mp4", save=True)
results = model.track(source="input_videos/input_video.mp4", save=True, persist=True)