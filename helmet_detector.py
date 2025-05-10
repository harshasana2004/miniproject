from ultralytics import YOLO
import cv2
import numpy as np

class HelmetDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = ["helmet", "no_helmet"]

    def detect(self, frame):
        results = self.model(frame)
        return results

    def draw_results(self, frame, results):
        annotated_frame = frame.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                label = f"{self.class_names[class_id]} {confidence:.2f}"
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for helmet, red for no_helmet
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated_frame