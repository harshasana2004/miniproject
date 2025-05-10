from ultralytics import YOLO
import torch.multiprocessing

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)  # Ensure correct multiprocessing method

    model = YOLO("yolov8m.pt")  # Load pre-trained YOLOv8m

    # Train with your dataset
    model.train(
        data="C:\\Users\\harsh\\PycharmProjects\\UpdatedProject\\config.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/train",
        name="helmet_yolov8m",
        exist_ok=True
    )
