from ultralytics import YOLO


MODEL_PATH = "models/yolo11n.pt"


def detect_objects(video_path):
    """
    Run YOLO11n object detection on a video.
    """

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=video_path,
        save=True,
        conf=0.25
    )

    return results


if __name__ == "__main__":
    video_path = "input/test_video.mp4"

    detect_objects(video_path)

    print("YOLO detection completed successfully.")