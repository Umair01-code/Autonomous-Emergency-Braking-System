
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# Project paths
MODEL_PATH = "models/yolo11n.pt"
INPUT_VIDEO = "input/test_video.mp4"
CSV_PATH = "data/AEB_integrated_results.csv"
OUTPUT_VIDEO = "runs/track/aeb_ttc_output.mp4"

# Appearance (OpenCV uses BGR)
LIGHT_GREEN = (144, 238, 144)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)


def main():
    for path in (MODEL_PATH, INPUT_VIDEO, CSV_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(CSV_PATH)

    required = {
        "frame", "track_id", "x1", "y1", "x2", "y2",
        "ttc_used", "aeb_decision"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # One record per frame and track ID
    df = df.drop_duplicates(
        subset=["frame", "track_id"], keep="last"
    )
    records = {
        (int(row.frame), int(row.track_id)): row
        for row in df.itertuples(index=False)
    }

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(OUTPUT_VIDEO).parent.mkdir(
        parents=True, exist_ok=True
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_number = 0

    try:
        # Re-run tracking in the same order as the original video.
        for result in model.track(
            source=INPUT_VIDEO,
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            conf=0.25,
            classes=[2, 3, 5, 7],
            verbose=False
        ):
            frame = result.orig_img.copy()

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int)

                for box, class_id, track_id in zip(
                    xyxy, class_ids, ids
                ):
                    x1, y1, x2, y2 = box
                    key = (frame_number, int(track_id))
                    row = records.get(key)

                    color = LIGHT_GREEN
                    label = f"{model.names[class_id]} | ID {track_id}"

                    if row is not None:
                        decision = str(row.aeb_decision)
                        ttc = float(row.ttc_used)

                        if decision == "AEB":
                            color = RED
                        elif decision == "WARNING":
                            color = ORANGE

                        if np.isfinite(ttc):
                            label += f" | TTC {ttc:.2f}s"
                        else:
                            label += " | TTC INF"

                        label += f" | {decision}"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        color, 2
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.48
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(
                        label, font, scale, thickness
                    )

                    label_y = max(th + 8, y1 - 6)
                    cv2.rectangle(
                        frame,
                        (x1, label_y - th - 5),
                        (min(width - 1, x1 + tw + 8), label_y + 3),
                        DARK, -1
                    )
                    cv2.putText(
                        frame, label, (x1 + 4, label_y),
                        font, scale, WHITE, thickness,
                        cv2.LINE_AA
                    )

            # Fixed status panel; decisions here are dataset summaries,
            # not necessarily the current ego vehicle's live decision.
            cv2.rectangle(frame, (15, 15), (420, 58), DARK, -1)
            cv2.putText(
                frame, "AEB PROTOTYPE | YOLO11n + ByteTrack",
                (25, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, WHITE, 1, cv2.LINE_AA
            )

            writer.write(frame)
            frame_number += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames")

    finally:
        cap.release()
        writer.release()

    print(f"\nOutput video saved: {Path(OUTPUT_VIDEO).resolve()}")
    print(f"Frames processed: {frame_number}")


if __name__ == "__main__":
    main()
