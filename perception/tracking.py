import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/yolo11n.pt"
INPUT_VIDEO = "input/test_video.mp4"

OUTPUT_VIDEO = "runs/track/aeb_tracking_clean.mp4"


# Light green in OpenCV BGR format
LIGHT_GREEN = (144, 238, 144)

# White text
WHITE = (255, 255, 255)

# Dark background for labels
DARK = (30, 30, 30)


# ============================================================
# VEHICLE TRACKING
# ============================================================

def track_vehicles(video_path):

    model = YOLO(MODEL_PATH)

    # Open original video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}")
    print(f"FPS: {fps}")

    # Create output directory
    import os
    os.makedirs("runs/track", exist_ok=True)

    # MP4 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # YOLO + ByteTrack
    # --------------------------------------------------------

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=0.25,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    frame_number = 0

    for result in results:

        frame = result.orig_img.copy()

        frame_number += 1

        # ----------------------------------------------------
        # DRAW VEHICLES
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()

            class_ids = boxes.cls.cpu().numpy().astype(int)

            confidences = boxes.conf.cpu().numpy()

            # Tracking IDs
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(xyxy)

            for box, class_id, confidence, track_id in zip(
                xyxy,
                class_ids,
                confidences,
                track_ids
            ):

                x1, y1, x2, y2 = map(int, box)

                # ------------------------------------------------
                # Thin light-green bounding box
                # ------------------------------------------------

                thickness = max(2, int(width / 2000))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    LIGHT_GREEN,
                    thickness
                )

                # ------------------------------------------------
                # Vehicle name
                # ------------------------------------------------

                class_name = model.names[class_id].upper()

                label = f"{class_name}  |  ID {track_id}"

                # ------------------------------------------------
                # Small clean label
                # ------------------------------------------------

                font = cv2.FONT_HERSHEY_SIMPLEX

                font_scale = max(0.45, width / 8500)

                text_thickness = 1

                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    text_thickness
                )

                # Label position above bounding box
                label_x = x1

                label_y = y1 - 8

                # If there isn't enough room above the box,
                # place the label inside the box.
                if label_y - text_height < 0:
                    label_y = y1 + text_height + 8

                # Background rectangle
                cv2.rectangle(
                    frame,
                    (
                        label_x,
                        label_y - text_height - 5
                    ),
                    (
                        label_x + text_width + 8,
                        label_y + 3
                    ),
                    DARK,
                    -1
                )

                # Label text
                cv2.putText(
                    frame,
                    label,
                    (
                        label_x + 4,
                        label_y
                    ),
                    font,
                    font_scale,
                    WHITE,
                    text_thickness,
                    cv2.LINE_AA
                )

        # ====================================================
        # CLEAN CORNER INFORMATION
        # ====================================================

        # Top-left title
        title = "AEB SYSTEM  |  VEHICLE TRACKING"

        cv2.rectangle(
            frame,
            (20, 20),
            (470, 65),
            DARK,
            -1
        )

        cv2.putText(
            frame,
            title,
            (32, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            1,
            cv2.LINE_AA
        )

        # Top-right technology information
        info = "YOLO11n  +  ByteTrack"

        (info_width, info_height), _ = cv2.getTextSize(
            info,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        cv2.rectangle(
            frame,
            (
                width - info_width - 35,
                20
            ),
            (
                width - 20,
                58
            ),
            DARK,
            -1
        )

        cv2.putText(
            frame,
            info,
            (
                width - info_width - 27,
                47
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            LIGHT_GREEN,
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Frame counter
        # ----------------------------------------------------

        frame_text = f"FRAME  {frame_number}"

        cv2.putText(
            frame,
            frame_text,
            (25, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            1,
            cv2.LINE_AA
        )

        # Write frame
        writer.write(frame)

        # Progress
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")

    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    cap.release()
    writer.release()

    print()
    print("==========================================")
    print("CLEAN TRACKING COMPLETED")
    print("==========================================")
    print(f"Frames processed: {frame_number}")
    print(f"Output: {OUTPUT_VIDEO}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    track_vehicles(INPUT_VIDEO)