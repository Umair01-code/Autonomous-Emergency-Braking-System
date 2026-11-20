# Autonomous Emergency Braking System

Real-time vehicle perception and collision-risk analysis using object detection, tracking, and Time-to-Collision (TTC) estimation.

## Overview

This project demonstrates an Autonomous Emergency Braking (AEB) pipeline for video-based driving scenarios. It:

- Detects vehicles in video using **YOLO11n**.
- Tracks detected vehicles across frames using **ByteTrack**.
- Estimates distance and Time-to-Collision (TTC).
- Classifies risk as `SAFE`, `WARNING`, or `AEB`.
- Produces annotated videos and CSV data for analysis.
- Provides a PySide6 desktop dashboard for reviewing the processed video and safety metrics.

> **Note:** This is a research and demonstration project. It is not intended for deployment in a real vehicle without extensive validation, calibration, and safety certification.

## How it works

The system calculates TTC using the relative speed between the ego vehicle and a target vehicle:

```text
TTC = distance / (ego speed - target speed)
```

The default risk thresholds are:

| TTC | Risk level |
| --- | --- |
| Greater than 3.0 seconds | SAFE |
| 1.5 to 3.0 seconds | WARNING |
| 1.5 seconds or less | AEB |

If the relative speed is zero or negative, TTC is treated as infinite because the target is not approaching.

## Project structure

```text
.
├── aeb/                    # TTC and risk-assessment logic
├── app/                    # Application resources
├── data/                   # CSV results
├── experiments/            # Experiment files
├── input/                  # Input videos
├── models/                 # YOLO model weights
├── notebooks/              # Analysis notebooks
├── perception/             # Detection, tracking, and video processing
├── simulation/             # Simulation components
├── tests/                  # Test files
├── main.py                 # PySide6 visualization dashboard
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Autonomous-Emergency-Braking-System.git
   cd Autonomous-Emergency-Braking-System
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place an input video at:

```text
input/test_video.mp4
```

### 1. Run object detection

```bash
python perception/detection.py
```

### 2. Run vehicle tracking

```bash
python perception/tracking.py
```

The tracking output is saved to:

```text
runs/track/aeb_tracking_clean.mp4
```

### 3. Generate the AEB video

After the required CSV file has been generated, run:

```bash
python perception/aeb_video.py
```

The annotated TTC/AEB video is saved to:

```text
runs/track/aeb_ttc_output.mp4
```

### 4. Launch the dashboard

```bash
python main.py
```

The dashboard displays the processed video, distance and TTC plots, and the current AEB decision.

## Core modules

- `aeb/ttc.py` — TTC calculation.
- `aeb/risk.py` — risk classification based on TTC.
- `perception/detection.py` — YOLO object detection.
- `perception/tracking.py` — vehicle tracking with ByteTrack.
- `perception/aeb_video.py` — annotated video generation.
- `main.py` — desktop visualization dashboard.

## Outputs

Typical generated outputs include:

- Annotated detection and tracking videos.
- AEB/TTC visualization videos.
- CSV files containing frame-level detection, tracking, distance, TTC, and risk data.

Generated videos and temporary run files are excluded from Git by `.gitignore`.

## Testing

Run the available tests with:

```bash
python -m pytest
```

## License

No license has been specified yet. Add a license before distributing or reusing this project publicly.
