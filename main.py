
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QMessageBox,
    QSlider, QProgressBar
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent

DEFAULT_VIDEO = ROOT / "runs" / "track" / "aeb_ttc_output.mp4"
DEFAULT_CSV = ROOT / "data" / "AEB_integrated_results.csv"


class MetricCard(QFrame):
    def __init__(self, title, value="--", color="#90EE90"):
        super().__init__()
        self.setObjectName("MetricCard")
        self.setMinimumHeight(105)

        layout = QVBoxLayout(self)

        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        self.value_label.setStyleSheet(f"color: {color};")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()


class AEBWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Autonomous Emergency Braking | AEB")
        self.resize(1400, 900)

        self.capture = None
        self.video_path = None
        self.csv_data = pd.DataFrame()
        self.current_frame = 0
        self.total_frames = 0
        self.playing = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.build_ui()
        self.load_default_data()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(14)

        header = QHBoxLayout()

        title = QLabel("AUTONOMOUS EMERGENCY BRAKING")
        title.setObjectName("HeaderTitle")

        subtitle = QLabel("Vehicle Perception • TTC • Collision Risk")
        subtitle.setObjectName("Subtitle")

        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header.addLayout(title_layout)
        header.addStretch()

        self.system_status = QLabel("SYSTEM READY")
        self.system_status.setObjectName("ReadyStatus")
        header.addWidget(self.system_status)

        root_layout.addLayout(header)

        metrics = QGridLayout()

        self.distance_card = MetricCard("TARGET DISTANCE", "-- m")
        self.speed_card = MetricCard("RELATIVE SPEED", "-- m/s")
        self.ttc_card = MetricCard("TIME TO COLLISION", "-- s")
        self.risk_card = MetricCard("AEB DECISION", "READY", "#90EE90")

        for index, card in enumerate([
            self.distance_card,
            self.speed_card,
            self.ttc_card,
            self.risk_card
        ]):
            metrics.addWidget(card, 0, index)

        root_layout.addLayout(metrics)

        content = QHBoxLayout()

        # Video area
        video_panel = QVBoxLayout()

        video_title = QLabel("PERCEPTION VIDEO")
        video_title.setObjectName("SectionTitle")
        video_panel.addWidget(video_title)

        self.video_label = QLabel("Load an annotated AEB video to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet(
            "background: #080D12; color: #B7C5D0; border-radius: 8px;"
        )

        video_panel.addWidget(self.video_label, 1)

        controls = QHBoxLayout()

        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self.choose_video)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)

        controls.addWidget(self.load_video_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.stop_button)

        video_panel.addLayout(controls)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderReleased.connect(self.seek_video)
        video_panel.addWidget(self.seek_slider)

        # Chart area
        chart_panel = QVBoxLayout()

        chart_title = QLabel("COLLISION RISK MONITOR")
        chart_title.setObjectName("SectionTitle")
        chart_panel.addWidget(chart_title)

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        chart_panel.addWidget(self.canvas, 1)

        content.addLayout(video_panel, 3)
        content.addLayout(chart_panel, 2)

        root_layout.addLayout(content, 1)

        footer = QHBoxLayout()

        self.frame_status = QLabel("Frame: 0 / 0")
        self.event_status = QLabel("Waiting for video")

        footer.addWidget(self.frame_status)
        footer.addStretch()
        footer.addWidget(self.event_status)

        root_layout.addLayout(footer)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #101820;
                color: #EAF0F5;
                font-family: Segoe UI;
                font-size: 13px;
            }

            QLabel#HeaderTitle {
                font-size: 23px;
                font-weight: bold;
                color: #FFFFFF;
            }

            QLabel#Subtitle {
                color: #9AAAB8;
                font-size: 12px;
            }

            QLabel#SectionTitle {
                font-size: 15px;
                font-weight: bold;
                color: #90EE90;
                padding: 5px;
            }

            QFrame#MetricCard {
                background-color: #1A2732;
                border: 1px solid #30424F;
                border-radius: 10px;
                padding: 8px;
            }

            QLabel#MetricTitle {
                color: #A8B8C5;
                font-size: 11px;
            }

            QLabel#MetricValue {
                font-size: 23px;
                font-weight: bold;
            }

            QLabel#ReadyStatus {
                color: #90EE90;
                font-weight: bold;
                padding: 10px;
            }

            QPushButton {
                background-color: #263946;
                color: white;
                border: 1px solid #405664;
                border-radius: 6px;
                padding: 9px 15px;
            }

            QPushButton:hover {
                background-color: #365363;
            }

            QSlider::groove:horizontal {
                height: 5px;
                background: #344653;
            }

            QSlider::handle:horizontal {
                background: #90EE90;
                width: 13px;
                margin: -5px 0;
                border-radius: 6px;
            }
        """)

    def load_default_data(self):
        if DEFAULT_CSV.exists():
            try:
                self.csv_data = pd.read_csv(DEFAULT_CSV)
                self.event_status.setText(
                    f"Loaded {len(self.csv_data)} CSV records"
                )
                self.update_chart()
            except Exception as exc:
                QMessageBox.warning(
                    self, "CSV Error", str(exc)
                )

        if DEFAULT_VIDEO.exists():
            self.open_video(str(DEFAULT_VIDEO))

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select annotated AEB video",
            str(ROOT),
            "Video files (*.mp4 *.avi *.mov)"
        )

        if path:
            self.open_video(path)

    def open_video(self, path):
        self.timer.stop()

        if self.capture:
            self.capture.release()

        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            self.capture = None
            QMessageBox.critical(
                self, "Video Error",
                f"Could not open video:\n{path}"
            )
            return

        self.video_path = path
        self.total_frames = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.current_frame = 0

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.timer.setInterval(
            max(1, int(1000 / fps)) if fps > 0 else 33
        )

        self.play_button.setText("Play")
        self.playing = False

        self.next_frame()
        self.system_status.setText("VIDEO LOADED")
        self.event_status.setText(Path(path).name)

    def toggle_playback(self):
        if self.capture is None:
            self.choose_video()
            return

        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        else:
            if self.current_frame >= self.total_frames:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0

            self.timer.start()
            self.playing = True
            self.play_button.setText("Pause")

    def stop_video(self):
        self.timer.stop()
        self.playing = False
        self.play_button.setText("Play")

        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.next_frame()

    def seek_video(self):
        if self.capture is None or self.total_frames <= 0:
            return

        position = int(
            self.seek_slider.value() / 1000 * self.total_frames
        )

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.current_frame = position
        self.next_frame()

    def next_frame(self):
        if self.capture is None:
            return

        success, frame = self.capture.read()

        if not success:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
            return

        self.current_frame = int(
            self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        )

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(
            int(self.current_frame / max(1, self.total_frames) * 1000)
        )
        self.seek_slider.blockSignals(False)

        self.frame_status.setText(
            f"Frame: {self.current_frame} / {self.total_frames}"
        )

        self.update_metrics_for_frame(self.current_frame)

    def update_metrics_for_frame(self, frame_number):
        if self.csv_data.empty or "frame" not in self.csv_data.columns:
            return

        rows = self.csv_data[
            self.csv_data["frame"] == frame_number
        ]

        if rows.empty:
            return

        row = rows.iloc[0]

        distance = row.get(
            "distance_smooth", row.get("estimated_distance", np.nan)
        )
        speed = row.get(
            "speed_smooth", row.get("relative_speed", np.nan)
        )
        ttc = row.get("ttc_used", row.get("ttc_smooth", np.nan))
        decision = str(
            row.get("aeb_decision", row.get("risk_smooth", "UNKNOWN"))
        )

        self.distance_card.value_label.setText(
            f"{distance:.2f} m" if pd.notna(distance) else "-- m"
        )

        self.speed_card.value_label.setText(
            f"{speed:.2f} m/s" if pd.notna(speed) else "-- m/s"
        )

        self.ttc_card.value_label.setText(
            f"{ttc:.2f} s"
            if pd.notna(ttc) and np.isfinite(ttc)
            else "INF"
        )

        self.risk_card.value_label.setText(decision)

        colors = {
            "SAFE": "#90EE90",
            "WARNING": "#FFA500",
            "AEB": "#FF5555",
            "AEB ACTIVE": "#FF5555",
            "UNKNOWN": "#D5DDE5"
        }

        color = colors.get(decision, "#D5DDE5")
        self.risk_card.value_label.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: bold;"
        )

        self.system_status.setText(
            "AEB ACTIVE" if decision in ("AEB", "AEB ACTIVE")
            else "MONITORING"
        )

        self.event_status.setText(
            f"Track ID: {row.get('track_id', '--')} | "
            f"Decision: {decision}"
        )

    def update_chart(self):
        self.figure.clear()

        if self.csv_data.empty or "frame" not in self.csv_data.columns:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "Load CSV results to view charts",
                ha="center", va="center"
            )
            ax.axis("off")
            self.canvas.draw()
            return

        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212, sharex=ax1)

        x = self.csv_data["frame"]

        distance_col = (
            "distance_smooth"
            if "distance_smooth" in self.csv_data.columns
            else "estimated_distance"
        )

        ttc_col = (
            "ttc_used" if "ttc_used" in self.csv_data.columns
            else "ttc_smooth" if "ttc_smooth" in self.csv_data.columns
            else "ttc"
        )

        if distance_col in self.csv_data.columns:
            ax1.plot(
                x, self.csv_data[distance_col],
                color="#4DA3FF", label="Distance (m)"
            )
            ax1.set_ylabel("Distance (m)")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.25)

        if ttc_col in self.csv_data.columns:
            ttc_values = pd.to_numeric(
                self.csv_data[ttc_col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

            ax2.plot(
                x, ttc_values,
                color="#E0A04B", label="TTC (s)"
            )
            ax2.axhline(
                1.5, color="red", linestyle="--",
                label="AEB threshold"
            )
            ax2.axhline(
                3.0, color="orange", linestyle=":",
                label="Warning threshold"
            )
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("TTC (s)")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.25)

        self.figure.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        self.timer.stop()

        if self.capture:
            self.capture.release()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = AEBWindow()
    window.show()

    sys.exit(app.exec())
