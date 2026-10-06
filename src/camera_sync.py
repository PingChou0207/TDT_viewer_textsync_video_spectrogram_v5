"""Timestamp-driven preview for TDT Cam# videos stored beside a block.

Cam# epoc onsets are the authoritative frame times.  AVI container FPS is
deliberately not used to align video to neural data.
"""

import os
import re

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import cv2
except ImportError:
    cv2 = None


_CAMERA_FILE = re.compile(r"(?:^|_)(Cam\d+)\.(avi|mp4)$", re.IGNORECASE)


def discover_cameras(block_path, epoc_onsets, epoc_values):
    """Return matching video/epoc pairs, without opening any video files."""
    if not block_path or not os.path.isdir(block_path):
        return []
    try:
        files = list(os.scandir(block_path))
    except OSError:
        return []

    videos = {}
    for entry in files:
        if not entry.is_file():
            continue
        match = _CAMERA_FILE.search(entry.name)
        if match:
            videos.setdefault(match.group(1).lower(), entry.path)

    cameras = []
    for store, onsets in epoc_onsets.items():
        path = videos.get(store.lower())
        if path is None:
            continue
        times = np.asarray(onsets, dtype=float).ravel()
        values = np.asarray(epoc_values.get(store, []), dtype=float).ravel()
        if values.size != times.size:
            values = np.arange(1, times.size + 1, dtype=float)
        valid = np.isfinite(times) & np.isfinite(values)
        times, values = times[valid], values[valid]
        if times.size == 0:
            continue
        order = np.argsort(times, kind="stable")
        times, values = times[order], values[order]
        # TDT Cam# event values are one-based AVI frame numbers.  Older or
        # malformed sessions may lack these values; then use epoc order.
        frame_ids = np.rint(values).astype(np.int64) - 1
        if np.any(frame_ids < 0) or np.any(np.abs(values - np.rint(values)) > 0.01):
            frame_ids = np.arange(times.size, dtype=np.int64)
        cameras.append((store, path, times, frame_ids))
    return sorted(cameras, key=lambda item: item[0].lower())


class CameraDock(QtWidgets.QDockWidget):
    """Random-access camera frame viewer synchronized to TDT epoc onsets."""

    timeRequested = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__("Camera", parent)
        self.setObjectName("TDTCameraDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.setMinimumWidth(260)
        self._cameras = []
        self._capture = None
        self._frame_count = 0
        self._current_event = -1
        self._current_frame = -1
        self._image = None
        self._play_start_time = 0.0
        self._elapsed = QtCore.QElapsedTimer()

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._select_camera)
        layout.addWidget(self.camera_combo)

        self.image_label = QtWidgets.QLabel("No camera loaded")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(240, 180)
        self.image_label.setStyleSheet("background: #181818; color: #dddddd;")
        layout.addWidget(self.image_label, 1)

        controls = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton("◀ Frame")
        self.play_button = QtWidgets.QPushButton("Play")
        self.next_button = QtWidgets.QPushButton("Frame ▶")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.play_button.clicked.connect(self.toggle_playback)
        self.next_button.clicked.connect(lambda: self.step(1))
        for button in (self.previous_button, self.play_button, self.next_button):
            controls.addWidget(button)
        layout.addLayout(controls)

        self.info_label = QtWidgets.QLabel("No camera timestamps")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.setWidget(content)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._play_tick)
        self.visibilityChanged.connect(self._visibility_changed)

    @property
    def playing(self):
        return self._timer.isActive()

    @property
    def has_camera(self):
        return bool(self._cameras)

    def configure(self, block_path, epoc_onsets, epoc_values):
        self.stop_playback()
        self._release_capture()
        self._cameras = discover_cameras(block_path, epoc_onsets, epoc_values)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItems([cam[0] for cam in self._cameras])
        self.camera_combo.blockSignals(False)
        self._current_event = -1
        self._image = None
        if self._cameras:
            self._select_camera(0)
        else:
            self.image_label.setText("No Cam# video with matching epoc found")
            self.info_label.setText("Camera requires a Cam# video and Cam# epoc in the block.")
        return self.has_camera

    def _release_capture(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._frame_count = 0
        self._current_frame = -1

    def _select_camera(self, index):
        self.stop_playback()
        self._release_capture()
        self._current_event = -1
        self._image = None
        if index < 0 or index >= len(self._cameras):
            return
        store, path, times, _ = self._cameras[index]
        if cv2 is None:
            self.image_label.setText("OpenCV is not installed")
            self.info_label.setText("Install opencv-python-headless to decode camera video.")
            return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            self.image_label.setText("Could not open camera video")
            self.info_label.setText(path)
            return
        self._capture = capture
        self._frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.info_label.setText(
            f"{store}: {times.size} timestamps, {self._frame_count} video frames"
        )
        self.seek(float(times[0]))

    def _current_camera(self):
        index = self.camera_combo.currentIndex()
        return self._cameras[index] if 0 <= index < len(self._cameras) else None

    def seek(self, seconds):
        camera = self._current_camera()
        if camera is None or self._capture is None:
            return
        store, _path, times, frame_ids = camera
        event = int(np.searchsorted(times, seconds, side="right") - 1)
        if event < 0:
            self._current_event = -1
            self._current_frame = -1
            self._image = None
            self.image_label.setPixmap(QtGui.QPixmap())
            self.image_label.setText("Camera has not started")
            self.info_label.setText(f"{store}: first frame at {times[0]:.3f} s")
            return
        frame_id = int(frame_ids[event])
        if frame_id < 0 or (self._frame_count and frame_id >= self._frame_count):
            self.image_label.setPixmap(QtGui.QPixmap())
            self.image_label.setText("No matching video frame")
            self.info_label.setText(f"{store}: frame {frame_id + 1} is outside the video")
            return
        if frame_id != self._current_frame:
            sequential = self._current_frame >= 0 and frame_id == self._current_frame + 1
            if not sequential and not self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
                self.info_label.setText(f"{store}: video seek failed at frame {frame_id + 1}")
                return
            ok, frame = self._capture.read()
            if not ok:
                self.info_label.setText(f"{store}: video decode failed at frame {frame_id + 1}")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            self._image = QtGui.QImage(
                rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888
            ).copy()
            self._current_frame = frame_id
            self._show_image()
        self._current_event = event
        self.info_label.setText(
            f"{store} frame {frame_id + 1} / {self._frame_count or '?'}"
            f" | TDT {times[event]:.3f} s | viewer {seconds:.3f} s"
        )

    def _show_image(self):
        if self._image is None:
            return
        pixmap = QtGui.QPixmap.fromImage(self._image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_image()

    def step(self, delta):
        camera = self._current_camera()
        if camera is None:
            return
        self.stop_playback()
        times = camera[2]
        event = min(max(0, self._current_event + delta), len(times) - 1)
        seconds = float(times[event])
        self.seek(seconds)
        self.timeRequested.emit(seconds)

    def toggle_playback(self):
        if self.playing:
            self.stop_playback()
            return
        camera = self._current_camera()
        if camera is None or self._capture is None:
            return
        times = camera[2]
        event = max(0, self._current_event)
        self._play_start_time = float(times[event])
        self._elapsed.start()
        self._timer.start()
        self.play_button.setText("Pause")

    def _play_tick(self):
        camera = self._current_camera()
        if camera is None:
            self.stop_playback()
            return
        seconds = self._play_start_time + self._elapsed.elapsed() / 1000.0
        if seconds > float(camera[2][-1]):
            self.stop_playback()
            return
        self.seek(seconds)
        self.timeRequested.emit(seconds)

    def stop_playback(self):
        self._timer.stop()
        self.play_button.setText("Play")

    def _visibility_changed(self, visible):
        if not visible:
            self.stop_playback()

    def release(self):
        self.stop_playback()
        self._release_capture()

    def closeEvent(self, event):
        self.stop_playback()
        super().closeEvent(event)
