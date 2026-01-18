"""
Viewer + Settings GUI.

Connects to recorder_service.py video stream and displays it.
Edits settings (pre-roll, buffer, output directory, NT settings, camera selection).
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAction,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from camera_utils import enumerate_camera_choices
from config_utils import load_config, save_config, get_config_path


class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        pre_roll_seconds: int = 2,
        buffer_seconds: int = 2,
        output_dir: str = "",
        nt_server_ip: str = "10.0.67.2",
        nt_boolean_key: str = "Teleop",
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout()

        hl_pre = QHBoxLayout()
        hl_pre.addWidget(QLabel("Pre-roll (seconds):"))
        self.pre_roll_spin = QSpinBox()
        self.pre_roll_spin.setRange(0, 30)
        self.pre_roll_spin.setValue(int(pre_roll_seconds))
        hl_pre.addWidget(self.pre_roll_spin)
        layout.addLayout(hl_pre)

        hl_buf = QHBoxLayout()
        hl_buf.addWidget(QLabel("Buffer after stop (seconds):"))
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(0, 60)
        self.buffer_spin.setValue(int(buffer_seconds))
        hl_buf.addWidget(self.buffer_spin)
        layout.addLayout(hl_buf)

        hl_out = QHBoxLayout()
        hl_out.addWidget(QLabel("Recordings folder:"))
        self.output_dir_edit = QLineEdit(output_dir)
        hl_out.addWidget(self.output_dir_edit, stretch=1)
        btn_browse = QPushButton("Browse")
        hl_out.addWidget(btn_browse)
        layout.addLayout(hl_out)

        def on_browse():
            start = Path(self.output_dir_edit.text().strip() or str(Path.home()))
            d = QFileDialog.getExistingDirectory(self, "Select recordings folder", str(start))
            if d:
                self.output_dir_edit.setText(d)

        btn_browse.clicked.connect(on_browse)

        hl_ip = QHBoxLayout()
        hl_ip.addWidget(QLabel("NetworkTables server IP:"))
        self.nt_ip_edit = QLineEdit(nt_server_ip)
        hl_ip.addWidget(self.nt_ip_edit)
        layout.addLayout(hl_ip)

        hl_key = QHBoxLayout()
        hl_key.addWidget(QLabel("NT boolean key:"))
        self.nt_key_edit = QLineEdit(nt_boolean_key)
        hl_key.addWidget(self.nt_key_edit)
        layout.addLayout(hl_key)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)


class StreamClient(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    status = pyqtSignal(str)

    def __init__(self, host: str, port: int):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.running = False
        self._thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                self.status.emit(f"Connecting to {self.host}:{self.port}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                sock.settimeout(2.0)
                self.status.emit("Connected")

                buf = b""
                needed = None
                while self.running:
                    try:
                        chunk = sock.recv(4096)
                        if not chunk:
                            raise RuntimeError("Disconnected")
                        buf += chunk

                        while True:
                            if needed is None:
                                if len(buf) < 4:
                                    break
                                needed = struct.unpack(">I", buf[:4])[0]
                                buf = buf[4:]
                            if len(buf) < needed:
                                break
                            payload = buf[:needed]
                            buf = buf[needed:]
                            needed = None

                            arr = np.frombuffer(payload, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.frame_ready.emit(frame)
                    except socket.timeout:
                        continue
            except Exception as e:
                self.status.emit(f"Disconnected: {e}")
                try:
                    sock.close()
                except Exception:
                    pass
                # retry
                for _ in range(20):
                    if not self.running:
                        return
                    threading.Event().wait(0.1)


def send_control(cmd: dict, host: str, port: int) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((host, int(port)))
        payload = (json.dumps(cmd) + "\n").encode("utf-8")
        s.sendall(payload)
        s.close()
        return True
    except Exception:
        return False


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Viewer")
        self.setGeometry(100, 100, 1100, 700)

        self.video_label = QLabel("Waiting for video stream...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaa;")

        self.camera_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.settings_btn = QPushButton("Settings")
        self.start_btn = QPushButton("Start (R)")
        self.stop_btn = QPushButton("Stop (S)")

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        top.addWidget(self.camera_combo, stretch=1)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.settings_btn)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.video_label, stretch=1)
        root.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # Menu
        m = self.menuBar().addMenu("File")
        act_open_cfg = QAction("Open config location", self)
        act_open_cfg.triggered.connect(self.open_config_location)
        m.addAction(act_open_cfg)
        m.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

        rec = self.menuBar().addMenu("Recorder")
        act_start = QAction("Start recording (manual)", self)
        act_start.triggered.connect(lambda: self._send_cmd({"cmd": "start"}))
        rec.addAction(act_start)
        act_stop = QAction("Stop recording (manual)", self)
        act_stop.triggered.connect(lambda: self._send_cmd({"cmd": "stop"}))
        rec.addAction(act_stop)
        act_auto = QAction("Auto (NetworkTables)", self)
        act_auto.triggered.connect(lambda: self._send_cmd({"cmd": "auto"}))
        rec.addAction(act_auto)

        # Signals
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)
        self.settings_btn.clicked.connect(self.open_settings)
        self.start_btn.clicked.connect(lambda: self._send_cmd({"cmd": "start"}))
        self.stop_btn.clicked.connect(lambda: self._send_cmd({"cmd": "stop"}))

        # Stream client
        cfg = load_config()
        stream_host = str(cfg.get("stream", {}).get("host", "127.0.0.1")) if isinstance(cfg.get("stream"), dict) else "127.0.0.1"
        stream_port = int(cfg.get("stream", {}).get("port", 8765)) if isinstance(cfg.get("stream"), dict) else 8765
        ctrl = cfg.get("control", {}) if isinstance(cfg.get("control"), dict) else {}
        self.control_host = str(ctrl.get("host", stream_host))
        self.control_port = int(ctrl.get("port", 8766))

        self.client = StreamClient(stream_host, stream_port)
        self.client.frame_ready.connect(self.display_frame)
        self.client.status.connect(self.status_label.setText)
        self.client.start()

        self.refresh_cameras()
        self._load_selected_camera_from_config()

    def _send_cmd(self, cmd: dict):
        ok = send_control(cmd, host=self.control_host, port=self.control_port)
        if not ok:
            self.status_label.setText("Recorder not reachable (start `recorder_service.py`)")

    def open_config_location(self):
        try:
            p = get_config_path()
            folder = str(p.parent)
            if os.name == "nt":
                os.startfile(folder)
        except Exception:
            pass

    def refresh_cameras(self):
        choices = enumerate_camera_choices()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if not choices:
            self.camera_combo.addItem("No cameras found", None)
            self.camera_combo.setEnabled(False)
        else:
            self.camera_combo.setEnabled(True)
            for label, idx in choices:
                self.camera_combo.addItem(label, idx)
        self.camera_combo.blockSignals(False)

    def _load_selected_camera_from_config(self):
        cfg = load_config()
        cam = cfg.get("camera", {}) if isinstance(cfg.get("camera"), dict) else {}
        try:
            sel = int(cam.get("selected_index"))
        except Exception:
            return
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == sel:
                self.camera_combo.setCurrentIndex(i)
                break

    def on_camera_selected(self, _i: int):
        idx = self.camera_combo.currentData()
        if idx is None:
            return
        # persist selection
        cfg = load_config()
        cfg.setdefault("camera", {})
        cfg["camera"]["selected_index"] = int(idx)
        save_config(cfg)
        # ask recorder to switch immediately (if running)
        self._send_cmd({"cmd": "switch_camera", "index": int(idx)})

    def open_settings(self):
        cfg = load_config()
        settings = cfg.get("settings", {}) if isinstance(cfg.get("settings"), dict) else {}
        nt_cfg = cfg.get("nt", {}) if isinstance(cfg.get("nt"), dict) else {}

        d = SettingsDialog(
            self,
            pre_roll_seconds=int(settings.get("pre_roll_seconds", 2)),
            buffer_seconds=int(settings.get("buffer_seconds", 2)),
            output_dir=str(settings.get("output_dir", str(Path.cwd()))),
            nt_server_ip=str(nt_cfg.get("server_ip", "10.0.67.2")),
            nt_boolean_key=str(nt_cfg.get("boolean_key", "Teleop")),
        )
        if d.exec_() != QDialog.Accepted:
            return

        cfg.setdefault("settings", {})
        cfg.setdefault("nt", {})
        cfg["settings"]["pre_roll_seconds"] = int(d.pre_roll_spin.value())
        cfg["settings"]["buffer_seconds"] = int(d.buffer_spin.value())
        cfg["settings"]["output_dir"] = d.output_dir_edit.text().strip() or str(Path.cwd())
        cfg["nt"]["server_ip"] = d.nt_ip_edit.text().strip() or "10.0.67.2"
        cfg["nt"]["boolean_key"] = d.nt_key_edit.text().strip() or "Teleop"
        save_config(cfg)

        # recorder will hot-reload config; also push auto reconnect hint
        self.status_label.setText("Saved settings (recorder will auto-reload)")

    def display_frame(self, frame: np.ndarray):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self.client.stop()
        except Exception:
            pass
        event.accept()

    def keyPressEvent(self, event):
        # Mirror headless recorder hotkeys for convenience
        try:
            k = event.key()
            if k == Qt.Key_R:
                self._send_cmd({"cmd": "start"})
                return
            if k == Qt.Key_S:
                self._send_cmd({"cmd": "stop"})
                return
            if k == Qt.Key_A:
                self._send_cmd({"cmd": "auto"})
                return
            if k == Qt.Key_Q:
                self.close()
                return
        except Exception:
            pass
        super().keyPressEvent(event)


def main():
    app = QApplication([])
    w = ViewerWindow()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()

