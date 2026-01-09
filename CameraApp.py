import sys
import os
import cv2
import time
from collections import deque
from Utils import RingBuffer
import threading
try:
    from networktables import NetworkTables
except Exception:
    NetworkTables = None

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox,
    QHBoxLayout, QVBoxLayout, QMessageBox, QShortcut, QLineEdit,
    QDialog, QDialogButtonBox, QFileDialog, QSpinBox
)

def probe_cameras(max_index: int = 10):
    """
    Try to enumerate camera devices.
    Prefer `pygrabber` to get device names; fall back to probing indices with OpenCV.
    Returns a list of tuples (index, name).
    """
    # Try pygrabber first to obtain human-friendly device names on Windows
    try:
        from pygrabber.dshow_graph import FilterGraph

        fg = FilterGraph()
        names = fg.get_input_devices() or []
        return [(i, name) for i, name in enumerate(names)]
    except Exception:
        # pygrabber unavailable or failed — fall back to probing with OpenCV
        available = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap is not None:
                    ret, _ = cap.read()
                    if ret:
                        available.append((i, f"Camera {i}"))
                    cap.release()
            except Exception:
                pass
        return available


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Camera Switcher")

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # rolling buffer of recent frame timestamps (seconds)
        self._frame_times = deque(maxlen=120)
        # recording buffers and control
        self._record_frames = []  # list of (timestamp, frame)
        self._record_start_ts = None
        self.recording = False
        self.record_writer = None
        self._record_writer_open = False
        # lock to protect record buffer when using background threads
        self._record_lock = threading.Lock()
        # timestamp when current recording started (for on-frame timer)
        self._rec_start_time = None
        # post-roll settings (seconds) and state
        self._postroll_seconds = 2
        self._postroll_active = False
        self._postroll_end = None
        # pre-roll (seconds) controls how many seconds of buffer are kept before starting a recording
        self._preroll_seconds = 2
        # NetworkTables boolean key name to look for (default 'Teleop')
        self._nt_bool_key = 'Teleop'
        # default recording resolution (width, height)
        self._resolution = (640, 480)
        # where to save recordings (can be configured via Settings)
        try:
            default_videos = Path.home() / 'Videos'
            if default_videos.exists():
                self._record_path = str(default_videos)
            else:
                self._record_path = str(Path.cwd())
        except Exception:
            self._record_path = str(Path.cwd())
        # initialize save path label
        try:
            self.save_path_label.setText(str(self._record_path))
        except Exception:
            pass

        # UI
        self.video_label = QLabel("No camera feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")

        self.camera_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.start_btn = QPushButton("Record")
        self.stop_btn = QPushButton("Stop Recording")

        # NetworkTables server input + connect
        self.nt_ip_edit = QLineEdit()
        self.nt_ip_edit.setFixedWidth(60)
        self.nt_ip_edit.setPlaceholderText("0")
        self.nt_connect_btn = QPushButton("Connect NT")
        self.nt_status = QLabel("NT: disabled")
        # save the IP when the user finishes editing the field
        self.nt_ip_edit.editingFinished.connect(self.save_settings)

        # Settings button opens settings dialog (post-roll, record path)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        # show current save path briefly in the controls
        self.save_path_label = QLabel("")
        self.save_path_label.setStyleSheet("font-size: 10px; color: #ccc;")
        # show config path (shortened) with tooltip for full path (kept internal)
        self.config_path_label = QLabel("")
        self.config_path_label.setStyleSheet("font-size: 10px; color: #aaa;")
        self.config_path_label.setToolTip("")

        # Button to open the recordings folder
        self.open_recordings_btn = QPushButton("Open Recordings")
        self.open_recordings_btn.clicked.connect(self._open_recordings_folder)
        self.open_recordings_btn.setToolTip(str(self._record_path))

        self.stop_btn.setEnabled(False)  

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Camera:"))
        controls.addWidget(self.camera_combo, stretch=1)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)

        # NetworkTables controls
        controls.addWidget(QLabel("Team #:"))
        controls.addWidget(self.nt_ip_edit)
        controls.addWidget(self.nt_connect_btn)
        controls.addWidget(self.nt_status)

        controls.addWidget(self.settings_btn)
        # save/config labels intentionally not added to main controls to avoid clutter
        # they remain available internally and in the Settings dialog


        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self.video_label, stretch=1)

        # bottom controls (right-aligned) with Open Recordings button
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.open_recordings_btn)
        root.addLayout(bottom)

        self.setLayout(root)

        # Signals
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.start_btn.clicked.connect(lambda: self.start_recording(manual=True))
        self.stop_btn.clicked.connect(self.stop_recording)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        self.camera_combo.currentIndexChanged.connect(lambda _: self.save_settings())

        # Load cameras on start
        self.refresh_cameras()

        # NT default ip and auto-connect attempt
        self._nt_server_ip = "10.0.67.1"
        self._nt_team_number = 67
        self.nt_connect_btn.clicked.connect(self._on_nt_connect_clicked)
        # config path: prefer %APPDATA%/CameraRecorder/config.json on Windows; migrate legacy home file if present
        try:
            appdata = os.getenv('APPDATA')
            if appdata:
                cfg_dir = Path(appdata) / "CameraRecorder"
            else:
                cfg_dir = Path.home()
            cfg_dir.mkdir(parents=True, exist_ok=True)
            new_path = cfg_dir / "config.json"
            old_path = Path.home() / ".camera_recorder_config.json"
            if old_path.exists() and not new_path.exists():
                try:
                    old_path.replace(new_path)
                    print(f"[CFG] migrated {old_path} -> {new_path}")
                except Exception:
                    try:
                        import shutil
                        shutil.copy2(old_path, new_path)
                        print(f"[CFG] copied {old_path} -> {new_path}")
                    except Exception as e:
                        print(f"[CFG] migration failed: {e}")
            self._config_path = new_path
        except Exception:
            self._config_path = Path("camera_recorder_config.json")

        self.load_settings()
        # try to connect automatically
        self._connect_networktables(self._nt_server_ip)

        # ensure the widget can receive key events and create window-level shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        sc_r = QShortcut(QKeySequence('R'), self)
        sc_r.setContext(Qt.WindowShortcut)
        sc_r.activated.connect(lambda: self.start_recording(manual=True))

        sc_s = QShortcut(QKeySequence('S'), self)
        sc_s.setContext(Qt.WindowShortcut)
        sc_s.activated.connect(self.stop_recording)

        sc_q = QShortcut(QKeySequence('Q'), self)
        sc_q.setContext(Qt.WindowShortcut)
        sc_q.activated.connect(self.close)

        # update save & config path labels at startup
        try:
            self._update_save_path_label()
            self._update_config_path_label()
        except Exception:
            pass

    def _update_save_path_label(self):
        try:
            display = self._record_path
            # shorten long paths
            if len(display) > 40:
                display = '...' + display[-37:]
            self.save_path_label.setText(display)
            # update tooltip for the open recordings button if present
            try:
                self.open_recordings_btn.setToolTip(str(self._record_path))
            except Exception:
                pass
        except Exception:
            pass

    def _update_config_path_label(self):
        try:
            p = getattr(self, '_config_path', None)
            if p:
                display = str(p)
                if len(display) > 40:
                    display = '...' + display[-37:]
                self.config_path_label.setText(display)
                self.config_path_label.setToolTip(str(p))
            else:
                self.config_path_label.setText("")
                self.config_path_label.setToolTip("")
        except Exception:
            pass

    def _team_to_ip(self, team_str: str) -> str:
        """Convert team number string to IP address. 0 = localhost, else 10.0.{team}.1"""
        try:
            team = int(team_str.strip())
            if team == 0:
                return "127.0.0.1"
            else:
                return f"10.0.{team}.1"
        except Exception:
            return "10.0.67.1"  # fallback default

    def _ip_to_team(self, ip: str) -> str:
        """Extract team number from IP address for display."""
        try:
            if ip == "127.0.0.1":
                return "0"
            parts = ip.split('.')
            if len(parts) == 4 and parts[0] == '10' and parts[1] == '0':
                return parts[2]
            return "67"  # fallback
        except Exception:
            return "67"

    def open_settings_dialog(self):
        d = SettingsDialog(self, preroll=self._preroll_seconds, postroll=self._postroll_seconds, record_path=self._record_path, nt_bool_key=self._nt_bool_key, resolution=getattr(self, '_resolution', (640, 480)))
        if d.exec_() == QDialog.Accepted:
            # apply settings
            self._preroll_seconds = int(d.preroll_spin.value())
            self._postroll_seconds = int(d.postroll_spin.value())
            self._record_path = d.record_path_edit.text().strip() or self._record_path
            # apply nt key
            try:
                newkey = d.nt_key_edit.text().strip()
                if newkey:
                    self._nt_bool_key = newkey
            except Exception:
                pass
            # ensure path exists
            try:
                Path(self._record_path).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # rebuild ring buffer according to new pre-roll setting while preserving recent frames
            try:
                old_frames = []
                try:
                    old_frames = [f for f in self.ring_buffer.get_frames() if f is not None]
                except Exception:
                    old_frames = []
                # estimate fps
                fps_guess = 30.0
                try:
                    cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if cap_fps and cap_fps > 1.0:
                        fps_guess = cap_fps
                except Exception:
                    pass
                try:
                    timer_interval = self.timer.interval() if hasattr(self, 'timer') and self.timer is not None else 30
                    timer_fps = 1000.0 / timer_interval if timer_interval > 0 else 30.0
                    if not (cap_fps and cap_fps > 1.0):
                        fps_guess = timer_fps
                except Exception:
                    pass
                new_n = max(1, int(round(self._preroll_seconds * fps_guess)))
                self.ring_buffer = RingBuffer(new_n)
                for fr in old_frames[-new_n:]:
                    try:
                        self.ring_buffer.add_frame(fr)
                    except Exception:
                        pass
            except Exception:
                pass

            # apply resolution change
            try:
                rw = int(d.res_w_spin.value())
                rh = int(d.res_h_spin.value())
                self._resolution = (rw, rh)
                # apply immediately if camera is open
                try:
                    if getattr(self, 'cap', None) is not None:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
                except Exception:
                    pass
            except Exception:
                pass

            self._update_save_path_label()
            self.save_settings()



    def refresh_cameras(self):
        was_running = self.timer.isActive()
        if was_running:
            self.stop_camera()

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        cams = probe_cameras(10)
        if not cams:
            self.camera_combo.addItem("No cameras found", None)
            self.camera_combo.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.video_label.setText("No cameras detected.\nClick Refresh or check permissions.")
        else:
            for item in cams:
                # each item is (index, name)
                if isinstance(item, tuple) and len(item) == 2:
                    idx, name = item
                else:
                    idx, name = item, f"Camera {item}"
                self.camera_combo.addItem(f"{idx}: {name}", idx)
            self.camera_combo.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.video_label.setText("Ready. Click Start.")

        self.camera_combo.blockSignals(False)

        # Always start the first available camera so the feed is shown immediately
        if cams:
            self.start_camera()

    def selected_camera_index(self):
        return self.camera_combo.currentData()

    def start_camera(self):
        cam_index = self.selected_camera_index()
        if cam_index is None:
            return

        self.open_camera(cam_index)

    def open_camera(self, cam_index: int):
        self.stop_camera()

        # Try to open camera
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            QMessageBox.critical(self, "Camera Error", f"Could not open camera {cam_index}.")
            return

        # Optional: set resolution (from settings)
        try:
            w, h = getattr(self, '_resolution', (640, 480))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass

        # Recording buttons: allow recording while camera is running
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("")

        # remember which camera index is open
        self.current_camera_index = cam_index

        # ensure any previous recorder is cleared
        self.record_writer = None
        self.recording = False
        # ring buffer for pre-roll frames (size based on pre-roll seconds and estimated fps)
        try:
            # estimate FPS using camera cap if available; fall back to timer interval
            fps_guess = 30.0
            try:
                cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if cap_fps and cap_fps > 1.0:
                    fps_guess = cap_fps
            except Exception:
                pass
            try:
                timer_interval = self.timer.interval() if hasattr(self, 'timer') and self.timer is not None else 30
                timer_fps = 1000.0 / timer_interval if timer_interval > 0 else 30.0
                if not (cap_fps and cap_fps > 1.0):
                    fps_guess = timer_fps
            except Exception:
                pass
            buf_frames = max(1, int(round(self._preroll_seconds * fps_guess)))
            self.ring_buffer = RingBuffer(buf_frames)
        except Exception:
            try:
                self.ring_buffer = RingBuffer(50)
            except Exception:
                self.ring_buffer = None
        # post-roll configuration: seconds to continue recording after Stop pressed
        self._postroll_active = False
        self._postroll_end = None        # start a global timer for display (runs all the time)
        self._global_timer_start = time.time()
        # manual mode (user started/stopped recording) — when False, network Teleop controls recording
        self._manual = False

        self.timer.start(30)  # ~33 fps

        # NetworkTables handle (connected via UI)
        self._nt = None
        # Note: connection attempted at startup using the default IP field above (if present)


    def stop_camera(self):
        # stop recording if active
        try:
            if getattr(self, 'recording', False):
                self.stop_recording()
        except Exception:
            pass

        if self.timer.isActive():
            self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.start_btn.setEnabled(True if self.camera_combo.isEnabled() else False)
        self.stop_btn.setEnabled(False)

    def on_camera_changed(self):
        # If currently running, switch immediately
        if self.timer.isActive():
            cam_index = self.selected_camera_index()
            if cam_index is not None:
                self.open_camera(cam_index)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # record timestamp for FPS estimation
        try:
            now = time.time()
            self._frame_times.append(now)
        except Exception:
            now = time.time()

        # draw recording status and elapsed time on the frame (display and recording)
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 0, 0)
            thickness = 2
            pos_status = (50, 50)
            pos_time = (500, 50)

            status_text = str(self.recording)
            cv2.putText(frame, status_text, pos_status, font, font_scale, color, thickness, cv2.LINE_AA)
            # elapsed: continuous timer that only resets on start_recording
            if getattr(self, '_global_timer_start', None) is None:
                self._global_timer_start = time.time()
            elapsed = time.time() - self._global_timer_start

            time_text = f"{elapsed:.1f}s"
            cv2.putText(frame, time_text, pos_time, font, font_scale, color, thickness, cv2.LINE_AA)
        except Exception:
            pass

        # If recording or post-roll active, buffer or write the timestamped frame
        if getattr(self, 'recording', False) or getattr(self, '_postroll_active', False):
            ts = now
            # If VideoWriter is open, write directly; otherwise buffer
            if getattr(self, '_record_writer_open', False) and self.record_writer:
                try:
                    self.record_writer.write(frame)
                except Exception as e:
                    print(f"[REC] VideoWriter write failed: {e}")
            else:
                # Fallback: buffer frames in memory
                try:
                    with self._record_lock:
                        self._record_frames.append((ts, frame.copy()))
                except Exception:
                    try:
                        with self._record_lock:
                            self._record_frames.append((ts, frame))
                    except Exception:
                        pass
        else:
            # maintain ring buffer when not recording (pre-roll)
            # Reduce copies: only copy if we need to preserve the frame
            try:
                self.ring_buffer.add_frame(frame)
            except Exception:
                pass

        # If not in manual mode, read network Teleop flag and start/stop recording accordingly
        try:
            if not self._manual and self._nt is not None:
                try:
                    key = getattr(self, '_nt_bool_key', 'Teleop') or 'Teleop'
                    teleop = bool(self._nt.getBoolean(key, False))
                except Exception:
                    teleop = False

                # print(f"[NT] Teleop={teleop} manual={self._manual} recording={self.recording} postroll={getattr(self, '_postroll_active', False)}")
                if teleop:
                    # if teleop becomes true and we're not recording, start
                    if not self.recording:
                        # if in post-roll, cancel and restart recording
                        if getattr(self, '_postroll_active', False):
                            self._postroll_active = False
                        # print("[NT] Teleop True -> automated start_recording()")
                        self.start_recording()
                else:
                    # teleop is false: if we were recording or in post-roll, begin stop/post-roll
                    if self.recording or getattr(self, '_postroll_active', False):
                        # print("[NT] Teleop False -> automated stop_recording()")
                        self.stop_recording()
        except Exception:
            print("[NT] Teleop handling exception", flush=True)
            pass

        # If post-roll is active and we've reached the end time, finalize recording
        try:
            if getattr(self, '_postroll_active', False) and getattr(self, '_postroll_end', 0) is not None:
                if time.time() >= self._postroll_end:
                    # finalize recording and clear post-roll
                    self._finalize_recording()
        except Exception:
            pass

        # Convert BGR (OpenCV) -> RGB (Qt)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Fit preview label while keeping aspect ratio
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        # save settings then stop recording and camera cleanly
        try:
            self.save_settings()
        except Exception:
            pass
        try:
            if getattr(self, 'recording', False):
                self.stop_recording()
        except Exception:
            pass
        self.stop_camera()
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts: R=start, S=stop, Q=quit."""
        try:
            k = event.key()
            if k == Qt.Key_R:
                # start recording
                self.start_recording()
                return
            if k == Qt.Key_S:
                # stop recording
                self.stop_recording()
                return
            if k == Qt.Key_Q:
                # quit app
                self.close()
                return
        except Exception:
            pass

    def showEvent(self, event):
        """Ensure the window becomes active and receives focus when shown so shortcuts work."""
        try:
            super().showEvent(event)
        except Exception:
            pass

        def _focus():
            try:
                # Raise and activate window; set focus to this widget
                self.raise_()
                self.activateWindow()
                self.setFocus(Qt.OtherFocusReason)
            except Exception:
                pass

        # run shortly after show to ensure the window manager has created the window
        QTimer.singleShot(50, _focus)

    def start_recording(self, manual: bool = False):
        """Begin recording to an MP4 file. Requires an opened camera.

        Args:
            manual: when True, this start was requested manually (UI/keyboard). When False it's automated (Teleop).
        """
        
        if self.cap is None or not getattr(self, 'cap', None):
            QMessageBox.warning(self, "No Camera", "No camera is open to record.")
            return

        if getattr(self, 'recording', False):
            return

        # prepare in-memory buffer for timestamped frames
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        idx = getattr(self, 'current_camera_index', '0')
        filename = f"record_{idx}_{ts}.mp4"
        # ensure directory exists and use configured record path
        try:
            rec_dir = Path(getattr(self, '_record_path', Path.cwd()))
            rec_dir.mkdir(parents=True, exist_ok=True)
            filename = str(rec_dir / filename)
        except Exception:
            filename = filename
        print(f"[REC] start_recording(manual={manual}) -> {filename}")

        # prefill recording buffer with frames from ring buffer (pre-roll)
        now = time.time()
        warm = [f for f in self.ring_buffer.get_frames() if f is not None]
        # estimate fps from recent frame times if available
        fps_est = None
        try:
            if len(self._frame_times) >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                if span > 0:
                    fps_est = (len(self._frame_times) - 1) / span
        except Exception:
            fps_est = None

        if not fps_est or fps_est < 1.0:
            try:
                cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                cap_fps = 0.0
            timer_interval = self.timer.interval() if hasattr(self, 'timer') and self.timer is not None else 30
            timer_fps = 1000.0 / timer_interval if timer_interval > 0 else 30.0
            fps_est = cap_fps if cap_fps and cap_fps > 1.0 else float(round(timer_fps))

        fps_est = max(1.0, min(120.0, float(fps_est)))

        # assign timestamps to warm frames ending at now
        n_w = len(warm)
        rec_frames = []
        for i, frm in enumerate(warm):
            # timestamp such that last warm frame is slightly before now
            ts_i = now - float(n_w - i) / fps_est
            rec_frames.append((ts_i, frm))

        # Open VideoWriter for incremental writing
        try:
            first_frame = warm[0] if warm else None
            if first_frame is None:
                # If no pre-roll frames, we'll write on first frame in update_frame
                h, w = 480, 640
            else:
                h, w = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.record_writer = cv2.VideoWriter(filename, fourcc, fps_est, (w, h))
            if self.record_writer and self.record_writer.isOpened():
                self._record_writer_open = True
                # Write pre-roll frames
                for _, frm in rec_frames:
                    try:
                        self.record_writer.write(frm)
                    except Exception:
                        pass
                print(f"[REC] VideoWriter opened, wrote {len(rec_frames)} pre-roll frames")
            else:
                print(f"[REC] VideoWriter failed to open, falling back to buffering")
                self._record_writer_open = False
                with self._record_lock:
                    self._record_frames = rec_frames
        except Exception as e:
            print(f"[REC] VideoWriter init exception: {e}, falling back to buffering")
            self._record_writer_open = False
            with self._record_lock:
                self._record_frames = rec_frames
        self._desired_filename = filename
        self.recording = True
        self._rec_start_time = time.time()
        # reset the global timer so it starts counting from zero for this recording
        self._global_timer_start = time.time()
        # entering manual mode if requested (UI/keyboard); automated starts (Teleop) don't set manual
        self._manual = bool(manual)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.setWindowTitle(f"PyQt5 Camera Switcher - Recording (buffering): {filename}")

    def load_settings(self):
        """Load persistent settings (NT server IP, selected camera, post-roll) from a JSON file."""
        try:
            p = Path(getattr(self, '_config_path', Path.home() / '.camera_recorder_config.json'))
            print(f"[CFG] load_settings path={p} exists={p.exists()}")
            if p.exists():
                try:
                    with p.open('r', encoding='utf-8') as fh:
                        cfg = json.load(fh)
                except Exception as e:
                    print(f"[CFG] Failed to read JSON: {e}")
                    cfg = {}
            else:
                cfg = {}

            print(f"[CFG] loaded cfg={cfg}")

            nt_block = cfg.get('nt') if isinstance(cfg.get('nt'), dict) else {}
            team_num = nt_block.get('team_number') if nt_block else None
            ip = nt_block.get('server_ip') if nt_block else None
            
            if team_num:
                self.nt_ip_edit.setText(str(team_num))
                if ip:
                    self._nt_server_ip = ip
                else:
                    self._nt_server_ip = self._team_to_ip(str(team_num))
            elif ip:
                # legacy: convert IP to team number for display
                self._nt_server_ip = ip
                team = self._ip_to_team(ip)
                self.nt_ip_edit.setText(team)
            # networktables boolean key
            try:
                nt_key = nt_block.get('boolean_key') if nt_block else None
            except Exception:
                nt_key = None
            if nt_key:
                try:
                    self._nt_bool_key = str(nt_key)
                except Exception:
                    pass

            post = None
            try:
                post = cfg.get('settings', {}).get('postroll_seconds')
            except Exception:
                post = None
            if post is not None:
                try:
                    pv = int(post)
                    self._postroll_seconds = pv
                    if hasattr(self, 'postroll_spin'):
                        self.postroll_spin.setValue(pv)
                except Exception:
                    pass

            # recording save path
            try:
                rec_path = cfg.get('settings', {}).get('record_path')
            except Exception:
                rec_path = None
            if rec_path:
                try:
                    self._record_path = str(rec_path)
                    self._update_save_path_label()
                except Exception:
                    pass

            # pre-roll
            try:
                pr = cfg.get('settings', {}).get('pre_roll_seconds')
            except Exception:
                pr = None
            if pr is not None:
                try:
                    self._preroll_seconds = int(pr)
                    # if camera already open, rebuild ring buffer with new size
                    try:
                        if getattr(self, 'cap', None) is not None and getattr(self, 'ring_buffer', None) is not None:
                            old_frames = [f for f in self.ring_buffer.get_frames() if f is not None]
                            # estimate fps similar to open_camera
                            fps_guess = 30.0
                            try:
                                cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                                if cap_fps and cap_fps > 1.0:
                                    fps_guess = cap_fps
                            except Exception:
                                pass
                            try:
                                timer_interval = self.timer.interval() if hasattr(self, 'timer') and self.timer is not None else 30
                                timer_fps = 1000.0 / timer_interval if timer_interval > 0 else 30.0
                                if not (cap_fps and cap_fps > 1.0):
                                    fps_guess = timer_fps
                            except Exception:
                                pass
                            new_n = max(1, int(round(self._preroll_seconds * fps_guess)))
                            self.ring_buffer = RingBuffer(new_n)
                            for fr in old_frames[-new_n:]:
                                try:
                                    self.ring_buffer.add_frame(fr)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass

            # resolution
            try:
                res = cfg.get('settings', {}).get('resolution')
            except Exception:
                res = None
            if res and isinstance(res, dict):
                try:
                    rw = int(res.get('w', 0))
                    rh = int(res.get('h', 0))
                    if rw > 0 and rh > 0:
                        self._resolution = (rw, rh)
                        # apply to open camera if present
                        try:
                            if getattr(self, 'cap', None) is not None:
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
                        except Exception:
                            pass
                except Exception:
                    pass
            # update config path label (show where settings are stored)
            try:
                self._update_config_path_label()
            except Exception:
                pass

            cam_idx = None
            try:
                cam_idx = cfg.get('camera', {}).get('selected_index')
            except Exception:
                cam_idx = None
            if cam_idx is not None:
                try:
                    saved_idx = int(cam_idx)
                    # find combo entry with itemData == saved_idx
                    for i in range(self.camera_combo.count()):
                        if self.camera_combo.itemData(i) == saved_idx:
                            self.camera_combo.setCurrentIndex(i)
                            # If camera not already running, start it
                            if not self.timer.isActive():
                                self.start_camera()
                            break
                except Exception as e:
                    print(f"[CFG] failed to apply saved camera index: {e}")
                    pass
        except Exception as e:
            print(f"[CFG] load_settings exception: {e}")
            pass

    def save_settings(self):
        """Save persistent settings (NT server IP, selected camera, post-roll) to a JSON file."""
        try:
            cfg = {}
            # load existing if present to avoid clobbering unknown keys
            p = Path(getattr(self, '_config_path', Path.home() / '.camera_recorder_config.json'))
            print(f"[CFG] save_settings path={p} exists_before={p.exists()}")
            if p.exists():
                try:
                    with p.open('r', encoding='utf-8') as fh:
                        cfg = json.load(fh) or {}
                except Exception as e:
                    print(f"[CFG] failed to read existing cfg: {e}")
                    cfg = {}

            # set nt (prefer the explicit text field, otherwise last known server ip)
            cfg.setdefault('nt', {})
            current_team = self.nt_ip_edit.text().strip()
            if current_team:
                # save team number and computed IP
                cfg['nt']['team_number'] = current_team
                cfg['nt']['server_ip'] = self._team_to_ip(current_team)
            else:
                # fallback to stored values
                last_ip = getattr(self, '_nt_server_ip', '') or cfg['nt'].get('server_ip', '')
                cfg['nt']['server_ip'] = last_ip
                cfg['nt']['team_number'] = self._ip_to_team(last_ip)
            # persist the boolean key name
            try:
                cfg['nt']['boolean_key'] = str(getattr(self, '_nt_bool_key', 'Teleop'))
            except Exception:
                pass

            # set camera
            sel = self.selected_camera_index()
            if sel is not None:
                cfg.setdefault('camera', {})
                cfg['camera']['selected_index'] = int(sel)
            # set settings
            cfg.setdefault('settings', {})
            cfg['settings']['postroll_seconds'] = int(getattr(self, '_postroll_seconds', 2))
            # pre-roll seconds
            cfg['settings']['pre_roll_seconds'] = int(getattr(self, '_preroll_seconds', 2))
            # resolution
            try:
                cfg.setdefault('settings', {})
                w, h = getattr(self, '_resolution', (640, 480))
                cfg['settings']['resolution'] = {'w': int(w), 'h': int(h)}
            except Exception:
                pass
            # record path
            cfg['settings']['record_path'] = str(getattr(self, '_record_path', ''))

            print(f"[CFG] writing cfg={cfg}")
            # write file atomically
            tmp = p.with_suffix('.tmp')
            with tmp.open('w', encoding='utf-8') as fh:
                json.dump(cfg, fh, indent=2)
            tmp.replace(p)
            print(f"[CFG] saved settings to {p}")
        except Exception as e:
            print(f"[CFG] save_settings exception: {e}")
            pass

    def _on_nt_connect_clicked(self):
        team_str = self.nt_ip_edit.text().strip()
        if not team_str:
            QMessageBox.warning(self, "NetworkTables", "Please enter a team number.")
            return
        ip = self._team_to_ip(team_str)
        self._connect_networktables(ip)

    def _open_recordings_folder(self):
        """Open the configured recordings folder in the OS file manager."""
        try:
            p = Path(getattr(self, '_record_path', Path.cwd()))
            # ensure it exists
            p.mkdir(parents=True, exist_ok=True)
            path_str = str(p)
            if os.name == 'nt':
                os.startfile(path_str)
            else:
                import subprocess, sys
                if sys.platform == 'darwin':
                    subprocess.call(['open', path_str])
                else:
                    subprocess.call(['xdg-open', path_str])
        except Exception as e:
            QMessageBox.warning(self, "Open Recordings", f"Could not open folder: {e}")

    def _connect_networktables(self, ip: str):
        """Attempt to initialize NetworkTables client to the given IP and update UI state."""
        if NetworkTables is None:
            QMessageBox.warning(self, "NetworkTables", "The 'networktables' package is not available.")
            self._nt = None
            self.nt_status.setText("NT: unavailable")
            return

        try:
            NetworkTables.initialize(server=ip)
            self._nt = NetworkTables.getTable("SmartDashboard")
            self._nt_server_ip = ip
            self.nt_status.setText(f"NT: {ip} connected")
            # persist the successful IP
            try:
                self.save_settings()
            except Exception:
                pass
        except Exception as e:
            self._nt = None
            self.nt_status.setText("NT: connect failed")
            QMessageBox.warning(self, "NetworkTables", f"Could not connect to {ip}: {e}")
            return

    def stop_recording(self):
        """Stop recording and finalize file (post-roll will continue for a few seconds)."""
        # print(f"[REC] stop_recording() called; recording={getattr(self, 'recording', False)} _manual={getattr(self, '_manual', False)}")
        if not getattr(self, 'recording', False):
            return

        # set recording flag false immediately, but continue post-roll buffering
        self.recording = False
        # exit manual mode if user stopped manually
        self._manual = False
        # start post-roll: continue buffering for a few more seconds then finalize
        self._postroll_active = True
        seconds = float(getattr(self, '_postroll_seconds', 3.0))
        self._postroll_end = time.time() + seconds
        # update UI to reflect post-roll
        self.setWindowTitle(f"PyQt5 Camera Switcher - Post-roll ({seconds}s)")

    def _finalize_recording(self):
        """Internal: close writer and finalize recording."""
        # Close the active VideoWriter if open
        try:
            if self._record_writer_open and self.record_writer:
                self.record_writer.release()
                self._record_writer_open = False
                self.record_writer = None
                print(f"[REC] VideoWriter closed and released")
        except Exception as e:
            print(f"[REC] error closing VideoWriter: {e}")

        # Check if there are any buffered frames (post-roll buffer)
        try:
            with self._record_lock:
                frames = list(getattr(self, '_record_frames', []))
                self._record_frames = []
        except Exception:
            frames = list(getattr(self, '_record_frames', []))
            try:
                self._record_frames = []
            except Exception:
                pass

        # If VideoWriter was used and no buffered frames, finalize immediately
        if self._record_writer_open == False and len(frames) == 0:
            # Already written to disk, just clean up UI
            self.recording = False
            self._postroll_active = False
            self._rec_start_time = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.setWindowTitle("PyQt5 Camera Switcher")
            filename = getattr(self, '_desired_filename', f'record_{self.current_camera_index}_{int(time.time())}.mp4')
            print(f"[REC] Recording finalized (disk write): {filename}")
            # Open the file
            try:
                os.startfile(filename)
            except Exception:
                pass
            return

        # If there are buffered frames (from post-roll or fallback mode), encode them
        if len(frames) == 0:
            # No frames to write
            self.recording = False
            self._postroll_active = False
            self._rec_start_time = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.setWindowTitle("PyQt5 Camera Switcher")
            print("[REC] finalize called but no frames to write")
            return

        filename = getattr(self, '_desired_filename', f'record_{self.current_camera_index}_{int(time.time())}.mp4')
        print(f"[REC] Finalizing buffered frames in background: frames={len(frames)}, filename={filename}")

        # reset UI/state immediately
        self.recording = False
        self._postroll_active = False
        self._rec_start_time = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.setWindowTitle("PyQt5 Camera Switcher")

        # start background thread to encode & save buffered frames
        try:
            t = threading.Thread(target=self._encode_and_save, args=(frames, filename))
            t.daemon = True
            t.start()
        except Exception as e:
            print(f"[REC] failed to start background encode thread: {e}")
            # fall back to synchronous encode
            try:
                self._encode_and_save(frames, filename)
            except Exception:
                pass

    def _encode_and_save(self, frames, filename):
        """Background worker: compute fps from timestamps and write frames to MP4."""
        try:
            n = len(frames)
            if n == 0:
                print("[ENC] no frames to encode")
                return
            t0 = frames[0][0]
            t1 = frames[-1][0]
            span = t1 - t0 if t1 > t0 else 0.0
            if span > 0 and n >= 2:
                fps = (n - 1) / span
            else:
                # fallback to camera/timer fps
                try:
                    cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    cap_fps = 0.0
                timer_interval = self.timer.interval() if hasattr(self, 'timer') and self.timer is not None else 30
                timer_fps = 1000.0 / timer_interval if timer_interval > 0 else 30.0
                fps = cap_fps if cap_fps and cap_fps > 1.0 else float(round(timer_fps))

            fps = max(1.0, min(120.0, float(fps)))

            first_frame = frames[0][1]
            h, w = first_frame.shape[:2]

            print(f"[ENC] encoding {n} frames span={span:.3f} fps={fps:.2f} -> {filename}")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            try:
                writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('VideoWriter failed to open')
            except Exception as e:
                print(f"[ENC] VideoWriter open failed: {e}")
                # inform user on UI thread
                try:
                    QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Recording Error", f"Could not write recording file:\n{e}"))
                except Exception:
                    pass
                return

            for _, frm in frames:
                try:
                    writer.write(frm)
                except Exception:
                    pass

            writer.release()

            # try to open the recorded file (Windows)
            try:
                os.startfile(filename)
            except Exception:
                pass

            print(f"[ENC] done writing {filename}")
        except Exception as e:
            print(f"[ENC] exception during encode: {e}")


class SettingsDialog(QDialog):
    def __init__(self, parent=None, preroll: int = 2, postroll: int = 2, record_path: str = "", nt_bool_key: str = 'Teleop', resolution: tuple = (640, 480)):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout()

        # Pre-roll
        hl0 = QHBoxLayout()
        hl0.addWidget(QLabel("Pre-roll (s):"))
        self.preroll_spin = QSpinBox()
        self.preroll_spin.setRange(0, 30)
        self.preroll_spin.setValue(int(preroll))
        hl0.addWidget(self.preroll_spin)
        layout.addLayout(hl0)

        # Post-roll
        hl_post = QHBoxLayout()
        hl_post.addWidget(QLabel("Post-roll (s):"))
        self.postroll_spin = QSpinBox()
        self.postroll_spin.setRange(0, 60)
        self.postroll_spin.setValue(int(postroll))
        hl_post.addWidget(self.postroll_spin)
        layout.addLayout(hl_post)

        # Resolution
        hl_res = QHBoxLayout()
        hl_res.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080", "Custom"])
        hl_res.addWidget(self.res_combo)
        self.res_w_spin = QSpinBox()
        self.res_w_spin.setRange(160, 7680)
        self.res_w_spin.setValue(640)
        self.res_h_spin = QSpinBox()
        self.res_h_spin.setRange(120, 4320)
        self.res_h_spin.setValue(480)
        self.res_w_spin.setFixedWidth(90)
        self.res_h_spin.setFixedWidth(90)
        hl_res.addWidget(self.res_w_spin)
        hl_res.addWidget(self.res_h_spin)
        layout.addLayout(hl_res)

        def _on_res_combo(idx):
            txt = self.res_combo.currentText()
            if txt == '640x480':
                self.res_w_spin.setValue(640)
                self.res_h_spin.setValue(480)
                self.res_w_spin.setEnabled(False)
                self.res_h_spin.setEnabled(False)
            elif txt == '1280x720':
                self.res_w_spin.setValue(1280)
                self.res_h_spin.setValue(720)
                self.res_w_spin.setEnabled(False)
                self.res_h_spin.setEnabled(False)
            elif txt == '1920x1080':
                self.res_w_spin.setValue(1920)
                self.res_h_spin.setValue(1080)
                self.res_w_spin.setEnabled(False)
                self.res_h_spin.setEnabled(False)
            else:
                # custom
                self.res_w_spin.setEnabled(True)
                self.res_h_spin.setEnabled(True)
        self.res_combo.currentIndexChanged.connect(_on_res_combo)
        # set initial selection based on provided resolution
        try:
            rw, rh = int(record_path and 640 or 640), int(record_path and 480 or 480)
        except Exception:
            rw, rh = 640, 480
        # if caller passed 'record_path' arg in place of resolution, fall back
        if isinstance(postroll, int):
            pass
        # caller will set values after creation in open_settings_dialog

        # Record path
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Save recordings to:"))
        self.record_path_edit = QLineEdit(record_path)
        hl2.addWidget(self.record_path_edit, stretch=1)
        browse = QPushButton("Browse")
        hl2.addWidget(browse)
        layout.addLayout(hl2)

        def on_browse():
            start = Path(self.record_path_edit.text() or str(Path.home()))
            d = QFileDialog.getExistingDirectory(self, "Select folder", str(start))
            if d:
                self.record_path_edit.setText(d)
        browse.clicked.connect(on_browse)

        # NetworkTables boolean key
        hl_nt = QHBoxLayout()
        hl_nt.addWidget(QLabel("NT boolean key:"))
        self.nt_key_edit = QLineEdit(nt_bool_key)
        hl_nt.addWidget(self.nt_key_edit)
        layout.addLayout(hl_nt)

        # initialize resolution UI from provided values if available
        try:
            rw, rh = int(resolution[0]), int(resolution[1])
        except Exception:
            rw, rh = 640, 480
        # select a preset if it matches
        if rw == 640 and rh == 480:
            self.res_combo.setCurrentText('640x480')
        elif rw == 1280 and rh == 720:
            self.res_combo.setCurrentText('1280x720')
        elif rw == 1920 and rh == 1080:
            self.res_combo.setCurrentText('1920x1080')
        else:
            self.res_combo.setCurrentText('Custom')
            self.res_w_spin.setValue(rw)
            self.res_h_spin.setValue(rh)
        # ensure Custom enables fields
        if self.res_combo.currentText() == 'Custom':
            self.res_w_spin.setEnabled(True)
            self.res_h_spin.setEnabled(True)
        else:
            self.res_w_spin.setEnabled(False)
            self.res_h_spin.setEnabled(False)

        # dialog buttons + Export/Reset in a single bottom row
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        hl3 = QHBoxLayout()
        hl3.addWidget(buttons)
        hl3.addStretch()
        export_btn = QPushButton("Export")
        reset_btn = QPushButton("Reset")
        hl3.addWidget(export_btn)
        hl3.addWidget(reset_btn)
        layout.addLayout(hl3)

        def on_export():
            parent = self.parent()
            if parent is None:
                QMessageBox.warning(self, "Export", "No parent application available to export from.")
                return
            try:
                default = str(Path.home() / "camera_recorder_export.json")
                dest, _ = QFileDialog.getSaveFileName(self, "Export settings", default, "JSON Files (*.json)")
                if dest:
                    # ensure latest settings are written
                    parent.save_settings()
                    shutil = __import__('shutil')
                    shutil.copy2(str(parent._config_path), dest)
                    QMessageBox.information(self, "Export", f"Exported settings to {dest}")
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", f"Could not export settings: {e}")

        def on_reset():
            parent = self.parent()
            if parent is None:
                return
            ans = QMessageBox.question(self, "Reset Settings", "Reset settings to defaults and delete config file? This cannot be undone.", QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.Yes:
                try:
                    p = Path(parent._config_path)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
                # apply defaults
                parent._nt_server_ip = "10.0.67.1"
                parent.nt_ip_edit.setText("67")
                parent._postroll_seconds = 2
                try:
                    default_videos = Path.home() / 'Videos'
                    if default_videos.exists():
                        parent._record_path = str(default_videos)
                    else:
                        parent._record_path = str(Path.cwd())
                except Exception:
                    parent._record_path = str(Path.cwd())
                # reset UI
                try:
                    parent._update_save_path_label()
                    parent._update_config_path_label()
                except Exception:
                    pass
                try:
                    if parent.camera_combo.count() > 0:
                        parent.camera_combo.setCurrentIndex(0)
                except Exception:
                    pass
                parent.save_settings()
                QMessageBox.information(self, "Reset", "Settings reset to defaults.")

        export_btn.clicked.connect(on_export)
        reset_btn.clicked.connect(on_reset)

        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CameraApp()
    w.show()
    sys.exit(app.exec_())
