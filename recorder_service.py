"""
Headless recorder service.

- Captures frames from a camera
- Publishes frames to a local TCP JPEG stream (for the viewer)
- Starts/stops recording via:
  - Keyboard (R=start manual, S=stop manual, A=auto/NT mode, Q=quit)
  - NetworkTables boolean key (when not in manual mode)
  - Optional control socket commands (JSON lines)
"""

from __future__ import annotations

import json
import os
import sys
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any

import cv2

from Utils import RingBuffer
from camera_utils import open_capture, ensure_dir
from config_utils import load_config, get_config_path

try:
    from networktables import NetworkTables
except Exception:
    NetworkTables = None


@dataclass
class Settings:
    camera_index: int = 0
    pre_roll_seconds: int = 2
    buffer_seconds: int = 2
    output_dir: str = str(Path.cwd())
    nt_server_ip: str = "10.0.67.2"
    nt_boolean_key: str = "Teleop"

    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30

    stream_host: str = "127.0.0.1"
    stream_port: int = 8765
    control_port: int = 8766
    jpeg_quality: int = 80
    stream_max_fps: int = 30


class FrameWriter:
    def __init__(self, file_name: str, frame_width: int, frame_height: int, fps: float = 30.0):
        self.file_name = file_name
        self.frame_queue: Queue = Queue()
        self.running = True

        w = int(frame_width)
        h = int(frame_height)
        fps_f = float(fps) if fps and fps > 0 else 30.0

        # On Jetson/Linux, prefer hardware H.264 encode via GStreamer (nvv4l2h264enc).
        # This is the biggest performance win for high-res/high-fps recording.
        self.output = None
        if sys.platform.startswith("linux"):
            try:
                # Derive sane encoder parameters for the target FPS.
                iframe = int(round(fps_f)) if fps_f > 0 else 30
                if iframe < 1:
                    iframe = 1
                if iframe > 240:
                    iframe = 240

                # Rough bitrate scaling: 8Mbps at 30fps; scale with fps.
                bitrate = int(8_000_000 * (fps_f / 30.0))
                if bitrate < 2_000_000:
                    bitrate = 2_000_000
                if bitrate > 60_000_000:
                    bitrate = 60_000_000

                # Quote the location for safety (spaces etc.)
                loc = str(file_name).replace('"', "")
                pipeline = (
                    # do-timestamp is critical: without timestamps, MP4 duration can be wrong
                    # (looks like it only recorded the first buffered frames).
                    "appsrc is-live=true do-timestamp=true format=time ! "
                    f"video/x-raw,format=BGR,width={w},height={h},framerate={int(round(fps_f))}/1 ! "
                    "queue max-size-buffers=120 leaky=downstream ! "
                    "videoconvert ! "
                    f"video/x-raw,format=I420,width={w},height={h},framerate={int(round(fps_f))}/1 ! "
                    # Ensure a steady output rate (drops/duplicates if needed to match timestamps)
                    "videorate ! "
                    f"video/x-raw,format=I420,width={w},height={h},framerate={int(round(fps_f))}/1 ! "
                    f"nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 iframeinterval={iframe} bitrate={bitrate} ! "
                    "h264parse ! qtmux ! "
                    f'filesink location="{loc}" sync=false'
                )
                out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps_f, (w, h), True)
                if out is not None and out.isOpened():
                    self.output = out
                else:
                    try:
                        if out is not None:
                            out.release()
                    except Exception:
                        pass
            except Exception:
                self.output = None

        # Fallback: OpenCV software mp4v (Windows / non-GStreamer builds)
        if self.output is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.output = cv2.VideoWriter(file_name, fourcc, fps_f, (w, h))

        if self.output is None or not self.output.isOpened():
            raise RuntimeError("VideoWriter failed to open (gstreamer/mp4v).")

    def write_frames(self):
        try:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                if frame is None:
                    break
                try:
                    self.output.write(frame)
                except Exception:
                    pass
        finally:
            # IMPORTANT (avoids segfaults):
            # Only release the VideoWriter from the same thread that is writing frames.
            try:
                self.output.release()
            except Exception:
                pass

    def add_frame(self, frame):
        try:
            self.frame_queue.put(frame.copy())
        except Exception:
            self.frame_queue.put(frame)

    def stop(self):
        self.running = False
        try:
            self.frame_queue.put(None)
        except Exception:
            pass


class VideoStreamServer:
    """
    Single-client TCP server sending frames as:
      [4-byte big-endian length][jpeg bytes]
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self._sock = None
        self._client = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        with self._lock:
            try:
                if self._client:
                    self._client.close()
            except Exception:
                pass
            self._client = None
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(1)
        s.settimeout(1.0)
        with self._lock:
            self._sock = s
        while self._running:
            try:
                conn, _addr = s.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    # replace any existing client
                    try:
                        if self._client:
                            self._client.close()
                    except Exception:
                        pass
                    self._client = conn
            except socket.timeout:
                continue
            except Exception:
                continue

    def send_jpeg(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            c = self._client
        if not c:
            return
        try:
            header = struct.pack(">I", len(jpeg_bytes))
            c.sendall(header + jpeg_bytes)
        except Exception:
            try:
                c.close()
            except Exception:
                pass
            with self._lock:
                if self._client is c:
                    self._client = None

    def has_client(self) -> bool:
        with self._lock:
            return self._client is not None


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        self.settings = Settings()
        self._config_path = get_config_path()
        self._cfg_mtime = 0.0

        self.cap = None
        self.sd = None  # SmartDashboard table

        self.manual_mode = False
        # When in manual_mode, this is the requested recording state.
        self.manual_target = False
        self.recording = False
        self.buffering = False
        self.buffer_until = 0.0
        # overlay timer start (reset on recording start)
        self.overlay_start_time = time.time()

        self.ring_buffer = RingBuffer(max(1, int(self.settings.pre_roll_seconds * max(1, self.settings.capture_fps))))

        self.writer: FrameWriter | None = None
        self.writer_thread: threading.Thread | None = None
        self.current_recording_path: str | None = None

        self.stream_server = VideoStreamServer(self.settings.stream_host, self.settings.stream_port)

        # Capture FPS estimate (used so recorded videos don't play too fast/slow)
        self._last_frame_ts: float | None = None
        self._fps_samples: deque[float] = deque(maxlen=60)
        self._fps_estimate: float = 30.0

    def _update_fps_estimate(self, ts: float) -> None:
        try:
            last = self._last_frame_ts
            self._last_frame_ts = ts
            if last is None:
                return
            dt = ts - last
            # ignore crazy deltas (camera reconnect, stalls, etc.)
            if dt <= 0.0 or dt > 1.0:
                return
            fps = 1.0 / dt
            if fps < 1.0 or fps > 240.0:
                return
            self._fps_samples.append(float(fps))
            if self._fps_samples:
                self._fps_estimate = sum(self._fps_samples) / float(len(self._fps_samples))
        except Exception:
            return

    def _record_fps(self, cap) -> float:
        # IMPORTANT:
        # On Linux (especially with GStreamer/V4L2), CAP_PROP_FPS is often wrong (commonly reports 30),
        # and using the "requested" FPS can make playback faster than realtime if the camera can't sustain it.
        # The best approximation is the *measured* capture rate from recent frames.

        # 1) If we have enough samples, prefer the measured FPS.
        try:
            samples = getattr(self, "_fps_samples", None)
            est = float(getattr(self, "_fps_estimate", 0.0))
            if samples is not None and len(samples) >= 15 and 5.0 <= est <= 240.0:
                return est
        except Exception:
            pass

        # 2) Try the camera-reported FPS (may be unreliable).
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if 5.0 <= fps <= 240.0:
                return fps
        except Exception:
            pass

        # 3) Fall back to the requested capture FPS from settings, if present.
        try:
            fps = float(getattr(self.settings, "capture_fps", 30))
            if 5.0 <= fps <= 240.0:
                return fps
        except Exception:
            pass

        # 4) Last resort.
        return 30.0

    def load_settings_from_config(self) -> Settings:
        cfg = load_config()
        settings_block = cfg.get("settings", {}) if isinstance(cfg.get("settings"), dict) else {}
        nt_block = cfg.get("nt", {}) if isinstance(cfg.get("nt"), dict) else {}
        cam_block = cfg.get("camera", {}) if isinstance(cfg.get("camera"), dict) else {}

        s = Settings()
        try:
            s.pre_roll_seconds = int(settings_block.get("pre_roll_seconds", s.pre_roll_seconds))
        except Exception:
            pass
        try:
            s.buffer_seconds = int(settings_block.get("buffer_seconds", s.buffer_seconds))
        except Exception:
            pass
        try:
            out_dir = settings_block.get("output_dir", s.output_dir)
            if out_dir:
                s.output_dir = str(out_dir)
        except Exception:
            pass
        try:
            s.nt_server_ip = str(nt_block.get("server_ip", s.nt_server_ip))
        except Exception:
            pass
        try:
            s.nt_boolean_key = str(nt_block.get("boolean_key", s.nt_boolean_key)) or "Teleop"
        except Exception:
            s.nt_boolean_key = "Teleop"
        try:
            s.camera_index = int(cam_block.get("selected_index", s.camera_index))
        except Exception:
            pass

        # capture resolution (uses legacy "resolution" block if present)
        try:
            res = settings_block.get("resolution", {}) if isinstance(settings_block.get("resolution"), dict) else {}
            w = int(res.get("w", s.capture_width))
            h = int(res.get("h", s.capture_height))
            if w > 0 and h > 0:
                s.capture_width = w
                s.capture_height = h
        except Exception:
            pass

        # capture fps (used for requesting camera mode and pre-roll sizing)
        try:
            fps = int(float(settings_block.get("capture_fps", s.capture_fps)))
            if fps < 1:
                fps = 1
            if fps > 240:
                fps = 240
            s.capture_fps = fps
        except Exception:
            pass

        # stream throttle (reduce CPU usage; only affects stream encoding rate)
        try:
            stream_cfg = cfg.get("stream", {}) if isinstance(cfg.get("stream"), dict) else {}
            s.stream_max_fps = int(stream_cfg.get("max_fps", s.stream_max_fps))
            if s.stream_max_fps < 1:
                s.stream_max_fps = 1
            if s.stream_max_fps > 120:
                s.stream_max_fps = 120
        except Exception:
            pass

        # optional stream overrides (safe defaults)
        try:
            s.stream_host = str(cfg.get("stream", {}).get("host", s.stream_host))
            s.stream_port = int(cfg.get("stream", {}).get("port", s.stream_port))
        except Exception:
            pass
        # optional control overrides
        try:
            ctrl = cfg.get("control", {}) if isinstance(cfg.get("control"), dict) else {}
            s.control_port = int(ctrl.get("port", s.control_port))
        except Exception:
            pass

        return s

    def _maybe_reload_config(self):
        try:
            mtime = self._config_path.stat().st_mtime
        except Exception:
            mtime = 0.0
        if mtime <= self._cfg_mtime:
            return
        self._cfg_mtime = mtime

        new_settings = self.load_settings_from_config()
        with self.lock:
            old = self.settings
            self.settings = new_settings
            # update ring buffer size if needed
            try:
                new_size = max(1, int(self.settings.pre_roll_seconds * max(1, self.settings.capture_fps)))
                if new_size != self.ring_buffer.size:
                    old_frames = self.ring_buffer.get_frames()
                    self.ring_buffer = RingBuffer(new_size)
                    for fr in old_frames[-new_size:]:
                        if fr is not None:
                            self.ring_buffer.add_frame(fr)
            except Exception:
                pass

        # reconnect NT if needed
        if old.nt_server_ip != new_settings.nt_server_ip:
            self._connect_nt(new_settings.nt_server_ip)

        # switch camera if needed (only if safe)
        if old.camera_index != new_settings.camera_index:
            self.switch_camera(new_settings.camera_index)

    def _connect_nt(self, ip: str):
        if NetworkTables is None:
            self.sd = None
            return
        try:
            if hasattr(NetworkTables, "shutdown"):
                try:
                    NetworkTables.shutdown()
                except Exception:
                    pass
            NetworkTables.initialize(server=str(ip))
            self.sd = NetworkTables.getTable("SmartDashboard")
            print(f"[NT] connected to {ip}")
        except Exception as e:
            self.sd = None
            print(f"[NT] connect failed: {e}")

    def switch_camera(self, camera_index: int) -> bool:
        with self.lock:
            if self.recording or self.buffering or self.writer is not None:
                print("[CAM] cannot switch while recording/buffering")
                return False

        new_cap = open_capture(
            camera_index,
            width=self.settings.capture_width,
            height=self.settings.capture_height,
            fps=self.settings.capture_fps,
        )
        if new_cap is None:
            print(f"[CAM] failed to open camera {camera_index}")
            return False

        with self.lock:
            old = self.cap
            self.cap = new_cap
            self.settings.camera_index = int(camera_index)
            # reset ring buffer for new camera
            try:
                self.ring_buffer = RingBuffer(max(1, int(self.settings.pre_roll_seconds * max(1, self.settings.capture_fps))))
            except Exception:
                pass

        try:
            if old is not None:
                old.release()
        except Exception:
            pass

        print(f"[CAM] switched to camera {camera_index}")
        return True

    def _start_recording(self):
        if self.writer is not None:
            return
        ensure_dir(self.settings.output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = str(Path(self.settings.output_dir) / f"record_{self.settings.camera_index}_{ts}.mp4")
        self.current_recording_path = file_path

        # get frame shape by peeking last frame
        with self.lock:
            frames = [f for f in self.ring_buffer.get_frames() if f is not None]
            last = frames[-1] if frames else None
        if last is None:
            # will create writer on first frame
            self.writer = None
            return
        h, w = last.shape[:2]
        try:
            with self.lock:
                cap = self.cap
            fps = self._record_fps(cap) if cap is not None else 30.0
            self.writer = FrameWriter(file_path, w, h, fps=float(fps))
        except Exception as e:
            print(f"[REC] failed to start writer: {e}")
            self.writer = None
            return
        self.writer_thread = threading.Thread(target=self.writer.write_frames, daemon=False)
        self.writer_thread.start()
        # write pre-roll frames
        for fr in frames:
            self.writer.add_frame(fr)
        print(f"[REC] start -> {file_path} (pre-roll={len(frames)} frames)")

    def _stop_writer(self):
        w = self.writer
        t = self.writer_thread
        self.writer = None
        self.writer_thread = None
        if w:
            try:
                w.stop()
            except Exception:
                pass
        if t:
            try:
                # Give the writer thread time to flush and release safely.
                # (Releasing from this thread can segfault when using GStreamer/Jetson encoders.)
                t.join()
            except Exception:
                pass

    def _finalize_recording(self):
        self._stop_writer()
        path = self.current_recording_path
        self.current_recording_path = None
        print(f"[REC] finalized -> {path}" if path else "[REC] finalized")

        # Auto-open the file after post-roll finalization
        if not path:
            return
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                import subprocess
                import sys

                if sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    def _handle_transition(self, desired_recording: bool):
        with self.lock:
            was_recording = self.recording

        if desired_recording and not was_recording:
            with self.lock:
                self.recording = True
                self.buffering = False
                self.overlay_start_time = time.time()
            self._start_recording()
            return

        if not desired_recording and was_recording:
            with self.lock:
                self.recording = False
                self.buffering = True
                self.buffer_until = time.time() + float(self.settings.buffer_seconds)
            print(f"[REC] stop -> buffering {self.settings.buffer_seconds}s")
            return

    def _desired_recording_state(self) -> bool:
        with self.lock:
            manual = self.manual_mode
            manual_target = self.manual_target
            key = self.settings.nt_boolean_key

        if manual:
            return bool(manual_target)

        # auto: NT controls recording
        if self.sd is None:
            return False
        try:
            return bool(self.sd.getBoolean(key, False))
        except Exception:
            return False

    def _capture_loop(self):
        self.stream_server.start()
        last_stream_ts = 0.0

        while self.running:
            self._maybe_reload_config()

            with self.lock:
                cap = self.cap
                writer = self.writer
                jpeg_quality = int(self.settings.jpeg_quality)
                overlay_t0 = float(getattr(self, "overlay_start_time", time.time()))

            if cap is None:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # update capture FPS estimate (used for correct recording speed)
            # perf_counter is monotonic/high-resolution (more stable for FPS calculation)
            try:
                self._update_fps_estimate(time.perf_counter())
            except Exception:
                pass

            # state transitions (compute desired first so we can overlay it)
            desired = self._desired_recording_state()
            self._handle_transition(desired)

            # overlay True/False + timer on the outgoing frame
            try:
                now = time.time()
                elapsed = now - overlay_t0
                cv2.putText(
                    frame,
                    str(bool(desired)),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{elapsed:.1f}s",
                    (500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            # publish frame to viewer
            try:
                if self.stream_server.has_client():
                    now = time.time()
                    max_fps = float(max(1, int(getattr(self.settings, "stream_max_fps", 30))))
                    if (now - last_stream_ts) >= (1.0 / max_fps):
                        last_stream_ts = now
                        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if ok:
                            self.stream_server.send_jpeg(enc.tobytes())
            except Exception:
                pass

            # write frame
            with self.lock:
                writer = self.writer
                recording = self.recording
                buffering = self.buffering

            if recording or buffering:
                if writer is None:
                    # create writer on-demand using this frame
                    ensure_dir(self.settings.output_dir)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = str(Path(self.settings.output_dir) / f"record_{self.settings.camera_index}_{ts}.mp4")
                    # remember the current output path for auto-open on finalize
                    with self.lock:
                        self.current_recording_path = file_path
                    h, w = frame.shape[:2]
                    try:
                        fps = self._record_fps(cap)
                        wtr = FrameWriter(file_path, w, h, fps=float(fps))
                    except Exception as e:
                        print(f"[REC] failed to create writer: {e}")
                        # can't record; drop back to non-recording (but keep streaming preview)
                        with self.lock:
                            self.recording = False
                            self.buffering = False
                        continue
                    th = threading.Thread(target=wtr.write_frames, daemon=False)
                    th.start()
                    # write pre-roll
                    with self.lock:
                        pre = [f for f in self.ring_buffer.get_frames() if f is not None]
                        self.writer = wtr
                        self.writer_thread = th
                    for fr in pre:
                        wtr.add_frame(fr)
                    print(f"[REC] writer created -> {file_path}")
                    writer = wtr

                try:
                    writer.add_frame(frame)
                except Exception:
                    pass
            else:
                # maintain pre-roll ring buffer
                try:
                    with self.lock:
                        self.ring_buffer.add_frame(frame.copy())
                except Exception:
                    pass

            # finalize buffering if time elapsed
            try:
                with self.lock:
                    buffering_now = bool(self.buffering)
                    buffer_until_now = float(self.buffer_until)
                if buffering_now and time.time() >= buffer_until_now:
                    with self.lock:
                        self.buffering = False
                    self._finalize_recording()
            except Exception:
                pass

        self.stream_server.stop()

    def _control_server(self):
        host = self.settings.stream_host
        port = int(self.settings.control_port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        s.settimeout(1.0)
        print(f"[CTL] listening on {host}:{port}")

        while self.running:
            try:
                conn, _addr = s.accept()
                conn.settimeout(1.0)
                buf = b""
                while self.running:
                    try:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            try:
                                msg = json.loads(line.decode("utf-8"))
                            except Exception:
                                continue
                            self._handle_control_msg(msg)
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            except socket.timeout:
                continue
            except Exception:
                continue

        try:
            s.close()
        except Exception:
            pass

    def _handle_control_msg(self, msg: dict[str, Any]):
        cmd = str(msg.get("cmd", "")).lower()
        if cmd == "start":
            with self.lock:
                self.manual_mode = True
                self.manual_target = True
            print("[CTL] start (manual)")
        elif cmd == "stop":
            with self.lock:
                self.manual_mode = True
                self.manual_target = False
            print("[CTL] stop (manual)")
        elif cmd == "auto":
            with self.lock:
                self.manual_mode = False
                self.manual_target = False
            print("[CTL] auto (NT)")
        elif cmd == "switch_camera":
            try:
                idx = int(msg.get("index"))
                self.switch_camera(idx)
            except Exception:
                pass
        elif cmd == "quit":
            print("[CTL] quit")
            self.running = False

    def start(self):
        # initial config
        self.settings = self.load_settings_from_config()
        self._cfg_mtime = 0.0
        # (re)create stream server using configured host/port
        try:
            self.stream_server = VideoStreamServer(self.settings.stream_host, self.settings.stream_port)
        except Exception:
            pass

        # init camera
        self.cap = open_capture(
            self.settings.camera_index,
            width=self.settings.capture_width,
            height=self.settings.capture_height,
            fps=self.settings.capture_fps,
        )
        if self.cap is None:
            self.cap = open_capture(0, width=self.settings.capture_width, height=self.settings.capture_height, fps=self.settings.capture_fps)
            self.settings.camera_index = 0
        if self.cap is None:
            raise RuntimeError("Could not open any camera.")

        # NT
        self._connect_nt(self.settings.nt_server_ip)

        # threads
        t_cap = threading.Thread(target=self._capture_loop, daemon=False)
        t_ctl = threading.Thread(target=self._control_server, daemon=True)
        t_cap.start()
        t_ctl.start()

        # keyboard loop (Windows)
        self._keyboard_loop()

        # shutdown
        self.running = False
        try:
            t_cap.join(timeout=2)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            self._stop_writer()
        except Exception:
            pass

    def _keyboard_loop(self):
        print("[KEY] R=start manual, S=stop manual, A=auto/NT, Q=quit")
        if os.name != "nt":
            # simple sleep loop on non-windows
            while self.running:
                time.sleep(0.25)
            return

        try:
            import msvcrt
        except Exception:
            while self.running:
                time.sleep(0.25)
            return

        while self.running:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    try:
                        c = ch.decode("utf-8", errors="ignore").lower()
                    except Exception:
                        c = ""
                    if c == "r":
                        with self.lock:
                            self.manual_mode = True
                            self.manual_target = True
                        print("[KEY] start (manual)")
                    elif c == "s":
                        with self.lock:
                            self.manual_mode = True
                            self.manual_target = False
                        print("[KEY] stop (manual)")
                    elif c == "a":
                        with self.lock:
                            self.manual_mode = False
                            self.manual_target = False
                        print("[KEY] auto (NT)")
                    elif c == "q":
                        print("[KEY] quit")
                        self.running = False
                        break
                time.sleep(0.01)
            except Exception:
                time.sleep(0.05)


def main():
    r = Recorder()
    r.start()


if __name__ == "__main__":
    main()

