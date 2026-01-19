"""
MJPEG web bridge for the headless recorder.

It connects to recorder_service.py's TCP JPEG stream (length-prefixed JPEG frames),
and exposes it to browsers as MJPEG over HTTP.

Endpoints:
  - /         simple HTML page with <img src="/mjpeg">
  - /mjpeg    multipart/x-mixed-replace MJPEG stream
  - /health   simple health check

Defaults:
  - Connect to recorder stream at 127.0.0.1:8765
  - Serve HTTP on 0.0.0.0:8080
"""

from __future__ import annotations

import html
import os
from pathlib import Path
import socket
import struct
import subprocess
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse, quote, unquote

from camera_utils import enumerate_camera_choices
from config_utils import load_config, save_config


class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._ts: float = 0.0

    def set(self, jpeg: bytes):
        with self._lock:
            self._jpeg = jpeg
            self._ts = time.time()

    def get(self) -> tuple[Optional[bytes], float]:
        with self._lock:
            return self._jpeg, self._ts


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def stream_reader(store: FrameStore, host: str, port: int):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((host, int(port)))
            s.settimeout(5.0)
            print(f"[WEB] connected to recorder stream {host}:{port}")

            while True:
                hdr = _recv_exact(s, 4)
                if hdr is None:
                    raise RuntimeError("stream disconnected")
                (length,) = struct.unpack(">I", hdr)
                if length <= 0 or length > 20_000_000:
                    raise RuntimeError(f"bad frame length: {length}")
                payload = _recv_exact(s, length)
                if payload is None:
                    raise RuntimeError("stream disconnected")
                store.set(payload)
        except Exception as e:
            print(f"[WEB] stream reconnecting: {e}")
            try:
                s.close()
            except Exception:
                pass
            time.sleep(0.5)


def _control_endpoint_from_config() -> tuple[str, int]:
    cfg = load_config()
    stream = cfg.get("stream", {}) if isinstance(cfg.get("stream"), dict) else {}
    ctrl = cfg.get("control", {}) if isinstance(cfg.get("control"), dict) else {}
    host = str(ctrl.get("host", stream.get("host", "127.0.0.1")))
    port = int(ctrl.get("port", 8766))
    return host, port


def _send_control(cmd: dict) -> bool:
    host, port = _control_endpoint_from_config()
    try:
        import json as _json

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((host, int(port)))
        s.sendall((_json.dumps(cmd) + "\n").encode("utf-8"))
        s.close()
        return True
    except Exception:
        return False


def _get_cfg_blocks() -> tuple[dict, dict, dict, dict]:
    cfg = load_config()
    settings = cfg.get("settings", {}) if isinstance(cfg.get("settings"), dict) else {}
    nt = cfg.get("nt", {}) if isinstance(cfg.get("nt"), dict) else {}
    camera = cfg.get("camera", {}) if isinstance(cfg.get("camera"), dict) else {}
    return cfg, settings, nt, camera


def _recordings_dir_from_config() -> Path:
    cfg = load_config()
    settings = cfg.get("settings", {}) if isinstance(cfg.get("settings"), dict) else {}
    out_dir = settings.get("output_dir") or settings.get("record_path")
    if not out_dir:
        out_dir = str(Path.cwd() / "output")
    try:
        p = Path(str(out_dir))
    except Exception:
        p = Path.cwd() / "output"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _list_recordings(dir_path: Path) -> list[Path]:
    try:
        # list master recordings; ignore generated web copies
        files = [
            p
            for p in dir_path.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".mp4"
            and not p.name.lower().endswith("_web.mp4")
        ]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files
    except Exception:
        return []


_TRANSCODE_LOCK = threading.Lock()
_TRANSCODE_IN_PROGRESS: set[str] = set()


def _web_copy_path(master: Path) -> Path:
    return master.with_name(master.stem + "_web.mp4")


def _ffmpeg_config() -> tuple[str, str, int]:
    """
    Returns (ffmpeg_path, preset, crf).
    """
    cfg = load_config()
    wt = cfg.get("web_transcode", {}) if isinstance(cfg.get("web_transcode"), dict) else {}
    ffmpeg_path = str(wt.get("ffmpeg_path", "ffmpeg"))
    preset = str(wt.get("preset", "veryfast"))
    try:
        crf = int(wt.get("crf", 23))
    except Exception:
        crf = 23
    return ffmpeg_path, preset, crf


def _ffmpeg_available(ffmpeg_path: str) -> bool:
    try:
        if shutil.which(ffmpeg_path):
            return True
        return Path(ffmpeg_path).exists()
    except Exception:
        return False


def _ensure_web_copy_async(master: Path) -> tuple[Optional[Path], str]:
    """
    Ensure a browser-playable H.264 MP4 exists.

    Returns (web_path if ready else None, status_message).
    """
    web = _web_copy_path(master)
    try:
        if web.exists() and web.stat().st_mtime >= master.stat().st_mtime:
            return web, "ready"
    except Exception:
        pass

    ffmpeg_path, preset, crf = _ffmpeg_config()
    if not _ffmpeg_available(ffmpeg_path):
        return None, "ffmpeg_not_found"

    key = str(master)
    with _TRANSCODE_LOCK:
        if key in _TRANSCODE_IN_PROGRESS:
            return None, "transcoding"
        _TRANSCODE_IN_PROGRESS.add(key)

    def _worker():
        try:
            # H.264 + faststart for browser playback; no audio
            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                str(master),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-preset",
                preset,
                "-crf",
                str(crf),
                str(web),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        finally:
            with _TRANSCODE_LOCK:
                _TRANSCODE_IN_PROGRESS.discard(key)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return None, "transcoding_started"


def _safe_join(base: Path, name: str) -> Optional[Path]:
    # prevent path traversal
    try:
        name = name.replace("\\", "/")
        if "/" in name or name.startswith("."):
            return None
        p = (base / name).resolve()
        base_r = base.resolve()
        if base_r not in p.parents and p != base_r:
            return None
        return p
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    # set by main()
    store: FrameStore = None  # type: ignore[assignment]

    def _send_headers(self, code: int, content_type: str):
        self.send_response(code)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def _redirect(self, location: str):
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _read_form(self) -> dict[str, str]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        body = b""
        if length > 0:
            body = self.rfile.read(length)
        try:
            parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
            return {k: (v[0] if v else "") for k, v in parsed.items()}
        except Exception:
            return {}

    def _send_file_with_range(self, path: Path, content_type: str):
        try:
            size = path.stat().st_size
        except Exception:
            self._send_headers(404, "text/plain; charset=utf-8")
            self.wfile.write(b"not found\n")
            return

        rng = self.headers.get("Range")
        start = 0
        end = size - 1
        status = 200

        if rng and rng.startswith("bytes="):
            try:
                spec = rng.split("=", 1)[1].strip()
                a, b = spec.split("-", 1)
                if a:
                    start = int(a)
                if b:
                    end = int(b)
                if start < 0:
                    start = 0
                if end >= size:
                    end = size - 1
                if start <= end:
                    status = 206
            except Exception:
                start = 0
                end = size - 1
                status = 200

        length = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            return

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)

        if u.path == "/" or u.path.startswith("/?"):
            self._send_headers(200, "text/html; charset=utf-8")
            qs = parse_qs(u.query)
            msg = (qs.get("msg", [""])[0] or "").strip()
            msg_html = f"<p style='color:#8f8'>{html.escape(msg)}</p>" if msg else ""
            # include a small recordings list at bottom of the page
            rec_dir = _recordings_dir_from_config()
            rec_files = _list_recordings(rec_dir)[:25]
            self.wfile.write(
                b"<!doctype html><html><head><meta charset='utf-8'>"
                b"<title>Camera Stream</title>"
                b"<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0}"
                b".wrap{padding:16px}img{max-width:100%;height:auto;border:1px solid #333}"
                b"button{margin-right:8px;margin-bottom:8px;padding:8px 12px}"
                b"a{color:#7af}li{margin:6px 0}"
                b"</style>"
                b"</head><body><div class='wrap'>"
                b"<h2>Camera Stream</h2>"
                b"<p><a href='/settings'>Settings</a></p>"
                b"<form method='post' action='/control'>"
                b"<button name='cmd' value='start' type='submit'>Start (manual)</button>"
                b"<button name='cmd' value='stop' type='submit'>Stop (manual)</button>"
                b"<button name='cmd' value='auto' type='submit'>Auto (NetworkTables)</button>"
                b"</form>"
                + msg_html.encode("utf-8") +
                b"<p>If you see a broken image, start <code>recorder_service.py</code> first.</p>"
                b"<img src='/mjpeg' />"
            )
            self.wfile.write(b"<h2 style='margin-top:24px'>Recordings</h2>")
            self.wfile.write(b"<p><a href='/recordings'>View all recordings</a></p>")
            if not rec_files:
                self.wfile.write(b"<p>No recordings found.</p>")
            else:
                self.wfile.write(b"<ul>")
                for p in rec_files:
                    name = p.name
                    display = html.escape(name)
                    q = quote(name)
                    self.wfile.write(
                        f"<li><a href='/play?f={q}'>Play</a> | <a href='/download?f={q}'>Download</a> — {display}</li>".encode(
                            "utf-8"
                        )
                    )
                self.wfile.write(b"</ul>")
            self.wfile.write(b"</div></body></html>")
            return

        if u.path == "/health":
            jpeg, ts = self.store.get()
            age = time.time() - ts if ts else 1e9
            ok = jpeg is not None and age < 5.0
            self._send_headers(200 if ok else 503, "text/plain; charset=utf-8")
            self.wfile.write(f"ok={ok} age={age:.2f}s\n".encode("utf-8"))
            return

        if u.path == "/settings":
            _cfg, settings, nt, cam = _get_cfg_blocks()
            choices = enumerate_camera_choices()
            selected = str(cam.get("selected_index", "")) if isinstance(cam, dict) else ""
            pre = str(settings.get("pre_roll_seconds", 2))
            buf = str(settings.get("buffer_seconds", 2))
            out_dir = str(settings.get("output_dir", settings.get("record_path", "")))
            nt_ip = str(nt.get("server_ip", "10.0.67.2"))
            nt_key = str(nt.get("boolean_key", "Teleop"))
            res = settings.get("resolution", {}) if isinstance(settings.get("resolution"), dict) else {}
            res_w = str(res.get("w", 1280))
            res_h = str(res.get("h", 720))
            ffmpeg_path, preset, crf = _ffmpeg_config()
            ff_ok = _ffmpeg_available(ffmpeg_path)

            self._send_headers(200, "text/html; charset=utf-8")
            self.wfile.write(
                b"<!doctype html><html><head><meta charset='utf-8'>"
                b"<title>Settings</title>"
                b"<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0}"
                b".wrap{padding:16px}label{display:block;margin-top:10px}"
                b"input,select{width:420px;max-width:95%;padding:6px;margin-top:4px}"
                b"button{margin-top:14px;padding:8px 12px}"
                b"a{color:#7af}</style>"
                b"</head><body><div class='wrap'>"
                b"<h2>Settings</h2>"
                b"<p><a href='/'>Back</a></p>"
                b"<form method='post' action='/settings'>"
            )

            self.wfile.write(b"<label>Camera</label><select name='camera_index'>")
            if not choices:
                self.wfile.write(b"<option value=''>No cameras found</option>")
            for label, idx in choices:
                sel = " selected" if str(idx) == selected else ""
                self.wfile.write(f"<option value='{idx}'{sel}>{html.escape(label)}</option>".encode("utf-8"))
            self.wfile.write(b"</select>")

            def _inp(label: str, name: str, value: str):
                self.wfile.write(f"<label>{html.escape(label)}</label>".encode("utf-8"))
                self.wfile.write(f"<input name='{html.escape(name)}' value='{html.escape(value)}' />".encode("utf-8"))

            _inp("Pre-roll seconds", "pre_roll_seconds", pre)
            _inp("Buffer seconds", "buffer_seconds", buf)
            _inp("Output directory", "output_dir", out_dir)
            _inp("NT server IP", "nt_server_ip", nt_ip)
            _inp("NT boolean key", "nt_boolean_key", nt_key)
            _inp("Resolution width", "resolution_w", res_w)
            _inp("Resolution height", "resolution_h", res_h)
            _inp("FFmpeg path (ffmpeg.exe)", "ffmpeg_path", ffmpeg_path)
            _inp("FFmpeg preset", "ffmpeg_preset", preset)
            _inp("FFmpeg crf", "ffmpeg_crf", str(crf))

            if ff_ok:
                self.wfile.write(b"<p style='color:#8f8'>FFmpeg: found</p>")
            else:
                self.wfile.write(b"<p style='color:#f88'>FFmpeg: NOT found (set path above)</p>")

            self.wfile.write(b"<button type='submit'>Save</button>")
            self.wfile.write(b"</form></div></body></html>")
            return

        if u.path == "/recordings":
            rec_dir = _recordings_dir_from_config()
            files = _list_recordings(rec_dir)
            self._send_headers(200, "text/html; charset=utf-8")
            self.wfile.write(
                b"<!doctype html><html><head><meta charset='utf-8'>"
                b"<title>Recordings</title>"
                b"<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0}"
                b".wrap{padding:16px}a{color:#7af}li{margin:6px 0}</style>"
                b"</head><body><div class='wrap'>"
                b"<h2>Recordings</h2>"
                b"<p><a href='/'>Back</a></p>"
            )
            if not files:
                self.wfile.write(b"<p>No recordings found.</p></div></body></html>")
                return
            self.wfile.write(b"<ul>")
            for p in files[:250]:
                name = p.name
                display = html.escape(name)
                q = quote(name)
                self.wfile.write(
                    f"<li><a href='/play?f={q}'>Play</a> | <a href='/download?f={q}'>Download</a> — {display}</li>".encode(
                        "utf-8"
                    )
                )
            self.wfile.write(b"</ul></div></body></html>")
            return

        if u.path == "/download":
            qs = parse_qs(u.query)
            f = (qs.get("f", [""])[0] or "").strip()
            rec_dir = _recordings_dir_from_config()
            p = _safe_join(rec_dir, unquote(f))
            if not p or not p.exists():
                self._send_headers(404, "text/plain; charset=utf-8")
                self.wfile.write(b"not found\n")
                return
            self._send_file_with_range(p, "video/mp4")
            return

        if u.path == "/video":
            qs = parse_qs(u.query)
            f = (qs.get("f", [""])[0] or "").strip()
            rec_dir = _recordings_dir_from_config()
            master = _safe_join(rec_dir, unquote(f))
            if not master or not master.exists():
                self._send_headers(404, "text/plain; charset=utf-8")
                self.wfile.write(b"not found\n")
                return
            web = _web_copy_path(master)
            if not web.exists():
                self._send_headers(404, "text/plain; charset=utf-8")
                self.wfile.write(b"web copy not ready\n")
                return
            self._send_file_with_range(web, "video/mp4")
            return

        if u.path == "/play":
            qs = parse_qs(u.query)
            f = (qs.get("f", [""])[0] or "").strip()
            rec_dir = _recordings_dir_from_config()
            p = _safe_join(rec_dir, unquote(f))
            if not p or not p.exists():
                self._send_headers(404, "text/plain; charset=utf-8")
                self.wfile.write(b"not found\n")
                return
            name = p.name
            display = html.escape(name)
            q = quote(name)

            web, status = _ensure_web_copy_async(p)
            self._send_headers(200, "text/html; charset=utf-8")
            head = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Play</title>"
                "<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0}"
                ".wrap{padding:16px}a{color:#7af}"
                "video{max-width:100%;height:auto;border:1px solid #333}</style>"
            )
            if web is None and status in ("transcoding_started", "transcoding"):
                head += "<meta http-equiv='refresh' content='2'>"
            head += "</head><body><div class='wrap'>"
            self.wfile.write(head.encode("utf-8"))
            self.wfile.write(f"<h2>{display}</h2>".encode("utf-8"))
            self.wfile.write(b"<p><a href='/recordings'>Back to recordings</a></p>")
            self.wfile.write(b"<h3>Browser video player</h3>")

            if web is not None and web.exists():
                self.wfile.write(f"<video controls autoplay preload='metadata' src='/video?f={q}'></video>".encode("utf-8"))
                self.wfile.write(b"<p>If it doesn't start, try reloading the page.</p>")
            else:
                if status == "ffmpeg_not_found":
                    ffmpeg_path, _preset, _crf = _ffmpeg_config()
                    self.wfile.write(
                        b"<p><b>Cannot play in browser yet:</b> this recording is not in a browser-compatible codec.</p>"
                        b"<p>Install FFmpeg and restart <code>web_stream.py</code>, then reload this page.</p>"
                        b"<p>Download FFmpeg and add it to PATH, or set <code>web_transcode.ffmpeg_path</code> in config.json.</p>"
                    )
                    self.wfile.write(f"<p>Current ffmpeg_path: <code>{html.escape(ffmpeg_path)}</code></p>".encode("utf-8"))
                    self.wfile.write(b"<p>Tip: open <a href='/settings'>Settings</a> and set the full path to ffmpeg.exe.</p>")
                else:
                    self.wfile.write(
                        b"<p>Creating a browser-compatible copy now... please wait a few seconds.</p>"
                        b"<p>This page will auto-refresh.</p>"
                    )

                self.wfile.write(f"<p><a href='/download?f={q}'>Download original file</a></p>".encode("utf-8"))

            self.wfile.write(b"</div></body></html>")
            return

        if u.path == "/mjpeg":
            boundary = "frame"
            self._send_headers(200, f"multipart/x-mixed-replace; boundary={boundary}")
            last_ts = 0.0
            try:
                while True:
                    jpeg, ts = self.store.get()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    if ts == last_ts:
                        time.sleep(0.01)
                        continue
                    last_ts = ts

                    self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    try:
                        self.wfile.flush()
                    except Exception:
                        pass
            except Exception:
                return

        self._send_headers(404, "text/plain; charset=utf-8")
        self.wfile.write(b"not found\n")

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        form = self._read_form()

        if u.path == "/control":
            cmd = (form.get("cmd") or "").strip().lower()
            ok = False
            if cmd in ("start", "stop", "auto"):
                ok = _send_control({"cmd": cmd})
            msg = "Sent command" if ok else "Recorder not reachable"
            self._redirect("/?msg=" + msg.replace(" ", "%20"))
            return

        if u.path == "/settings":
            cfg = load_config()
            cfg.setdefault("settings", {})
            cfg.setdefault("nt", {})
            cfg.setdefault("camera", {})
            cfg.setdefault("web_transcode", {})

            pre = form.get("pre_roll_seconds", "").strip()
            buf = form.get("buffer_seconds", "").strip()
            out_dir = form.get("output_dir", "").strip()
            nt_ip = form.get("nt_server_ip", "").strip()
            nt_key = form.get("nt_boolean_key", "").strip()
            cam_idx = form.get("camera_index", "").strip()
            rw = form.get("resolution_w", "").strip()
            rh = form.get("resolution_h", "").strip()
            ffmpeg_path = form.get("ffmpeg_path", "").strip()
            ffmpeg_preset = form.get("ffmpeg_preset", "").strip()
            ffmpeg_crf = form.get("ffmpeg_crf", "").strip()

            try:
                if pre != "":
                    cfg["settings"]["pre_roll_seconds"] = int(float(pre))
            except Exception:
                pass
            try:
                if buf != "":
                    cfg["settings"]["buffer_seconds"] = int(float(buf))
            except Exception:
                pass
            if out_dir:
                cfg["settings"]["output_dir"] = out_dir
                cfg["settings"]["record_path"] = out_dir
            if nt_ip:
                cfg["nt"]["server_ip"] = nt_ip
            if nt_key:
                cfg["nt"]["boolean_key"] = nt_key
            try:
                if cam_idx != "":
                    cfg["camera"]["selected_index"] = int(float(cam_idx))
            except Exception:
                pass
            try:
                w = int(float(rw)) if rw else 0
                h = int(float(rh)) if rh else 0
                if w > 0 and h > 0:
                    cfg["settings"].setdefault("resolution", {})
                    cfg["settings"]["resolution"]["w"] = w
                    cfg["settings"]["resolution"]["h"] = h
            except Exception:
                pass

            if ffmpeg_path:
                cfg["web_transcode"]["ffmpeg_path"] = ffmpeg_path
            if ffmpeg_preset:
                cfg["web_transcode"]["preset"] = ffmpeg_preset
            try:
                if ffmpeg_crf:
                    cfg["web_transcode"]["crf"] = int(float(ffmpeg_crf))
            except Exception:
                pass

            save_config(cfg)

            # Try to apply camera immediately
            try:
                if cam_idx != "":
                    _send_control({"cmd": "switch_camera", "index": int(float(cam_idx))})
            except Exception:
                pass

            self._redirect("/?msg=Saved")
            return

        self._send_headers(404, "text/plain; charset=utf-8")
        self.wfile.write(b"not found\n")

    def log_message(self, format, *args):  # noqa: A002
        return


def main():
    cfg = load_config()
    stream = cfg.get("stream", {}) if isinstance(cfg.get("stream"), dict) else {}
    host = str(stream.get("host", "127.0.0.1"))
    port = int(stream.get("port", 8765))

    web = cfg.get("web", {}) if isinstance(cfg.get("web"), dict) else {}
    web_host = str(web.get("host", "0.0.0.0"))
    web_port = int(web.get("port", 8080))

    store = FrameStore()
    Handler.store = store

    t = threading.Thread(target=stream_reader, args=(store, host, port), daemon=True)
    t.start()

    httpd = ThreadingHTTPServer((web_host, web_port), Handler)
    print(f"[WEB] serving MJPEG on http://{web_host}:{web_port}/ (stream source {host}:{port})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

