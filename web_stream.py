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
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

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

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)

        if u.path == "/" or u.path.startswith("/?"):
            self._send_headers(200, "text/html; charset=utf-8")
            qs = parse_qs(u.query)
            msg = (qs.get("msg", [""])[0] or "").strip()
            msg_html = f"<p style='color:#8f8'>{html.escape(msg)}</p>" if msg else ""
            self.wfile.write(
                b"<!doctype html><html><head><meta charset='utf-8'>"
                b"<title>Camera Stream</title>"
                b"<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0}"
                b".wrap{padding:16px}img{max-width:100%;height:auto;border:1px solid #333}"
                b"button{margin-right:8px;margin-bottom:8px;padding:8px 12px}"
                b"a{color:#7af}"
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
                b"</div></body></html>"
            )
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

            self.wfile.write(b"<button type='submit'>Save</button>")
            self.wfile.write(b"</form></div></body></html>")
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

            pre = form.get("pre_roll_seconds", "").strip()
            buf = form.get("buffer_seconds", "").strip()
            out_dir = form.get("output_dir", "").strip()
            nt_ip = form.get("nt_server_ip", "").strip()
            nt_key = form.get("nt_boolean_key", "").strip()
            cam_idx = form.get("camera_index", "").strip()
            rw = form.get("resolution_w", "").strip()
            rh = form.get("resolution_h", "").strip()

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

