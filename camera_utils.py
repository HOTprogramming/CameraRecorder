import sys
from pathlib import Path

import cv2


def open_capture(camera_index: int, *, width: int = 1280, height: int = 720):
    """
    Open a camera by index.

    - Windows: try CAP_DSHOW (then fallback backends)
    - Linux: try CAP_V4L2 (then fallback)
    - Other: default backend

    Returns cv2.VideoCapture or None.
    """
    idx = int(camera_index)

    # Pick sensible backends per-OS
    if sys.platform.startswith("win"):
        backend_candidates = [cv2.CAP_DSHOW, getattr(cv2, "CAP_MSMF", 0), 0]
    elif sys.platform.startswith("linux"):
        backend_candidates = [getattr(cv2, "CAP_V4L2", 0), 0]
    else:
        backend_candidates = [0]

    cap = None
    try:
        for backend in backend_candidates:
            try:
                if backend:
                    cap = cv2.VideoCapture(idx, int(backend))
                else:
                    cap = cv2.VideoCapture(idx)

                if cap is not None and cap.isOpened():
                    break
            except Exception:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = None

        if cap is None or not cap.isOpened():
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            return None

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        return cap
    except Exception:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None


def enumerate_camera_choices() -> list[tuple[str, int]]:
    """
    Returns list of (label, camera_index) pairs.
    Uses cv2_enumerate_cameras if available; falls back to probing indices 0..9.

    - Windows: de-dupe duplicate interfaces by stable device path (before '#{...}').
    - Linux: prefer listing /dev/video* devices (with friendly names if available).
    """
    choices: list[tuple[str, int]] = []

    # Linux: enumerate /dev/video* first (more reliable than "index probing")
    if sys.platform.startswith("linux"):
        try:
            devs = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            out: list[tuple[str, int]] = []
            seen: set[int] = set()
            for p in devs:
                # /dev/video0 -> 0
                suffix = p.name.replace("video", "", 1)
                if not suffix.isdigit():
                    continue
                idx = int(suffix)
                if idx in seen:
                    continue
                seen.add(idx)

                name = f"Video{idx}"
                try:
                    sys_name = Path(f"/sys/class/video4linux/video{idx}/name")
                    if sys_name.exists():
                        txt = sys_name.read_text(encoding="utf-8", errors="ignore").strip()
                        if txt:
                            name = txt
                except Exception:
                    pass

                out.append((f"{name} ({idx})", idx))

            if out:
                return out
        except Exception:
            pass

    try:
        from cv2_enumerate_cameras import enumerate_cameras  # type: ignore

        cams = list(enumerate_cameras())
        groups: dict[tuple, dict] = {}

        for cam in cams:
            try:
                idx = int(getattr(cam, "index"))
                name = str(getattr(cam, "name", f"Camera {idx}"))
                path = str(getattr(cam, "path", "") or "").strip()
                vid = getattr(cam, "vid", None)
                pid = getattr(cam, "pid", None)

                stable_path = ""
                if sys.platform.startswith("win") and path:
                    lp = path.lower()
                    if lp.endswith("\\global"):
                        lp = lp[: -len("\\global")]
                    stable_path = lp.split("#{", 1)[0].strip()

                if stable_path:
                    key = ("stable_path", stable_path)  # Windows de-dupe
                elif vid is not None and pid is not None:
                    key = ("vidpid", int(vid), int(pid), name)
                else:
                    key = ("name", name)

                existing = groups.get(key)
                if existing is None or idx < existing["idx"]:
                    groups[key] = {"idx": idx, "name": name}
            except Exception:
                continue

        for g in sorted(groups.values(), key=lambda x: (x["name"].lower(), x["idx"])):
            label = f'{g["name"]} ({g["idx"]})'
            choices.append((label, int(g["idx"])))

        if choices:
            return choices
    except Exception:
        pass

    # Fallback probe
    for i in range(10):
        cap = None
        try:
            cap = open_capture(i)
            if cap is not None and cap.isOpened():
                choices.append((f"Camera {i}", i))
        except Exception:
            pass
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    return choices


def ensure_dir(p: str) -> str:
    try:
        Path(p).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

