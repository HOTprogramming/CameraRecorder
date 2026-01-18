import cv2
from pathlib import Path


def open_capture(camera_index: int, *, width: int = 1280, height: int = 720):
    """
    Open a camera by index. Tries CAP_DSHOW first (works well for small indices),
    then falls back to default backend (needed for some cv2_enumerate_cameras indices).
    Returns cv2.VideoCapture or None.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
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
    De-dupes Windows duplicate interfaces by stable device path (before '#{...}').
    """
    choices: list[tuple[str, int]] = []

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
                if path:
                    lp = path.lower()
                    if lp.endswith("\\global"):
                        lp = lp[: -len("\\global")]
                    stable_path = lp.split("#{", 1)[0].strip()

                if stable_path:
                    key = ("stable_path", stable_path)
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

