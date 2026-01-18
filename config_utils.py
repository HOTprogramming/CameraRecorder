import json
import os
from pathlib import Path
from typing import Any


def get_config_path() -> Path:
    """
    Prefer %APPDATA%/CameraRecorder/config.json on Windows.
    Fallback to user's home directory on other platforms.
    """
    try:
        appdata = os.getenv("APPDATA")
        if appdata:
            cfg_dir = Path(appdata) / "CameraRecorder"
        else:
            cfg_dir = Path.home() / ".camera_recorder"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "config.json"
    except Exception:
        return Path("camera_recorder_config.json")


def load_config() -> dict[str, Any]:
    p = get_config_path()
    try:
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    p = get_config_path()
    try:
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        tmp.replace(p)
    except Exception:
        # best-effort persistence; avoid crashing the app for settings writes
        pass

