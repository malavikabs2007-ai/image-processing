"""KITTI calibration parsing and LiDAR-to-camera projection."""
from pathlib import Path
import numpy as np

class CalibrationError(ValueError):
    """Raised for malformed or incomplete KITTI calibration data."""

def read_calibration(path):
    """Read KITTI key: numeric-values calibration records into arrays."""
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        try:
            arr = np.asarray([float(v) for v in values.split()], dtype=np.float64)
        except ValueError as exc:
            raise CalibrationError(f"Invalid numeric values for {key.strip()}") from exc
        if arr.size:
            result[key.strip()] = arr
    return result

def matrices(calibration):
    """Return P2 (3x4), rectification (4x4), and velo-to-camera (4x4)."""
    try:
        p = calibration.get("P2", calibration.get("P_rect_02")).reshape(3, 4)
        r0 = calibration.get("R0_rect", calibration.get("R_rect_00", np.eye(3).ravel())).reshape(3, 3)
        tr = calibration.get("Tr_velo_to_cam", calibration.get("Tr_velo_cam")).reshape(3, 4)
    except (AttributeError, ValueError) as exc:
        raise CalibrationError("Calibration needs P2, R0_rect (optional), and Tr_velo_to_cam") from exc
    r = np.eye(4); r[:3, :3] = r0
    t = np.eye(4); t[:3, :] = tr
    return p, r, t

def load_matrices(path):
    """Load a calibration file and return its projection matrices."""
    return matrices(read_calibration(path))
