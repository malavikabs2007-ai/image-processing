"""Dataset file loading and validation helpers."""
from pathlib import Path
import numpy as np
import cv2

def load_image(source):
    """Read an image path or uploaded bytes as a BGR array; reject bad inputs."""
    if isinstance(source, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Cannot read image. Choose a valid PNG/JPG camera image.")
    return image

def load_lidar(source):
    """Read KITTI float32 x,y,z,intensity records from a .bin file/bytes."""
    if isinstance(source, (bytes, bytearray)):
        data = np.frombuffer(source, dtype=np.float32)
    else:
        if Path(source).suffix.lower() != ".bin":
            raise ValueError("LiDAR input must be a KITTI .bin file")
        data = np.fromfile(source, dtype=np.float32)
    if data.size == 0 or data.size % 4:
        raise ValueError("Invalid KITTI .bin: expected non-empty float32 x,y,z,intensity records")
    points = data.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError("LiDAR file contains non-finite values")
    return points

def read_kitti_labels(path):
    """Parse KITTI object labels; ground truth is clearly marked as labels."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 15:
            continue
        try:
            x1,y1,x2,y2 = map(float, parts[4:8])
            h,w,l = map(float, parts[8:11]); x,y,z = map(float, parts[11:14]); yaw=float(parts[14])
        except ValueError:
            continue
        rows.append({"class":parts[0],"bbox_2d":[x1,y1,x2,y2],"dimensions_hwl":[h,w,l],"location_xyz": [x,y,z],"rotation_y":yaw,"source":"KITTI ground-truth label"})
    return rows
