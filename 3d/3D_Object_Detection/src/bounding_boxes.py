"""Bounding-box geometry utilities for LiDAR cluster proposals."""
import numpy as np

def corners_from_cluster(center_xyz, dimensions_xyz):
    """Return eight axis-aligned 3D corners in LiDAR coordinates."""
    c=np.asarray(center_xyz); half=np.asarray(dimensions_xyz)/2
    signs=np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    return c+signs*half

def project_box(corners, calibration):
    """Project LiDAR-frame cuboid corners to image pixels; invalid corners become None."""
    from utils.calibration import matrices
    p,r,t=matrices(calibration)
    cam=(r@t@np.column_stack((corners,np.ones(len(corners)))).T).T
    uv=(p@cam.T).T
    result=[]
    for q,z in zip(uv,cam[:,2]): result.append((q[:2]/q[2]).tolist() if z>0 and q[2]>0 else None)
    return result
