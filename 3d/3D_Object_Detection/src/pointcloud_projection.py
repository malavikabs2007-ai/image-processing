"""KITTI projection of LiDAR returns into camera image coordinates."""
import numpy as np
from utils.calibration import matrices

def project_points(points, calibration, image_shape):
    """Project Velodyne points; return visible (u,v,depth,intensity) and source indices."""
    p,r,t=matrices(calibration); xyz=points[:,:3]
    homogeneous=np.column_stack((xyz,np.ones(len(xyz)))); camera=(r@t@homogeneous.T).T
    depth=camera[:,2]; projected=(p@camera.T).T; uv=projected[:,:2]/np.maximum(projected[:,2:3],1e-8)
    h,w=image_shape[:2]; valid=(depth>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    intensity=points[:,3] if points.shape[1]>3 else np.zeros(len(points))
    return np.column_stack((uv[valid],depth[valid],intensity[valid])),np.flatnonzero(valid)

def draw_projection(image, projected):
    """Overlay projected returns colored by distance on a BGR image."""
    import cv2
    out=image.copy()
    if not len(projected): return cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    max_depth=max(float(np.percentile(projected[:,2],95)),1.0)
    for u,v,d,_ in projected[::max(1,len(projected)//15000)]:
        color=cv2.applyColorMap(np.array([[int(255*max(0,1-d/max_depth))]],dtype=np.uint8),cv2.COLORMAP_JET)[0,0]
        cv2.circle(out,(int(u),int(v)),2,tuple(int(c) for c in color),-1)
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
