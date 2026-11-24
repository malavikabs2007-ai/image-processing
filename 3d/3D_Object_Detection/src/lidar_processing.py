"""Point cloud filtering and a transparent clustering baseline."""
import numpy as np

def clean_points(points, max_range=80.0):
    """Drop invalid, near-origin and out-of-range returns; preserve intensity."""
    xyz=points[:,:3]; dist=np.linalg.norm(xyz,axis=1)
    keep=np.isfinite(points).all(axis=1)&(xyz[:,0]>0.5)&(dist<max_range)
    return points[keep]

def cluster_objects(points, eps=0.8, min_points=8, max_clusters=80, max_sample_points=40000):
    """Run Open3D DBSCAN on a bounded sample; return unclassified geometry proposals."""
    if len(points)<min_points: return []
    try: import open3d as o3d
    except ImportError as exc: raise RuntimeError("Install Open3D with: pip install open3d") from exc
    # Bound DBSCAN cost on full KITTI scans while preserving original point indices.
    source_ids=np.arange(len(points))
    if len(points)>max_sample_points:
        source_ids=np.linspace(0,len(points)-1,max_sample_points,dtype=np.int64)
    sample=points[source_ids]
    cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(sample[:,:3].astype(float))
    labels=np.asarray(cloud.cluster_dbscan(eps=eps,min_points=min_points,print_progress=False)); objects=[]
    for label in np.unique(labels):
        if label<0: continue
        local_ids=np.flatnonzero(labels==label); ids=source_ids[local_ids]; cluster=points[ids]; xyz=cluster[:,:3]; lo=xyz.min(axis=0); hi=xyz.max(axis=0)
        if len(cluster)<min_points or np.any(hi-lo>12): continue
        objects.append({"class":"LiDAR proposal (unclassified)","center_xyz":((lo+hi)/2).tolist(),"dimensions_xyz":(hi-lo).tolist(),"distance_m":float(np.linalg.norm(xyz[:,:2].mean(axis=0))),"point_count":int(len(xyz)),"points_xyz":xyz,"point_indices":ids})
        if len(objects)>=max_clusters: break
    return objects
