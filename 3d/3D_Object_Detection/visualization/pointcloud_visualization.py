"""Open3D point cloud rendering with a safe Matplotlib fallback."""
import numpy as np

def pointcloud_figure(points, max_points=50000):
    """Create a browser-friendly 3D scatter plot colored by range."""
    import matplotlib.pyplot as plt
    stride=max(1,len(points)//max_points); xyz=points[::stride,:3]
    fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111,projection="3d")
    if len(xyz):
        distance=np.linalg.norm(xyz,axis=1); ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=distance,s=1,cmap="turbo")
    ax.set_xlabel("X forward (m)"); ax.set_ylabel("Y lateral (m)"); ax.set_zlabel("Z up (m)")
    ax.set_title("LiDAR point cloud (range colored)"); fig.tight_layout()
    return fig

def open3d_viewer(points):
    """Open an optional local Open3D interactive viewer; requires a desktop session."""
    import open3d as o3d
    cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(points[:,:3].astype(float))
    o3d.visualization.draw_geometries([cloud],window_name="KITTI LiDAR point cloud")
