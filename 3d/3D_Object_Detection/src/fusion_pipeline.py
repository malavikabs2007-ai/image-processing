"""Calibrated projection and geometry-to-camera association."""
from src.pointcloud_projection import project_points
from src.sensor_fusion import associate

def fuse(image_shape, points, calibration, camera_detections, lidar_objects):
    """Project the input points and associate clusters using their original indices."""
    projected,indices=project_points(points,calibration,image_shape)
    index_to_pixel={int(i):(float(p[0]),float(p[1])) for i,p in zip(indices,projected)}
    for obj in lidar_objects:
        obj["pixels"]=[index_to_pixel[int(i)] for i in obj["point_indices"] if int(i) in index_to_pixel]
    return projected,associate(camera_detections,lidar_objects)
