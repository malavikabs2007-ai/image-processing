# Project report outline

## Abstract
This project implements an educational camera–LiDAR data pipeline for KITTI-style autonomous-driving data. It loads camera images and Velodyne scans, projects points with calibration, detects pedestrians with OpenCV HOG, clusters LiDAR geometry with Open3D DBSCAN, and associates clusters with image boxes.

## Method
The camera baseline is OpenCV's built-in HOG people detector. The LiDAR baseline filters returns by range and clusters 3D points. Calibration transforms Velodyne coordinates to rectified camera coordinates and projects them with the camera projection matrix. A cluster receives a camera class only when enough of its projected points fall within a real camera detection box. Axis-aligned cuboids summarize the LiDAR cluster.

The PointPillars-style module only quantizes returns into vertical pillars and encodes per-point features. It has no prediction head or trained weights. It is an extension point, not a trained detector.

## Reproducibility and evaluation
Record the KITTI split and frame IDs, calibration, detector settings, package versions, hardware, and any model checkpoint provenance. A real benchmark study must train a model on the training split and evaluate on a held-out split using the chosen KITTI 3D/BEV average-precision protocol. No benchmark metrics are supplied by this repository.

## Limitations and future work
HOG is pedestrian-only and not optimized for modern road scenes. DBSCAN clusters geometry without semantic labels. Projection depends on exact frame synchronization and calibration. Cuboids are axis-aligned approximations. Future work should train a real PointPillars network, add a verified image classifier, evaluate per-class 3D/BEV AP, and measure runtime and failure cases.
