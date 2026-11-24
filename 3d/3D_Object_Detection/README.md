# 3D Object Detection from LiDAR + Camera Fusion for Autonomous Vehicles

A local Flask web dashboard demonstrating camera detections, KITTI LiDAR loading, calibrated projection, geometric proposals, and camera/LiDAR association. The system is modular and intentionally reports only measurements it actually has.

## Model scope and honest reporting

- Camera uses torchvision's pretrained COCO Faster R-CNN. Its first use downloads weights. COCO scores are detector confidence scores, not calibrated probabilities.
- LiDAR is filtered and clustered with Open3D DBSCAN. Each cluster produces an axis-aligned geometric extent and is explicitly an **unclassified proposal**.
- Camera/LiDAR mode projects points with KITTI calibration and associates image detections with geometric clusters. Association does not turn the proposal generator into a trained 3D detector.
- `models/pointpillars_lite.py` contains pillarization and an untrained point feature encoder. It is **not a genuinely trained PointPillars detector** and has no trained detection head or checkpoint.
- Model Comparison intentionally shows no AP numbers. KITTI 3D AP and BEV AP require labeled validation data, model predictions, and a named evaluation protocol. This project has not generated those measurements.
- The synthetic option creates a repeatable point cloud for UI and geometry testing. It does not create semantic labels, detections, or accuracy claims.

## Windows and VS Code setup

1. Install 64-bit Python 3.10 or 3.11 and the VS Code Microsoft Python extension.
2. Open this folder in VS Code. In a PowerShell terminal, create and activate an environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Torch and Open3D are large dependencies. If Torch/torchvision wheels conflict, install a matching pair from the official PyTorch install selector, then install the remaining requirements.
3. Select `.venv` using **Python: Select Interpreter** in VS Code.
4. Start the server from the project root:

   ```powershell
   python app.py
   ```

5. Open `http://127.0.0.1:5000`. Choose **Object Detection → Load synthetic sample → Run detection** for an offline LiDAR-only UI check. Fusion's pretrained camera model needs an internet connection once to fetch weights.

## KITTI data

Download KITTI 3D Object Detection data from the official KITTI site and respect its terms. The web application accepts synchronized files from the dataset:

- Camera image: PNG/JPEG from `image_2`
- LiDAR: raw float32 `x,y,z,intensity` `.bin` from `velodyne`
- Fusion calibration: matching `calib` text file containing `P2` and `Tr_velo_to_cam`; `R0_rect` is optional
- Labels: `label_2` is ground truth for a separately configured evaluation workflow; the web app never presents labels as predictions

Preserve matching frame IDs. A typical local arrangement is `dataset/images/`, `dataset/lidar/`, `dataset/calibration/`, and optional `dataset/labels/`. The UI uploads the paired files directly; it does not assume a full dataset is present.

## Pages

- **Dashboard:** session totals and recent run history.
- **Object Detection:** upload image and LiDAR; optionally select calibrated fusion. Synthetic sample generation is built in.
- **3D Visualization:** latest camera overlay and Matplotlib bird's-eye point cloud.
- **Model Comparison:** explains which measurements are unavailable and why.
- **Detection History:** SQLite-backed past run metadata and object counts.

## Data and outputs

SQLite history is stored in `database/detections.sqlite3`. Processed overlays and bird's-eye images are saved under `outputs/visualizations/`. Uploaded input files are temporary and removed after processing. Synthetic test inputs are in `dataset/synthetic/` after creating the sample.

## Troubleshooting

- **No module named Flask/Open3D:** confirm the project `.venv` is selected and repeat `python -m pip install -r requirements.txt`.
- **COCO weight download fails:** the LiDAR-only and synthetic modes work without model weights; fusion's camera detector requires the cached/downloaded weights.
- **LiDAR file rejected:** KITTI Velodyne must be raw float32 records of four values per point (x, y, z, intensity), not zipped or text data.
- **Projection/fusion error or no association:** verify that image, scan, and calibration share a frame ID and vehicle calibration. No overlap is a valid result.
- **No class detections:** DBSCAN proposals are not semantically classified. The counted classes only include class associations produced in fusion.
- **No AP values:** train/evaluate a real detector on a KITTI validation split and record the split, checkpoint, IoU protocol, and evaluator before reporting AP.

## Project structure

`app.py` Flask backend; `models/` camera and PointPillars-style utilities; `src/` LiDAR, projection, association and geometry; `utils/` KITTI data and calibration helpers; `visualization/` point-cloud plots; `templates/` server-rendered dashboard pages; `static/` CSS and JavaScript; `database/` SQLite history; `dataset/` synthetic and optional KITTI files.

## Interactive 3D view

After each detection, the result page and **3D Visualization** page include a Plotly point-cloud viewer. Drag to orbit, scroll to zoom, hover for XYZ/range/intensity, and use the mode bar to reset or save a view. Axis-aligned cluster extents are drawn as box outlines. The viewer samples up to 30,000 points to keep browser interaction responsive; use a real KITTI scan for a dense, realistic-looking scene. Synthetic samples are intentionally sparse and are only for checking the pipeline.
