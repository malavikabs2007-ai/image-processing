"""Streamlit workspace for image, LiDAR, projection and fusion demonstrations."""
from pathlib import Path
import sys, json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.data_loader import load_image,load_lidar,read_kitti_labels
from utils.calibration import read_calibration
from models.detector import CameraDetector
from src.lidar_processing import clean_points,cluster_objects
from src.pointcloud_projection import project_points,draw_projection
from src.camera_processing import draw_detections
from src.fusion_pipeline import fuse
from src.bounding_boxes import corners_from_cluster,project_box
from visualization.pointcloud_visualization import pointcloud_figure

st.set_page_config(page_title="3D Detection: Camera + LiDAR",layout="wide")
st.title("3D Object Detection Using LiDAR and Camera Fusion")
st.caption("Educational KITTI baseline • sensor-derived geometry • no pretrained 3D network is bundled")
page=st.sidebar.radio("Navigation",["Home","Upload Camera Image","Upload LiDAR File","3D Object Detection","Camera–LiDAR Fusion","3D Point Cloud Visualization","Detection Results","About Project"])
@st.cache_resource
def camera_detector(backend): return CameraDetector(backend=backend)

def parse_calibration(upload):
    """Parse the uploaded KITTI text calibration without writing it to disk."""
    from utils.calibration import CalibrationError
    from pathlib import Path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt",delete=False) as f: f.write(upload.getvalue()); name=f.name
    try: return read_calibration(name)
    finally: Path(name).unlink(missing_ok=True)

def process_camera(file, backend, confidence=0.45):
    """Load an uploaded/captured image and run the selected real camera detector."""
    image=load_image(file.getvalue()); detections=camera_detector(backend).detect(image, confidence=confidence)
    return image,detections

def process_lidar(file):
    """Validate and filter uploaded KITTI LiDAR returns."""
    points=clean_points(load_lidar(file.getvalue()))
    if len(points)>120000: points=points[np.linspace(0,len(points)-1,120000).astype(int)]
    return points

if page=="Home":
    st.subheader("A reproducible sensor-fusion baseline")
    st.write("Upload or capture a camera image for pretrained COCO road-object detection, then pair synchronized KITTI camera, Velodyne .bin, and calibration files for projected LiDAR and geometric fusion.")
    st.info("No trained PointPillars checkpoint is included. LiDAR clusters are proposals, not semantic detections. The pretrained COCO model needs an internet connection for its first weight download; cached weights work offline. Fusion needs synchronized files and matching calibration.")
    st.markdown("**Quick start:** create a venv, install `requirements.txt`, run `streamlit run app.py`, then upload synchronized files under Camera–LiDAR Fusion.")
elif page in ("Upload Camera Image","3D Object Detection"):
    st.subheader("Camera image and object detection")
    st.caption("COCO Faster R-CNN detects people, cars, motorcycles, buses, trucks, bicycles, and selected road objects. First use downloads pretrained weights. Scores are detector confidence, not calibrated probabilities. Choose OpenCV HOG for a lightweight pedestrian-only fallback when available.")
    backends=["COCO Faster R-CNN"] + (["OpenCV HOG (pedestrians only)"] if hasattr(cv2,"HOGDescriptor") else [])
    backend=st.selectbox("Camera model",backends,key="camera_backend")
    confidence=st.slider("Minimum confidence",0.10,0.90,0.45,0.05,key="camera_confidence")
    source=st.radio("Image source",["Upload image","Capture with camera"],horizontal=True,key="camera_source")
    file=st.file_uploader("Choose a road image",type=["png","jpg","jpeg"],key="camera") if source=="Upload image" else st.camera_input("Take a camera photo",key="camera_capture")
    if file:
        try:
            image,dets=process_camera(file,backend,confidence); left,right=st.columns(2)
            left.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption="Original camera image",use_container_width=True)
            right.image(draw_detections(image,dets),caption=f"Detections • {backend}",use_container_width=True)
            st.write(f"Actual detections: {len(dets)}")
            if dets: st.dataframe(pd.DataFrame([{**d,"bbox_2d":str(d['bbox_2d'])} for d in dets]),use_container_width=True)
            else: st.info("The selected model returned no boxes above this confidence threshold. Try a lower threshold or a clearer image.")
            st.session_state["camera_image"]=image; st.session_state["camera_detections"]=dets
        except Exception as exc: st.error(f"Camera processing failed: {exc}")
elif page=="Upload LiDAR File":
    st.subheader("KITTI Velodyne scan")
    file=st.file_uploader("Choose a .bin scan (float32 x,y,z,intensity)",type=["bin"],key="lidar")
    if file:
        try:
            pts=process_lidar(file); st.session_state["points"]=pts
            st.success(f"Loaded {len(pts):,} filtered points")
            st.pyplot(pointcloud_figure(pts),use_container_width=True)
        except Exception as exc: st.error(f"LiDAR processing failed: {exc}")
elif page=="Camera–LiDAR Fusion":
    st.subheader("Synchronized camera + LiDAR + calibration")
    col1,col2=st.columns(2)
    image_file=col1.file_uploader("Camera image (.png/.jpg)",type=["png","jpg","jpeg"],key="fusion_img")
    lidar_file=col2.file_uploader("LiDAR scan (.bin)",type=["bin"],key="fusion_lidar")
    calib_file=st.file_uploader("KITTI calibration (.txt)",type=["txt"],key="calib")
    if image_file and lidar_file and calib_file and st.button("Run calibrated fusion"):
        try:
            image=load_image(image_file.getvalue()); pts=process_lidar(lidar_file); calib=parse_calibration(calib_file)
            detections=camera_detector("COCO Faster R-CNN").detect(image); proposals=cluster_objects(pts)
            projected,objects=fuse(image.shape,pts,calib,detections,proposals)
            overlay=draw_projection(image,projected); overlay_bgr=cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)
            for obj in objects:
                corners=corners_from_cluster(obj["center_xyz"],obj["dimensions_xyz"]); uv=project_box(corners,calib)
                edges=[(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
                for a,b in edges:
                    if uv[a] and uv[b]: cv2.line(overlay_bgr,tuple(np.round(uv[a]).astype(int)),tuple(np.round(uv[b]).astype(int)),(0,255,255),2)
            c1,c2=st.columns(2); c1.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption="Camera frame",use_container_width=True); c2.image(cv2.cvtColor(overlay_bgr,cv2.COLOR_BGR2RGB),caption="Projected LiDAR + fused cuboids",use_container_width=True)
            st.write(f"Projected returns: {len(projected):,} • LiDAR clusters: {len(proposals)} • fused class-supported cuboids: {len(objects)}")
            if objects: st.dataframe(pd.DataFrame([{k:v for k,v in o.items() if k!="points_xyz"} for o in objects]),use_container_width=True)
            else: st.info("No cluster had enough projected points inside a camera detection box. This is not an error and no detections were fabricated.")
            st.session_state["last_results"]=[{k:v for k,v in o.items() if k!="points_xyz"} for o in objects]
        except Exception as exc: st.error(f"Fusion failed. Check synchronized files and KITTI calibration keys: {exc}")
    elif not (image_file and lidar_file and calib_file): st.info("Provide all three synchronized files. KITTI calibration must contain P2, Tr_velo_to_cam and optionally R0_rect.")
elif page=="3D Point Cloud Visualization":
    file=st.file_uploader("Choose KITTI .bin",type=["bin"],key="cloud_view")
    pts=st.session_state.get("points")
    if file:
        try: pts=process_lidar(file); st.session_state["points"]=pts
        except Exception as exc: st.error(str(exc)); pts=None
    if pts is not None: st.pyplot(pointcloud_figure(pts),use_container_width=True)
    else: st.info("Upload a scan on this page or use Upload LiDAR File first.")
elif page=="Detection Results":
    st.subheader("Latest fusion output")
    results=st.session_state.get("last_results",[])
    if results: st.dataframe(pd.DataFrame(results),use_container_width=True); st.download_button("Download results JSON",json.dumps(results,indent=2),file_name="detections.json",mime="application/json")
    else: st.info("Run Camera–LiDAR Fusion during this session to populate results. No canned sample detections are shown.")
elif page=="About Project":
    st.subheader("Scope and limitations")
    st.markdown("""This project demonstrates pretrained COCO camera detection, optional OpenCV HOG pedestrian detection, KITTI loading and calibrated Velodyne projection, Open3D DBSCAN geometric proposals, image-to-point-cloud association, axis-aligned LiDAR cuboids, and Streamlit visualization.

**Not included:** a KITTI-trained 3D detector or PointPillars weights, temporal tracking, or evaluation claims. COCO supplies 2D categories; it was not trained specifically for KITTI, and bicycles are not always riders. HOG only targets pedestrians. DBSCAN does not provide semantic classes. A fused cuboid appears only when a real camera detection and a real LiDAR proposal overlap after calibrated projection. Data must be time synchronized and calibration must correspond to the vehicle setup.

The KITTI label parser reads ground-truth labels for analysis; those labels are never presented as model predictions.""")
    st.code("python -m pip install -r requirements.txt\nstreamlit run app.py",language="powershell")



