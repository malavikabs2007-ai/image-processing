"""Flask dashboard for an educational camera/LiDAR detection baseline."""
from pathlib import Path
from datetime import datetime
import json, sqlite3, uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from utils.data_loader import load_image, load_lidar
from utils.calibration import read_calibration
from src.lidar_processing import clean_points, cluster_objects
from src.pointcloud_projection import project_points, draw_projection
from src.sensor_fusion import associate
from models.detector import CameraDetector
from visualization.pointcloud_visualization import pointcloud_figure

ROOT=Path(__file__).resolve().parent
UPLOAD=ROOT/'outputs'/'uploads'; RESULTS=ROOT/'outputs'/'visualizations'; DB=ROOT/'database'/'detections.sqlite3'
UPLOAD.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=40*1024*1024
_camera_detector=None

def connect():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
    with connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS detections(id TEXT PRIMARY KEY, created_at TEXT, image_name TEXT, lidar_name TEXT, mode TEXT, object_count INTEGER, cars INTEGER, pedestrians INTEGER, cyclists INTEGER, results TEXT, image_path TEXT, bev_path TEXT, point_count INTEGER)")
init_db()

def get_detector():
    global _camera_detector
    if _camera_detector is None: _camera_detector=CameraDetector('COCO Faster R-CNN')
    return _camera_detector

def upload(file, extensions):
    if not file or not file.filename: raise ValueError('Upload the required files to run detection.')
    name=secure_filename(file.filename)
    if not name or '.' not in name or name.rsplit('.',1)[1].lower() not in extensions: raise ValueError(f'Unsupported file: {name}')
    path=UPLOAD/(uuid.uuid4().hex+'_'+name); file.save(path); return path,name

def clean_result(obj):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in obj.items() if k not in ('points_xyz','point_indices','pixels')}

def render_bev(points, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=pointcloud_figure(points); ax=fig.axes[0]
    # Draw axis-aligned cluster extents in the same XYZ coordinate frame.
    for obj in results:
        center=np.asarray(obj.get('center_xyz',[0,0,0]),dtype=float); dims=np.asarray(obj.get('dimensions_xyz',[0,0,0]),dtype=float)
        lo=center-dims/2; hi=center+dims/2
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        for a,b in ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)):
            ax.plot(*zip(corners[a],corners[b]),color='#ffbd68',linewidth=1.2)
    path=RESULTS/(uuid.uuid4().hex+'_bev.png'); fig.savefig(path,dpi=110,bbox_inches='tight'); plt.close(fig); return path

def render_interactive_3d(points, results, run_id):
    """Write a standalone Plotly 3D cloud that supports orbit, zoom, and hover."""
    import plotly.graph_objects as go
    stride=max(1,int(np.ceil(len(points)/30000)))
    cloud=points[::stride]
    xyz=cloud[:,:3]
    distance=np.linalg.norm(xyz,axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
        marker=dict(size=2,color=distance,colorscale='Turbo',opacity=.88,
                    colorbar=dict(title='Range<br>(m)',bgcolor='rgba(12,16,27,.75)',tickfont=dict(color='#d8def0'))),
        name='LiDAR returns',customdata=np.column_stack((distance,cloud[:,3] if cloud.shape[1]>3 else np.zeros(len(cloud)))),
        hovertemplate='X %{x:.2f} m<br>Y %{y:.2f} m<br>Z %{z:.2f} m<br>Range %{customdata[0]:.1f} m<br>Intensity %{customdata[1]:.2f}<extra></extra>'))
    edges=((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))
    for i,obj in enumerate(results):
        center=np.asarray(obj.get('center_xyz',[0,0,0]),dtype=float); dims=np.asarray(obj.get('dimensions_xyz',[0,0,0]),dtype=float)
        lo=center-dims/2; hi=center+dims/2
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        xs=[];ys=[];zs=[]
        for a,b in edges: xs.extend((corners[a,0],corners[b,0],None)); ys.extend((corners[a,1],corners[b,1],None)); zs.extend((corners[a,2],corners[b,2],None))
        label=obj.get('class','Unclassified proposal')
        fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode='lines',line=dict(color='#ffc66d',width=5),name=f'{label} {i+1}',hovertemplate=f'{label}<br>center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) m<extra></extra>'))
        fig.add_trace(go.Scatter3d(x=[center[0]],y=[center[1]],z=[hi[2]],mode='text',text=[label],textposition='top center',textfont=dict(size=10,color='#ffe1aa'),showlegend=False,hoverinfo='skip'))
    fig.update_layout(template='plotly_dark',paper_bgcolor='#0b0e18',plot_bgcolor='#0b0e18',
        margin=dict(l=0,r=0,t=44,b=0),height=650,title=dict(text='Interactive LiDAR point cloud · drag to orbit · scroll to zoom',font=dict(size=14,color='#e6e9f5')),
        scene=dict(xaxis_title='X · forward (m)',yaxis_title='Y · lateral (m)',zaxis_title='Z · height (m)',
            bgcolor='#0b0e18',aspectmode='data',camera=dict(eye=dict(x=-1.65,y=-1.65,z=1.15))),
        legend=dict(bgcolor='rgba(17,22,36,.8)',font=dict(color='#d8def0')),
        uirevision=run_id)
    path=RESULTS/(run_id+'_3d.html')
    path.write_text(fig.to_html(full_html=True,include_plotlyjs=True,config={'scrollZoom':True,'displaylogo':False,'responsive':True}),encoding='utf-8')
    return path

def run_detection(image_path,lidar_path,mode,calib_path=None):
    image=load_image(str(image_path)); points=clean_points(load_lidar(str(lidar_path)))
    if len(points)>120000: points=points[np.linspace(0,len(points)-1,120000).astype(int)]
    proposals=cluster_objects(points); camera=[]; overlay=image.copy()
    if mode=='fusion':
        if calib_path is None: raise ValueError('Fusion needs a KITTI calibration file with P2 and Tr_velo_to_cam.')
        calibration=read_calibration(calib_path); camera=get_detector().detect(image)
        pixels,ids=project_points(points,calibration,image.shape)
        mapping={int(i):(float(p[0]),float(p[1])) for i,p in zip(ids,pixels)}
        for obj in proposals: obj['pixels']=[mapping[int(i)] for i in obj['point_indices'] if int(i) in mapping]
        associated=associate(camera,proposals)
        # Keep geometric proposals, including unassociated ones; never invent semantic labels.
        results=[]
        for obj in proposals:
            item=clean_result(obj)
            for match in associated:
                if match.get('center_xyz')==obj.get('center_xyz') and match.get('dimensions_xyz')==obj.get('dimensions_xyz'):
                    item.update({k:v for k,v in match.items() if k not in ('points_xyz','point_indices','pixels')}); break
            results.append(item)
        from src.pointcloud_projection import draw_projection
        overlay=draw_projection(image,pixels)
        for det in camera:
            x1,y1,x2,y2=det['bbox_2d']; cv2.rectangle(overlay,(x1,y1),(x2,y2),(210,70,255),2)
            cv2.putText(overlay,f"{det['class']} {det['confidence']:.2f}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(210,70,255),1)
    else: results=[clean_result(o) for o in proposals]
    run_id=uuid.uuid4().hex
    visual=RESULTS/(run_id+'_overlay.jpg'); cv2.imwrite(str(visual),overlay); bev=render_bev(points,results); viewer=render_interactive_3d(points,results,run_id)
    counts={name:sum(1 for r in results if r.get('class')==name) for name in ('Car','Pedestrian','Cyclist')}
    counts['objects']=len(results)
    return results,counts,visual,bev,viewer,len(points),camera

@app.get('/')
def dashboard():
    with connect() as c:
        totals=dict(c.execute('SELECT COUNT(*) images,COALESCE(SUM(object_count),0) objects,COALESCE(SUM(cars),0) cars,COALESCE(SUM(pedestrians),0) pedestrians,COALESCE(SUM(cyclists),0) cyclists FROM detections').fetchone())
        recent=[dict(x) for x in c.execute('SELECT * FROM detections ORDER BY created_at DESC LIMIT 8')]
    return render_template('dashboard.html',totals=totals,recent=recent)

@app.get('/detect')
def detect_page(): return render_template('detect.html')
@app.get('/visualization')
def visualization():
    with connect() as c: row=c.execute('SELECT * FROM detections ORDER BY created_at DESC LIMIT 1').fetchone()
    latest=dict(row) if row else None
    if latest:
        stem=Path(latest['image_path']).stem.removesuffix('_overlay')
        latest['viewer_path']=f"outputs/visualizations/{stem}_3d.html"
    return render_template('visualization.html',row=latest)
@app.get('/comparison')
def comparison(): return render_template('comparison.html')
@app.get('/history')
def history():
    with connect() as c: rows=[dict(x) for x in c.execute('SELECT * FROM detections ORDER BY created_at DESC LIMIT 100')]
    return render_template('history.html',rows=rows)
@app.get('/api/comparison')
def comparison_data():
    return jsonify({'available':False,'metrics':None,'reason':'3D/BEV AP requires KITTI validation labels and evaluated detector predictions. No trained PointPillars checkpoint or validation run is configured, so no AP values are reported.'})
@app.get('/outputs/<path:name>')
def output_file(name):
    root=(ROOT/'outputs').resolve(); path=(root/name).resolve()
    if root not in path.parents: abort(404)
    return send_from_directory(root,name)
@app.get('/synthetic/<path:name>')
def synthetic_file(name): return send_from_directory(ROOT/'dataset'/'synthetic',name)
@app.post('/api/synthetic')
def synthetic():
    rng=np.random.default_rng(12); clouds=[]
    for center,n in [((12,2,0),180),((23,-4,0),140),((32,6,0),80)]:
        xyz=rng.normal(center,(.65,.55,.45),(n,3)); clouds.append(np.column_stack((xyz,rng.uniform(.1,1,n))))
    (ROOT/'dataset'/'synthetic'/'sample.bin').write_bytes(np.vstack(clouds).astype(np.float32).tobytes())
    image=np.full((375,1242,3),105,np.uint8); cv2.line(image,(0,285),(1242,285),(210,210,210),3); cv2.imwrite(str(ROOT/'dataset'/'synthetic'/'sample.png'),image)
    return jsonify({'image':'/synthetic/sample.png','lidar':'/synthetic/sample.bin','note':'Synthetic geometry only; proposals are unclassified and no benchmark scores are generated.'})
@app.post('/api/detect')
def detect_api():
    temp=[]
    try:
        image,image_name=upload(request.files.get('image'),{'png','jpg','jpeg'}); temp.append(image)
        lidar,lidar_name=upload(request.files.get('lidar'),{'bin'}); temp.append(lidar)
        mode=request.form.get('mode','lidar')
        calib=None
        if mode=='fusion': calib,calib_name=upload(request.files.get('calibration'),{'txt'}); temp.append(calib)
        results,counts,visual,bev,viewer,npoints,camera=run_detection(image,lidar,mode,calib)
        rid=visual.stem.removesuffix('_overlay'); created=datetime.now().astimezone().isoformat(timespec='seconds')
        with connect() as c: c.execute('INSERT INTO detections VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(rid,created,image_name,lidar_name,mode,counts['objects'],counts['Car'],counts['Pedestrian'],counts['Cyclist'],json.dumps(results),visual.relative_to(ROOT).as_posix(),bev.relative_to(ROOT).as_posix(),npoints))
        return jsonify({'id':rid,'created_at':created,'mode':mode,'results':results,'counts':counts,'point_count':npoints,'camera_detections':camera,'overlay':'/outputs/'+visual.relative_to(ROOT/'outputs').as_posix(),'bev':'/outputs/'+bev.relative_to(ROOT/'outputs').as_posix(),'viewer':'/outputs/'+viewer.relative_to(ROOT/'outputs').as_posix()})
    except Exception as exc: return jsonify({'error':str(exc)}),400
    finally:
        for path in temp:
            try: path.unlink(missing_ok=True)
            except OSError: pass

if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=False)
