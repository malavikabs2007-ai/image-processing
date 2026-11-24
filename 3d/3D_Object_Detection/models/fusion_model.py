"""Data association for image detections and LiDAR geometric proposals."""
def associate(camera_detections, lidar_objects, minimum_points=5):
    """Attach a camera class only when a LiDAR cluster projects inside its 2D box."""
    matches=[]
    for obj in lidar_objects:
        pixels=obj.get("pixels", [])
        if len(pixels) < minimum_points:
            continue
        best=None; best_fraction=0.0
        for det in camera_detections:
            x1,y1,x2,y2=det["bbox_2d"]
            inside=sum(x1<=x<=x2 and y1<=y<=y2 for x,y in pixels)/len(pixels)
            if inside>best_fraction:
                best,best_fraction=det,inside
        if best is not None and best_fraction >= 0.25:
            item={k:v for k,v in obj.items() if k!="pixels"}
            item.update({"class":best["class"],"camera_confidence":best["confidence"],"association_fraction":best_fraction,"source":"LiDAR cluster + camera association"})
            matches.append(item)
    return matches
