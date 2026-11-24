"""KITTI class aliases and format validation helpers."""
KITTI_CLASSES={"Car","Van","Truck","Pedestrian","Person_sitting","Cyclist","Tram","Misc","DontCare"}

def normalize_class(label):
    """Normalize common KITTI labels without claiming that a detector predicted them."""
    return {"Van":"Car","Person_sitting":"Pedestrian","Tram":"Other road object","Misc":"Other road object"}.get(label,label)
