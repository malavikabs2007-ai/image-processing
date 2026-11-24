"""OpenCV image drawing helpers."""
import cv2

COLORS={"Car":(255,90,30),"Pedestrian":(40,220,80),"Cyclist":(0,200,255),"Truck":(200,60,255)}

def draw_detections(image, detections):
    """Draw model boxes, confidence scores, and class names on a copy of the image."""
    out=image.copy()
    for d in detections:
        x1,y1,x2,y2=map(int,d["bbox_2d"]); color=COLORS.get(d["class"],(0,180,255))
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        label=f'{d["class"]} {d["confidence"]:.2f}'
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        top=max(0,y1-th-10); cv2.rectangle(out,(x1,top),(x1+tw+8,top+th+8),color,-1)
        cv2.putText(out,label,(x1+4,top+th+2),cv2.FONT_HERSHEY_SIMPLEX,.55,(15,15,15),2)
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
