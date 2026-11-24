"""Camera detectors: lightweight HOG and pretrained COCO Faster R-CNN."""
import cv2

# COCO labels that are useful in a road scene; model IDs follow torchvision COCO.
COCO_ROAD_LABELS = {
    1: "Pedestrian", 2: "Bicycle", 3: "Car", 4: "Motorcycle",
    6: "Bus", 8: "Truck", 10: "Traffic light", 12: "Traffic sign",
    13: "Other road object", 14: "Other road object",
}

class CameraDetector:
    """Run an actual detector; scores are model confidence, not calibrated probability."""
    def __init__(self, backend="COCO Faster R-CNN", confidence=0.45, device=None):
        self.backend = backend
        self.confidence = float(confidence)
        self.model = None
        if backend == "COCO Faster R-CNN":
            try:
                import torch
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
                weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device).eval()
                self.torch = torch
            except Exception as exc:
                raise RuntimeError(
                    "Could not load the pretrained COCO detector. Check that torch and "
                    "torchvision are installed and allow the first-run model weight download. "
                    f"Details: {exc}"
                ) from exc
        elif backend == "OpenCV HOG (pedestrians only)":
            if not hasattr(cv2, "HOGDescriptor"):
                raise RuntimeError("This OpenCV build does not include HOGDescriptor. Reinstall requirements.txt (it pins OpenCV 4.x), or choose COCO Faster R-CNN.")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            raise ValueError(f"Unknown camera detector: {backend}")

    def detect(self, image, confidence=None):
        """Return pixel boxes and scores, optionally overriding the minimum score."""
        threshold = self.confidence if confidence is None else float(confidence)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a valid 3-channel BGR camera image")
        if self.backend == "OpenCV HOG (pedestrians only)":
            boxes, weights = self.hog.detectMultiScale(image, winStride=(8,8), padding=(8,8), scale=1.05)
            detections = []
            for (x,y,w,h), score in zip(boxes, weights):
                confidence = float(score)
                if confidence >= threshold:
                    detections.append({"class":"Pedestrian", "confidence":min(confidence,1.0),
                        "bbox_2d":[int(x),int(y),int(x+w),int(y+h)], "source":self.backend})
            return detections

        # torchvision expects RGB float tensors in [0, 1].
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2,0,1).float().div(255).to(self.device)
        with self.torch.inference_mode():
            output = self.model([tensor])[0]
        detections=[]
        for box, label, score in zip(output["boxes"].cpu().numpy(), output["labels"].cpu().numpy(), output["scores"].cpu().numpy()):
            class_name = COCO_ROAD_LABELS.get(int(label))
            if class_name is None or float(score) < threshold:
                continue
            x1,y1,x2,y2 = box.tolist()
            h,w=image.shape[:2]
            detections.append({"class":class_name, "confidence":float(score),
                "bbox_2d":[max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2))],
                "source":self.backend})
        return detections


