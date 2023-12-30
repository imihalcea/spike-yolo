from typing import Dict
import numpy as np
import supervision as sv
import cv2
from stats import FpsComputer
from settings import Settings

class Annotations:
    def __init__(self, platform_info:str, settings:Settings):
        self.classes = settings.classes
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
        self.platform_info = platform_info

    def annotate(self, frame:np.ndarray, detections:sv.Detections, fps:FpsComputer) -> np.ndarray:
        if detections is not None:
            frame = self.box_annotator.annotate(frame, detections)
            labels = [
                f"{self.classes[class_id]} {confidence:0.1f}"
                for _, _, confidence, class_id, _
                in detections
            ]
            frame = self.label_annotator.annotate(frame, detections, labels)
        
        cv2.putText(frame, f"FPS: {int(fps.compute().mean)}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)     
        cv2.putText(frame, f"{self.platform_info}", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        
        return frame