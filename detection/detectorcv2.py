from typing import Dict, List, Tuple, Optional, Callable
import cv2
import numpy as np
import supervision as sv

from detection.detector import Detector
from settings import Settings
from stats import Computers, DurationComputer, Stats

class DetectorCv2(Detector):
    def __init__(self, settings:Settings, now:Callable[[],float]):
        super().__init__(settings)
        self.now = now
        self.model = cv2.dnn.readNetFromONNX(f"{settings.model}.onnx")
        if settings.runtime is not None:
            self.model.setPreferableBackend(settings.runtime.cv2_dnn_backend)
            self.model.setPreferableTarget(settings.runtime.cv2_dnn_target)
        
        self.computers = Computers()\
            .add(self.detect.__name__, DurationComputer())\
            .add(self.preprocess.__name__, DurationComputer())\
            .add(self.postprocess.__name__, DurationComputer())\
            .add(self.predict.__name__, DurationComputer())                
    
    def preprocess(self, original_image:np.ndarray) -> Tuple[np.ndarray, float]:
        [height, width, _] = original_image.shape

        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        scale = length / self.settings.optimizations.imgsz
        self.computers.update(self.preprocess.__name__, 1, self.now())

        return image, scale

    def postprocess(self, outputs:np.ndarray, scale:float, score_threshold:float=0.0) -> Optional[sv.Detections]:
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= score_threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                scaled_xyxy = [round(box[0] * scale), 
                round(box[1] * scale), 
                round((box[0] + box[2]) * scale), 
                round((box[1] + box[3]) * scale)]
                
                boxes.append(scaled_xyxy)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                
        self.computers.update(self.postprocess.__name__, 1, self.now())
        if(len(boxes) == 0): return None
        return sv.Detections(
            xyxy = np.array(boxes),
            confidence = np.array(scores),
            class_id = np.array(class_ids))    

    def predict(self, image:np.ndarray, model:cv2.dnn.Net) -> np.ndarray:
        img_size = self.settings.optimizations.imgsz
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(img_size, img_size), swapRB=True)
        model.setInput(blob)
        self.computers.update(self.predict.__name__, 1, self.now())
        return model.forward()
        
    def detect(self, frame:np.ndarray) -> sv.Detections:
        frame, scale = self.preprocess(frame)
        onnx_outputs = self.predict(frame, self.model)
        detections = self.postprocess(
            onnx_outputs, 
            scale, 
            self.settings.score_threshold)
        self.computers.update(self.detect.__name__, 1, self.now())
        return detections if detections is None else detections.with_nms(self.settings.nms_threshold, self.settings.nms_class_agnostic)    


    def statistics(self) -> Dict[str, Stats]:
        return self.computers.compute()