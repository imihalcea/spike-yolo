import cv2, time, yolo_classes, tools
from recorder import Recorder
from typing import List, Dict, Tuple, Optional, Callable, Any
from settings import Settings
from detection.detectorcv2 import DetectorCv2
from annotate import Annotations
from stats import FpsComputer, Stats



CLASSES = yolo_classes.ALL

def main():
    settings = Settings.load_from_env(CLASSES)
    recorder = Recorder(settings)
    detector = DetectorCv2(settings, time.time)
    annot = Annotations(tools.uname_info, settings)
    fps_computer = FpsComputer()
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.capture_height)
    
    while True:
        ret, frame = cap.read()
        detections = detector.detect(frame)
        frame = annot.annotate(frame, detections, fps_computer)
        
        if settings.recording_annotations:
            recorder.write(frame)
        
        display_frame(frame)
        fps_computer.update(1, time.time())
        if (cv2.waitKey(1) == ord('q')):
            break
            
    cap.release()
    recorder.release()
    cv2.destroyAllWindows()

#@time_logger
def display_frame(frame):
    cv2.imshow("LiveCam", frame)

if __name__ == "__main__":
    main()
    