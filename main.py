import cv2, time
import supervision as sv
from ultralytics import YOLO

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def main():
    
    # to save the video
    writer= cv2.VideoWriter('webcam_yolo.mp4', 
                            cv2.VideoWriter_fourcc(*'DIVX'), 
                            7, 
                            (FRAME_WIDTH, FRAME_HEIGHT))
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    model = YOLO("yolov8s.pt")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    
    names = model.model.names    
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        result = model.predict(frame, agnostic_nms=True, conf=0.6)[0]
        detections = sv.Detections.from_ultralytics(result)
        frame = box_annotator.annotate(frame, detections)
        labels = [
            f"{names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        frame = label_annotator.annotate(frame, detections, labels) 
        
        #writer.write(frame)
        
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        
        cv2.putText(frame, f"FPS: {int(fps)}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA) 
  
        cv2.imshow("LiveCam", frame)

        if (cv2.waitKey(1) == ord('q')):
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()