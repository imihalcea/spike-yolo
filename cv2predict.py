import cv2, time, yolo_classes
import numpy as np
import supervision as sv

from typing import List, Dict, Tuple, Optional, Callable, Any

ONNX_MODEL="yolov8n.onnx"
MODEL_IMAGE_SIZE=128
CAPTURE_FRAME_WIDTH = 800
CAPTURE_FRAME_HEIGHT = 600
CLASSES = yolo_classes.ALL

def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {(end_time - start_time) * 1000} ms.")
        return result
    return wrapper

#@time_logger
def prepare_image(original_image:np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Prepare a square image for inference
    """
    [height, width, _] = original_image.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    scale = length / MODEL_IMAGE_SIZE

    return image, scale

#@time_logger
def predict(image:np.ndarray, model:cv2.dnn.Net) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), swapRB=True)
    model.setInput(blob)
    return model.forward()

#@time_logger
def from_onnx(outputs:np.ndarray, scale:float, score_threshold:float=0.0) -> Optional[sv.Detections]:
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
    if(len(boxes) == 0): return None
    return sv.Detections(
        xyxy = np.array(boxes),
        confidence = np.array(scores),
        class_id = np.array(class_ids))
    
def inference_pipeline(image:np.ndarray, model:cv2.dnn.Net, score_threshold:float=0.25, nms_threshold:float=0.5, class_agnostic:bool=True) -> Optional[sv.Detections]:
    image, scale = prepare_image(image)
    onnx_outputs = predict(image, model)
    detections = from_onnx(onnx_outputs, scale, score_threshold)
    return detections if detections is None else detections.with_nms(nms_threshold, class_agnostic)    

def main():
    
    writer= cv2.VideoWriter('webcam_yolo_onnx.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            20.0, 
                            (CAPTURE_FRAME_WIDTH, CAPTURE_FRAME_HEIGHT))
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_FRAME_HEIGHT)

    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(ONNX_MODEL)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        detections = inference_pipeline(frame, model, score_threshold=0.4)

        if detections is not None:
            frame = box_annotator.annotate(frame, detections)
            labels = [
                f"{CLASSES[class_id]} {confidence:0.1f}"
                for _, _, confidence, class_id, _
                in detections
            ]
            frame = label_annotator.annotate(frame, detections, labels) 

        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        
        cv2.putText(frame, f"FPS: {int(fps)}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA) 

        writer.write(frame)
        
        display_frame(frame)
        if (cv2.waitKey(1) == ord('q')):
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

#@time_logger
def display_frame(frame):
    cv2.imshow("LiveCam", frame)

if __name__ == "__main__":
    main()
    