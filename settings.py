from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Optimizations:
    onnx:bool=True
    imgsz:int=128
    int8:bool=True
    half:bool=False
    dynamic:bool=False
    simplify:bool=False
    opset:int=None

@dataclass
class Runtime:
    name:str
    cv2_dnn_backend:str
    cv2_dnn_target:str

@dataclass
class Settings:
    classes:List[str]
    model:str
    score_threshold:float=0.25
    nms_threshold:float=0.5
    nms_class_agnostic:bool=True
    capture_width:int=800
    capture_height:int=600
    capture_source:int=0
    recording_annotations:bool=True
    recording_fps:int=30
    optimizations:Optimizations=None
    runtime:Runtime=None    
    
    @staticmethod
    def load_from_env(classes:List[str]) -> "Settings":
        #create a new settings object and populate it with the environment variables 
        #that are defined in the settings.env file
        import os, cv2
        
        model = os.getenv("MODEL","yolov8n")
        score_threshold = float(os.getenv("SCORE_THRESHOLD", "0.3"))
        nms_threshold = float(os.getenv("NMS_THRESHOLD","0.6"))
        nms_class_agnostic = bool(os.getenv("NMS_CLASS_AGNOSTIC", "0"))
        
        export_onnx = bool(os.getenv("EXPORT_ONNX","1"))
        img_size = int(os.getenv("IMG_SIZE", "224"))
        quantize_int8 = bool(os.getenv("QUANTIZE_INT8","1"))
        quantize_fp16 = bool(os.getenv("QUANTIZE_FP16", "0"))
        dynamic = bool(os.getenv("DYNAMIC"))
        simplify = bool(os.getenv("SIMPLIFY"))
        opset = int(os.getenv("OPSET","12"))
        
        capture_width = int(os.getenv("CAPTURE_WIDTH", "800"))
        capture_height = int(os.getenv("CAPTURE_HEIGHT", "600"))
        capture_source = int(os.getenv("CAPTURE_SOURCE", "0"))
        recording_annotations = bool(os.getenv("RECORDING_ANNOTATIONS"))
        recording_fps = int(os.getenv("RECORDING_FPS", "1"))
        
        cv2_runtime = bool(os.getenv("CV2_RUNTIME", "1"))
        cv2_dnn_backend = int(os.getenv("CV2_DNN_BACKEND", cv2.dnn.DNN_BACKEND_DEFAULT))
        cv2_dnn_target = int(os.getenv("CV2_DNN_TARGET", cv2.dnn.DNN_TARGET_CPU))
        
        optimizations = None
        if export_onnx:
            optimizations = Optimizations(onnx=export_onnx, imgsz=img_size, int8=quantize_int8, half=quantize_fp16, dynamic=dynamic, simplify=simplify, opset=opset)

        if cv2_runtime:
            runtime = Runtime(name="cv2", cv2_dnn_backend=cv2_dnn_backend, cv2_dnn_target=cv2_dnn_target)
            
        return Settings(classes=classes, 
                        model=model, 
                        score_threshold=score_threshold, 
                        nms_threshold=nms_threshold, 
                        nms_class_agnostic=nms_class_agnostic, 
                        capture_width=capture_width, 
                        capture_height=capture_height, 
                        capture_source=capture_source, 
                        recording_annotations=recording_annotations, 
                        recording_fps=recording_fps, 
                        optimizations=optimizations,
                        runtime=runtime)