from ultralytics import YOLO
from settings import Settings
import yolo_classes as classes

settings = Settings.load_from_env(classes.ALL)
if settings.optimizations is not None:
    model = YOLO(settings.model)
    optimizations = settings.optimizations
    model.export(
        format="onnx", 
        imgsz=optimizations.imgsz, 
        int8=optimizations.int8)