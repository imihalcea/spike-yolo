import numpy as np
import supervision as sv
from typing import List, Dict, Tuple, Optional, Callable, Any
from settings import Settings
from stats import Stats, Computer

class Detector:
    def __init__(self, settings:Settings):
        self.settings = settings
    
    def detect(self, frame:np.ndarray) -> sv.Detections:
        raise NotImplementedError()
    
    def statistics(self) -> Dict[str, Stats]:
        raise NotImplementedError()
    