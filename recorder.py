import cv2

from settings import Settings

class Recorder:
    def __init__(self, settings:Settings):
        self.writer = cv2.VideoWriter(f"rec_{settings.model}.mp4", 
                                      cv2.VideoWriter_fourcc(*'DIVX'), 
                                      30, 
                                      (settings.capture_width, settings.capture_height))
    def write(self, frame):
        self.writer.write(frame)
        
    def release(self):
        self.writer.release()
        
    def __del__(self):
        self.release()