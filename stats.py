import math

from dataclasses import dataclass
from datetime import time
from typing import Dict, Optional, Tuple

@dataclass
class Stats:
    min: float
    max: float
    mean: float
    sum: float
    std: float
    count: int

class Computer:
    def update(self, value:float, at:time):
        raise NotImplementedError()
    
    def compute(self) -> Stats:
        raise NotImplementedError()

class BasicStatsComputer(Computer):
    def __init__(self):
        self.min = 0
        self.max = 0
        self.mean = 0    
        self.std = 0
        self.variance = 0
        self.sum = 0
        self.count = 0
    
    def update(self, value:float, at:time):
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        self.sum += value
        self.count += 1
        self.mean = self.sum / self.count
        self.variance = (self.variance + (value - self.mean) ** 2) / self.count
        self.std = math.sqrt(self.variance)
        
    def compute(self) -> Stats:
        return Stats(
            min = self.min,
            max = self.max,
            mean = self.mean,
            sum = self.sum,
            std = self.std,
            count = self.count)

class DurationComputer(Computer):
    def __init__(self):
        self.computer = BasicStatsComputer()
        self.prev_frame_time = None
    
    def update(self, _:float, at:time):
        if self.prev_frame_time is None:
            self.prev_frame_time = at
        self.computer.update(at-self.prev_frame_time, at) 
        self.prev_frame_time = at
        
    def compute(self) -> Stats:
        return self.computer.compute()
        
class FpsComputer(Computer):
    def __init__(self):
        self.computer = BasicStatsComputer()
        self.prev_frame_time = 0
    
    def update(self, _:float, at:time):
        fps = 1/(at-self.prev_frame_time) 
        self.prev_frame_time = at
        self.computer.update(fps, at) 

        
    def compute(self) -> Stats:
        return self.computer.compute()        
    
class Computers:
    def __init__(self):
        self.computers:Dict[str, Computer] = {}
    
    def add(self, name:str, computer:Computer) -> "Computers":
        self.computers[name] = computer
        return self
    
    def update(self, name:str, value:float, at:time):
        computer = self.computers[name]
        if computer is not None:
            computer.update(value, at)
    
    def compute(self) -> Dict[str, Stats]:
        return {name: computer.compute() for name, computer in self.computers}    
    
    