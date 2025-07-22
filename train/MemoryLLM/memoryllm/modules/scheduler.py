import torch
import numpy as np

class LinearDecayScheduler:
    def __init__(self, end=1e-4, start=1e-3, duration=20000):
        self.end = end
        self.start = start
        self.duration = duration
    
    def get_ratio(self, step):
        # step 0: start
        # step inf: end
        if step < self.duration:
            return self.start - (self.start - self.end) * step / self.duration
        else:
            return self.end
