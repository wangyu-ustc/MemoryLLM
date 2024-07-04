import torch
import numpy as np

class LinearDecayScheduler:
    def __init__(self, min_weight=1e-4, max_weight=1e-3, max_steps=20000):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_steps = max_steps
    
    def get_weight(self, step):
        # step 0: max_weight
        # step inf: min_weight
        if step < self.max_steps:
            return self.max_weight - (self.max_weight - self.min_weight) * step / self.max_steps
        else:
            return self.min_weight
