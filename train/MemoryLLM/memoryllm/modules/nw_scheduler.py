import torch
import numpy as np

class LinearDecayScheduler:
    def __init__(self, min_weight=1e-4, max_weight=1e-3, 
                delta_w=1e-5, cur_weight=None, bias=0.0, decrease_prob=1.0):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.step = 0
        self.previous_loss = None
        self.delta_w = delta_w
        self.bias = bias
        self.cur_weight = min_weight if cur_weight is None else cur_weight
        self.decrease_prob = decrease_prob
    
    def reset(self):
        self.step = 0
        self.previous_loss = None
        self.cur_weight = self.min_weight

    def update_weight(self, loss):

        if torch.isnan(loss).any():
            return

        if self.previous_loss is None:
            self.previous_loss = loss.item()
        else:
            if loss <= self.previous_loss:
                self.cur_weight += self.delta_w
            else:
                if np.random.random() < self.decrease_prob:
                    self.cur_weight -= self.delta_w

            new_avg_loss = (self.previous_loss * self.step + loss.item()) / (self.step + 1)
            if new_avg_loss < self.previous_loss:
                self.previous_loss = new_avg_loss + self.bias
            else:
                self.previous_loss = new_avg_loss
            
        self.step += 1
        self.cur_weight = max(self.min_weight, min(self.cur_weight, self.max_weight))

    def get_weight(self):
        return self.cur_weight

class ExponentialDecayScheduler:
    def __init__(self, min_weight=1e-4, max_weight=1e-3, delta_w=0.999):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.step = 0
        self.previous_loss = None
        self.delta_w = delta_w
        self.cur_weight = self.min_weight

    def get_weight(self):
        return self.cur_weight

    def update_weight(self, loss):

        if self.previous_loss is None:
            self.previous_loss = loss.item()

        else:
            if loss < self.previous_loss:
                self.cur_weight /= delta_w
            else:
                self.cur_weight *= delta_w
            
            # self.previous_loss = (self.previous_loss * self.step + loss.item()) / (self.step + 1)
            self.previous_loss = loss
        
        self.cur_weight = max(self.min_weight, min(self.cur_weight, self.max_weight))
        self.step += 1

        return self.cur_weight