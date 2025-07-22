import torch
import numpy as np

class LinearDecayScheduler:
    def __init__(self, min_weight=0.1, 
                 max_weight=1.0, 
                 num_tokens=256,
                 loss_type='L2'):
        
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights = torch.linspace(min_weight, max_weight, num_tokens)

        assert loss_type in ['L1', 'L2']
        self.loss_type = loss_type

    def get_weight(self):
        return self.weights

    # def get_regularization_loss(self, m0, m1, m2):
    #     # Take openllama for example
    #     # m0: [L, N, D]
    #     # m1: [bs, L, N, D]
    #     # m2: [bs, L, N, D]
    #     if self.loss_type == 'L2':
    #         return torch.norm(m1 - m0.unsqueeze(0)) + torch.norm((m2 - m1))
    #     elif self.loss_type == 'L1':
    #         return torch.mean(torch.abs(m1 - m0.unsqueeze(0))) + torch.mean(torch.abs(m2 - m1))
    #     else:
    #         raise ValueError(f"Unknown loss type: {self.loss_type}")

    def get_regularization_loss(self, m0, m1):
        # Take openllama for example
        # m0: [L, N, D]
        # m1: [bs, L, N, D]
        # m2: [bs, L, N, D]
        if self.loss_type == 'L2':
            difference_norm = torch.norm((m1 - m0.unsqueeze(0)).transpose(0, 2).reshape(m0.shape[1], -1), dim=1)
            m0_norm = torch.norm(m0.transpose(0, 1).reshape(m0.shape[1], -1), dim=1)
            return torch.mean((difference_norm > m0_norm * self.weights.to(m0.device)) * (difference_norm - m0_norm * self.weights.to(m0.device))), \
                m0_norm.mean().detach(), difference_norm.mean().detach()
        
        elif self.loss_type == 'L1':
            # return torch.mean(torch.abs(m1 - m0.unsqueeze(0)))
            raise NotImplementedError
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


