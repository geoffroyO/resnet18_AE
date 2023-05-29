import torch

class ComputeLoss:
    def forward(self, x, x_hat):
        return torch.mean((x-x_hat).pow(2))
    
    def forward_test(self, x, x_hat):
        return torch.mean((x-x_hat).pow(2), dim=(1, 2, 3))