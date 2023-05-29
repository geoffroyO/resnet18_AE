import torch
from torch.autograd import Variable

class ComputeLoss:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def forward(self, x, x_hat):
        reconst_loss = torch.mean((x-x_hat).pow(2))
        return Variable(reconst_loss, requires_grad=True)
    
    def forward_test(self, x, x_hat):
        reconst_loss = torch.mean((x-x_hat).pow(2), dim=(1, 2, 3))
        return reconst_loss