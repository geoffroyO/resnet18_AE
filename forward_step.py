import torch
from torch.autograd import Variable

class ComputeLoss:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def forward(self, x, x_hat):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))
        return Variable(reconst_loss, requires_grad=True)