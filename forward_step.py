import torch

class ComputeLoss:
    def forward(self, x, x_hat):
        return torch.mean((x-x_hat).pow(2))
    
    def forward_test(self, x, x_hat, args):
        ps = args.patch_size
        idx = ps//2
        return (x-x_hat).pow(2)[:, :, idx, idx]