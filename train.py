import torch
from torch import optim

from barbar import Bar

from model import AE
from forward_step import ComputeLoss

class TrainerAE:
    def __init__(self, args, data, device):
        self.train_loader = data
        self.device = device
        self.args = args


    def train(self):
        self.model = AE().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.device)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                x_hat= self.model(x)

                loss = self.compute.forward(x, x_hat)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training AE... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
                

        

