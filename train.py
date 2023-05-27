import torch
from torch import optim

from barbar import Bar

from model import AE
from forward_step import ComputeLoss

import numpy as np

from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.energy_meter import EnergyContext

class TrainerAE:
    def __init__(self, args, data, device):
        self.train_loader = data
        self.device = device
        self.args = args
        self.csv_handler = CSVHandler(self.args.save_path + 'result.csv')


    def train(self):
        self.model = AE().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.device)
        self.model.train()
        min_loss = torch.inf
        hist_loss = []
        with EnergyContext(handler=self.csv_handler, start_tag='0') as ctx:
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
                    break
                if total_loss < min_loss:
                        min_loss = total_loss
                        torch.save(self.model.state_dict(), self.args.save_path + 'best-model-parameters.pt')
                        torch.save(optimizer.state_dict(), self.args.save_path + 'best-optim-parameters.pt')
                
                print('Training AE... Epoch: {}, Loss: {:.3f}'.format(
                    epoch, total_loss/len(self.train_loader)))
                hist_loss.append(total_loss/len(self.train_loader))
                ctx.record(tag=f'{epoch+1}')
                break
        self.csv_handler.save_data()
        hist_loss = np.array(hist_loss)
        np.save(self.args.save_path + 'hist_loss.npy', hist_loss)
                

        

