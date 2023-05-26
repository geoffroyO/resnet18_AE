
import torch
from torch import optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from barbar import Bar
from tqdm import tqdm

from model import AE
from forward_step import ComputeLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    num_epochs=20
    lr=1e-4
    batch_size=1024
    nb_channels=3
    main_path='/Users/geoffroy/Documents/dataG/PPMI_longi/'
    patch_size=64
    fold=0
    
args = Args()

class Data(Dataset):
    def __init__(self, N):
        self.data = np.random.normal(0, 1, size=(N, 3, 64, 64)).astype(np.float32)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

train_loader = DataLoader(Data(1000000))

model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

compute = ComputeLoss(model, device)
model.train()
min_loss = torch.inf
for epoch in range(args.num_epochs):
    total_loss = 0
    for x in tqdm(train_loader):
        x = x.float().to(device)
        optimizer.zero_grad()
        
        x_hat= model(x)

        loss = compute.forward(x, x_hat)
        if loss < min_loss:
            min_loss = loss.data
            torch.save(model.state_dict(), args.save_path + 'best-model-parameters.pt')
            torch.save(optimizer.state_dict(), args.save_path + 'best-optim-parameters.pt')
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
    print('Training AE... Epoch: {}, Loss: {:.3f}'.format(
            epoch, total_loss/len(train_loader)))