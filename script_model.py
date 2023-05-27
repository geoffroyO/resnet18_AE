import torch
from preprocess import get_data
from train import TrainerAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    num_epochs=10
    lr=1e-4
    batch_size=1024
    nb_channels=3
    main_path='/gpfswork/rech/hlp/uha64uw/data/PPMI_longi/'
    save_path='/gpfswork/rech/hlp/uha64uw/res/resnet18/'
    patch_size=64
    fold=0
    
args = Args()
data_loader = get_data(args)
trainer = TrainerAE(args, data_loader, device)
trainer.train()