import torch
from preprocess import get_data
from train import TrainerAE
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="batch size", default=1024, type=int)
    parser.add_argument("-ps", "--patch_size", help="patch size", default=64, type=int)
    parser.add_argument("-p", "--main_path", help="path of the data", type=str)
    parser.add_argument("-sp", "--save_path", help="path to save result", type=str)
    parser.add_argument("-m", "--model", help="model of AE", default='resnet18', type=str)
    parser.add_argument("-e", "--epochs", help="epochs", default=5, type=int)
    args_in = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Args:
        num_epochs=args_in.epochs
        lr=1e-4
        batch_size=args_in.batch_size
        nb_channels=3
        main_path=args_in.main_path
        save_path=args_in.save_path
        patch_size=args_in.patch_size
        fold=3
        
    args = Args()
    if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    data_loader = get_data(args)
    print(args_in.model)
    trainer = TrainerAE(args, data_loader, device, args_in.model)
    trainer.train()
