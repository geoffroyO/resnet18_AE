import torch
from preprocess import get_data
from model import AE
from forward_step import ComputeLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dipy.io.image import load_nifti
from preprocess import load_folds

import numpy as np
import argparse
from barbar import Bar

from tqdm import tqdm

class DataTest(Dataset):
    def __init__(self, subject_name, subject_age, main_path, args):
        self.patch_size = args.patch_size

        path_atlas = main_path + f'{subject_name}/atlas/{subject_age}/'
        path_T1 = main_path + f'{subject_name}/T1/{subject_age}/'
        path_diffusion = main_path + f'{subject_name}/diffusion/{subject_age}/'

        atlas_neuro, _ = load_nifti(path_atlas + 'wneuromorphometrics.nii')

        FA, _ = load_nifti(path_diffusion + '/FA_masked_norm.nii')
        MD, _ = load_nifti(path_diffusion + '/MD_masked_norm.nii')
        T1, _ = load_nifti(path_T1 + '/T1_masked_norm.nii')
        
        self.data = np.stack([FA, MD, T1])

        x, y, z = np.where(atlas_neuro > 0)
        self.idx = list(zip(x, y, z))
            
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        x, y, z = self.idx[index]
        patch = self.data[:, x:x+self.patch_size, y:y+self.patch_size, z] 
        _, H, W = patch.shape
        if H + W != self.patch_size * 2:
            pad_widths = [(0, 0), (0, self.patch_size - H), (0, self.patch_size - W)]
            patch = np.pad(patch, pad_widths, mode='constant')
        return patch

def compute_quantile(args, model, device):
    data_loader = get_data(args, mode='Train_test')
    compute = ComputeLoss()
    losses = []

    print('Compute quantile...')
    with torch.no_grad():
        k = 0
        for x in Bar(data_loader):
            x = x.float().to(device)
            x_hat= model(x)
            loss = compute.forward_test(x, x_hat, args)
            losses.append(loss.detach().cpu().numpy())
            k += 1
            if k == 200:
                break
        
    losses = np.concatenate(losses, axis=0)
    print(losses.shape)
    return np.quantile(losses, args.alpha)

def inference_sub(args, model, device, empi_quantile):
    compute = ComputeLoss()
    controls_test_it, patients_it = load_folds(args.fold, args.main_path, 'Test')
    
    controls_test_ano = []
    print('Inference on controls test...')
    for control_test_name, control_test_age in tqdm(controls_test_it):
        control_test_data = DataTest(control_test_name, control_test_age, args.main_path + 'controls/', args)
        control_test_loader = DataLoader(control_test_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        losses = []
        with torch.no_grad():
            for x in control_test_loader:
                x = x.float().to(device)
                x_hat= model(x)
                loss = compute.forward_test(x, x_hat, args)
                losses.append(loss.detach().cpu().numpy())
        losses = np.concatenate(losses)
        print(losses.shape)
        controls_test_ano.append((losses < empi_quantile).sum())

    patients_ano = []
    print('Inference on patients...') # Parallel GPU?
    for patients_name, patients_age in tqdm(patients_it):
        patients_data = DataTest(patients_name, patients_age, args.main_path + 'patients/', args)
        patients_loader = DataLoader(patients_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        losses = []
        with torch.no_grad():
            for x in patients_loader:
                x = x.float().to(device)
                x_hat= model(x)
                loss = compute.forward_test(x, x_hat, args)
                losses.append(loss.item())
        losses = np.array(losses)
        patients_ano.append((losses < empi_quantile).sum())

    return np.array(controls_test_ano), np.array(patients_ano)

def gmean(controls_test_ano, patients_ano):
    N_max = max(controls_test_ano + patients_ano)
    g_mean = []

    for k in range(N_max):
        controls_test_ano_t = controls_test_ano >= k
        patients_ano_t = patients_ano >= k

        fp = controls_test_ano_t.sum()
        tp = patients_ano_t.sum()
        fn = (1 - patients_ano_t).sum()
        tn = (1 - controls_test_ano_t).sum()

        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)

        g_mean.append(np.sqrt(tpr * (1 - fpr)))

    return max(g_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="batch size", default=1024, type=int)
    parser.add_argument("-ps", "--patch_size", help="patch size", default=64, type=int)
    parser.add_argument("-p", "--main_path", help="path of the data", type=str)
    parser.add_argument("-sp", "--save_path", help="path to save result", type=str)
    parser.add_argument("-m", "--model", help="model of AE", default='resnet18', type=str)
    parser.add_argument("-a", "--alpha", help="FPR", default=.02, type=float)
    args_in = parser.parse_args()

    class Args:
        batch_size=args_in.batch_size
        nb_channels=3
        main_path=args_in.main_path
        save_path=args_in.save_path
        patch_size=args_in.patch_size
        alpha = args_in.alpha
        fold=3

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AE(args_in.model).to(device)
    model.load_state_dict(torch.load(args_in.save_path + 'best-model-parameters.pt'))
    model.eval()

    empi_quantile = compute_quantile(args, model, device)
    controls_test_ano, patients_ano = inference_sub(args, model, device, empi_quantile)

    print('**********************')
    print(gmean(controls_test_ano, patients_ano))
    print('**********************')


