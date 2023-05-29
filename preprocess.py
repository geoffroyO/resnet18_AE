import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dipy.io.image import load_nifti
from tqdm import tqdm


def load_folds(fold, main_path, mode):
    if mode == "Train":
        folds_controls = pd.read_excel(main_path + 'controls.xlsx')

        selection = folds_controls[folds_controls["fold_{}".format(fold)] == 1]
        controls_names = selection['Subject'].to_numpy(dtype=str)
        controls_ages = selection['Age'].to_numpy(dtype=str)
        controls_it = [(name, age) for name, age in zip(controls_names, controls_ages)]

        return controls_it

    if mode == "Test":
        selection = folds_controls[folds_controls["fold_{}".format(fold)] == 2]
        controls_names_test = selection['Subject'].to_numpy(dtype=str)
        controls_ages_test = selection['Age'].to_numpy(dtype=str)

        
        controls_test_it = [(name, age) for name, age in zip(controls_names_test, controls_ages_test)]

        patients = pd.read_excel(main_path + 'patients.xlsx')
        patients_names, patients_ages = patients['Subject'], patients['Age']
        patients_it = [(name, age) for name, age in zip(patients_names, patients_ages)]

        return controls_test_it, patients_it
    

class Data(Dataset):
    def __init__(self, subjects_it, args, main_path):
        self.patch_size = args.patch_size
        self.data = {} 
        self.idx = []
        for subject_name, subject_age in tqdm(subjects_it):
            path_atlas = main_path + f'{subject_name}/atlas/{subject_age}/'
            path_T1 = main_path + f'{subject_name}/T1/{subject_age}/'
            path_diffusion = main_path + f'{subject_name}/diffusion/{subject_age}/'

            atlas_neuro, _ = load_nifti(path_atlas + 'wneuromorphometrics.nii')

            FA, _ = load_nifti(path_diffusion + '/FA_masked_norm.nii')
            MD, _ = load_nifti(path_diffusion + '/MD_masked_norm.nii')
            T1, _ = load_nifti(path_T1 + '/T1_masked_norm.nii')
            
            self.data[(subject_name, subject_age)] = np.stack([FA, MD, T1])

            x, y, z = np.where(atlas_neuro > 0)
            sn = [subject_name] * len(x)
            sa = [subject_age] * len(x)
            self.idx += list(zip(sn, sa, x, y, z))
            
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        sub_name, sub_age, x, y, z = self.idx[index]
        if self.patch_size % 2 == 0:
            tmp = self.patch_size // 2
            patch = self.data[(sub_name, sub_age)][:, (x-tmp):(x+tmp), (y-tmp):(y+tmp), z] 
        else:
            tmp = self.patch_size // 2
            patch = self.data[(sub_name, sub_age)][:, (x-tmp):(x+tmp+1), (y-tmp):(y+tmp+1), z] 

        _, H, W = patch.shape
        if H + W != self.patch_size * 2:
            pad_widths = [(0, 0), (0, self.patch_size - H), (0, self.patch_size - W)]
            patch = np.pad(patch, pad_widths, mode='constant')
        return patch

def get_data(args, mode="Train"):
    if mode == "Train":
        controls_it = load_folds(args.fold, args.main_path, mode)
        main_path = args.main_path + "controls/"
        train = Data(controls_it, args, main_path)
        dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        return dataloader_train
