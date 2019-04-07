import numpy as np
import glob
import torch
import natsort
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class MagLoader(Dataset):
    def __init__(self, mag_path='', T1_path='', T2_path='', max_scans=999999):
        self.mag_data = []
        self.T1_data = []
        self.T2_data = []
        self.data_shape = None
        
        # load files
        for filename in natsort.natsorted(glob.glob(mag_path))[0:max_scans]:
            self.mag_data.append(Variable(torch.from_numpy(np.load(filename))))
        print("loaded: ", mag_path)
        print(len(self.mag_data), " files")
        print()
        
        for filename in natsort.natsorted(glob.glob(T1_path))[0:max_scans]:
            self.T1_data.append(Variable(torch.from_numpy(np.load(filename))))
        print("loaded: ", T1_path)
        print(len(self.T1_data), " files")
        print()

        for filename in natsort.natsorted(glob.glob(T2_path))[0:max_scans]:
            self.T2_data.append(Variable(torch.from_numpy(np.load(filename))))
        print("loaded: ", T2_path)
        print(len(self.T2_data), " files")
        print()

        # check we loaded correct number and shape
        assert len(self.mag_data) == len(self.T1_data) == len(self.T2_data)
        assert self.mag_data[0][0].shape == self.T1_data[0][0].shape == self.T2_data[0][0].shape, "shape mismatch"
        
        self.data_shape = self.mag_data[0][0].shape
        print("slice shape: ", self.data_shape)

    def __len__(self):
        return len(self.mag_data)*np.product(self.data_shape)

    def __getitem__(self, idx):
        #retrieve 
        list_idx = idx//np.product(self.data_shape)
        matrix_idx = idx%np.product(self.data_shape)
        row_idx = matrix_idx//self.data_shape[1]
        column_idx = matrix_idx%self.data_shape[1]

        return self.mag_data[list_idx][:,row_idx,column_idx], self.T1_data[list_idx][0][row_idx,column_idx], self.T2_data[list_idx][0][row_idx,column_idx]