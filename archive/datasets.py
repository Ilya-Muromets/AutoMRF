import numpy as np
import glob
import torch
import natsort
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class ClassComplexLoader(Dataset):
    def __init__(self, data_path='', T1_path='', T2_path='', max_scans=999999, num_classes=8):
        self.complex_data = []
        self.T1_data = []
        self.T2_data = []
        self.data_shape = None
        self.T1_class_counts = np.zeros(num_classes)
        self.T2_class_counts = np.zeros(num_classes)


        # load files
        for filename in natsort.natsorted(glob.glob(data_path))[0:max_scans]:
            self.complex_data.append(Variable(torch.from_numpy(np.load(filename))))
        print("loaded: ", data_path)
        print(len(self.complex_data), " files")
        print()
        
        for filename in natsort.natsorted(glob.glob(T1_path))[0:max_scans]:
            # round to int within class number
            T1 = np.load(filename).astype(np.float64)
            T1 = T1/(2**16-1) # assuming uint16
            T1 = np.round((num_classes-0.51)*T1, 0).astype(np.int)

            # keep track of how many in each class
            self.T1_class_counts += np.bincount(T1.flatten(), minlength=num_classes)
            self.T1_data.append(Variable(torch.from_numpy(T1)))
        print("loaded: ", T1_path)
        print(len(self.T1_data), " files")
        print()

        for filename in natsort.natsorted(glob.glob(T2_path))[0:max_scans]:
            # round to int within class number
            T2 = np.load(filename).astype(np.float64)
            T2 = T2/(2**16-1) # assuming uint16
            T2 = np.round((num_classes-0.51)*T2, 0).astype(np.int)

            # keep track of how many in each class
            self.T2_class_counts += np.bincount(T2.flatten(), minlength=num_classes)
            self.T2_data.append(Variable(torch.from_numpy(T2)))
        print("loaded: ", T2_path)
        print(len(self.T2_data), " files")
        print()

        # check we loaded correct number and shape
        assert len(self.complex_data) == len(self.T1_data) == len(self.T2_data)
        assert self.complex_data[0][0].shape == self.T1_data[0][0].shape == self.T2_data[0][0].shape, "shape mismatch"
        
        self.data_shape = self.complex_data[0][0].shape
        print()
        print("slice shape: ", self.data_shape)
        print()

    def __len__(self):
        return len(self.complex_data)*np.product(self.data_shape)

    def __getitem__(self, idx):
        #retrieve 
        list_idx = idx//np.product(self.data_shape)
        matrix_idx = idx%np.product(self.data_shape)
        row_idx = matrix_idx//self.data_shape[1]
        column_idx = matrix_idx%self.data_shape[1]
        T1 = self.T1_data[list_idx][0][row_idx,column_idx]
        T2 = self.T2_data[list_idx][0][row_idx,column_idx]

        complex_datum = self.complex_data[list_idx][:,row_idx,column_idx] # index into matrix
        # stack real component on top of imaginary component of data, shape now (2,len/2)
        complex_datum = torch.stack((complex_datum[0:len(complex_datum)//2],
                                     complex_datum[len(complex_datum)//2:len(complex_datum)]))
        return complex_datum, T1, T2