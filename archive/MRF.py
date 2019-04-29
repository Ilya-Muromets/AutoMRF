import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import torch
from torch.autograd import Variable
from pytorch_complex_tensor import ComplexTensor
D = np.load("data/dictionary_mrf.npy")
mrf_dict = scipy.io.loadmat('/mikRAID/jtamir/projects/MRF_direct_contrast_synthesis/data/DictionaryAndSequenceInfo/fp_dictionary.mat')
fp_dict = mrf_dict['fp_dict']
t1_list = mrf_dict['t1_list']
t2_list = mrf_dict['t2_list']
t1_t2_list = np.hstack((t1_list, t2_list))
device = torch.device("cpu")
# D = ComplexTensor(D.T).to(device)


for filename in glob.glob("data/MRF/*"):
    print(filename, "started")
    test = np.load(filename).T
    test = test[0:500] + 1j*test[500:1000]
    test = test.reshape(500,-1)
    # test = ComplexTensor(test).to(device)
    
    # res = torch.mm(D.t(), test)
    # res = res.cpu().data.numpy()
    res = np.abs(np.dot(D, test))
    
    indices = np.argmax(res, axis=0) # find max correlation

    T1_mapping = t1_list[indices].reshape(320,320).astype(np.uint16) # map to t1
    T2_mapping = t2_list[indices].reshape(320,320).astype(np.uint16) # map to t2
    PD_mapping = np.max(res, axis=0).reshape(320,320) # map to proton density
    np.save("data/T1map/" + filename.split("/")[-1], T1_mapping)
    np.save("data/T2map/" + filename.split("/")[-1], T2_mapping)
    np.save("data/PDmap/" + filename.split("/")[-1], PD_mapping)
    print(filename, "done")