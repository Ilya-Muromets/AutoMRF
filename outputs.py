import os
import random
import numpy as np
import torch
import glob
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils.architectures.autoreg import SimpleClass
from utils.architectures.inceptionv4 import InceptionV4
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import compare_ssim as ssim
import natsort
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def rescale(arr, scale=255):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * scale)).astype(np.float64)

def T1fromMRF(MRF, regressor, model):
    device = torch.device("cuda:0,1")
    
    regressor.to(device)
    regressor.load_state_dict(torch.load(model))
    a = regressor.eval()

    MRF = MRF.reshape(1000,-1).T
    MRF = Variable(torch.from_numpy(MRF).type(torch.FloatTensor))
    MRF = MRF.to(device).view(-1,2,500)

    T1_array = np.array([])
    # break calculation into blocks so not to kill the GPU
    blocks = 256
    slice_len = int(np.ceil(MRF.size()[0]/blocks))
    max_len = len(MRF)
    for i in range(blocks):
        if i*slice_len >= max_len:
            break
        if i % (blocks//10) == 0:
            print(i)
        T1 = regressor(MRF[i*slice_len:(i+1)*slice_len])
        T1 = T1.data.max(1)[1]
        T1 = T1.data.cpu().numpy()
        T1_array = np.concatenate((T1_array, T1.flatten()))

    print("mean: ", np.mean(T1_array))
    return T1_array.reshape(320,320)

filenames = natsort.natsorted(glob.glob("/mikQNAP/augmented_data/MRF/*"))
filenames = filenames[-41:-1]

regressor = InceptionV4(num_classes=256)
for i in np.round(np.linspace(0,1,11),1):
    model = "models/alpha" + str(i)
    for j, filename in enumerate(filenames):
        MRF = np.load(filename)
        T1_recon = T1fromMRF(MRF, regressor, model)
        np.save("outputs/alpha" + str(i) + "trial" + str(j), T1_recon)