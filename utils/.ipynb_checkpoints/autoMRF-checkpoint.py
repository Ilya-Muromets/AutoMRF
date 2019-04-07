from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import glob
import time
import matplotlib.pyplot as plt
from utils.autoreg.autoreg import SimpleReg
from torch.nn.modules.loss import L1Loss

class AutoMRF(object):
    def __init__(self, batchsize=128, epochs=10, workers=4):
        self.batchsize=batchsize
        self.num_epoch=epochs
        self.workers=workers

    def fit(self, dataset, test_dataset):
        blue = lambda x:'\033[94m' + x + '\033[0m'

        # initialise dataloader as single thread else socket errors
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))

        # try:
        #     os.makedirs(self.outf)
        # except OSError:
        #     pass

        # if self.model != '':
        #     classifier.load_state_dict(torch.load(self.model))


        regressor = SimpleReg()
        regressor.cuda()

        optimizer = optim.SGD(regressor.parameters(), lr=0.001, momentum=0.99)
        
        num_batch = len(dataset)//self.batchsize
        test_acc = []
        loss_function = L1Loss()
        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor),  data[2].type(torch.FloatTensor)
                MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()

                optimizer.zero_grad()
                regressor = regressor.train()
                pred = regressor(MRF)
                # print((pred.view(2,2),torch.stack((T1,T2))))

                loss = loss_function(pred.view(2,self.batchsize), torch.stack((T1,T2)))
                if i % (num_batch//40) == 0:
                    print('[%d: %d/%d] train loss: %f' %(epoch, i, num_batch, loss.item()))
                loss.backward()
                optimizer.step()
    
                if i % (num_batch//5) == 0:
                    j, data = next(enumerate(testdataloader, 0))
                    MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor),  data[2].type(torch.FloatTensor)
                    MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                    MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()
                    regressor = regressor.train()
                    pred = regressor(MRF)
                  
                    loss = loss_function(pred.view(2,self.batchsize), torch.stack((T1,T2)))
                    print('[%d: %d/%d] %s loss: %f' %(epoch, i, num_batch, blue('test'), loss.item()))


            torch.save(regressor.state_dict(),"model")
