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
from utils.architectures.autoreg import SimpleReg, ComplexReg, ComplexRNN, SimplestReg
from utils.architectures.inceptionv4 import InceptionV4
from torch.nn.modules.loss import CrossEntropyLoss
from time import gmtime, strftime

class AutoMRF(object):
    def __init__(self, batchsize=128, epochs=10, workers=4, num_classes=8):
        self.batchsize=batchsize
        self.num_epoch=epochs
        self.workers=workers
        self.num_classes=num_classes

    def fit(self, dataset, test_dataset):
        curr_date = strftime("%d-%H:%M:%S", gmtime())
        blue = lambda x:'\033[94m' + x + '\033[0m'

        # initialise dataloader as single thread else socket errors
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True)#, num_workers=int(self.workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True)#, num_workers=int(self.workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))

        # try:
        #     os.makedirs(self.outf)
        # except OSError:
        #     pass

        # if self.model != '':
        #     classifier.load_state_dict(torch.load(self.model))


        regressor = SimpleReg(self.num_classes)
        regressor.cuda()

        optimizer = optim.Adagrad(regressor.parameters(), lr=0.001)
        
        num_batch = len(dataset)//self.batchsize
        test_acc = []

        class_sample_count = dataset.T1_class_counts
        print("Class sample count: ", class_sample_count)
        weights = 1 / torch.from_numpy(class_sample_count).type(torch.FloatTensor)

        loss_function = CrossEntropyLoss(weight=weights.cuda())

        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor), data[2].type(torch.LongTensor)
                MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()
                
                optimizer.zero_grad()
                regressor = regressor.train()
                pred = regressor(MRF).view(self.batchsize,-1)
                loss = loss_function(pred, T1)
                
                if i % (num_batch//40) == 0:
                    print('[%d: %d/%d] train loss: %f' %(epoch, i, num_batch, loss.item()))
                loss.backward()
                optimizer.step()
    
                if i % (num_batch//5) == 0:
                    j, data = next(enumerate(testdataloader, 0))
                    MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor),  data[2].type(torch.LongTensor)
                    MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                    MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()

                    regressor = regressor.train()
                    pred = regressor(MRF).view(self.batchsize,-1)
                    # print(pred[0], T1[0])

                    # print("prediction: ", pred[0])
                    # print("T1: ", T1[0])
                    # print("T2: ", T2[0])
                  
                    loss = loss_function(pred, T1)
                    # print('[%d: %d/%d] %s loss: %f' %(epoch, i, num_batch, blue('test'), loss.item()))
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(T1.data).cpu().sum()
                    print(pred_choice[0:10])
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(self.batchsize)))


            torch.save(regressor.state_dict(),"models/model" + curr_date)
