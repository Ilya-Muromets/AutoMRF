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
from utils.architectures.autoreg import SimpleClass, SimpleReg
from utils.architectures.inceptionv4 import InceptionV4
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, SmoothL1Loss
from time import gmtime, strftime

class AutoClassMRF(object):
    def __init__(self, batchsize=128, epochs=10, workers=1, num_classes=8, alpha=0.9, model=None, model_name=None):
        self.batchsize=batchsize
        self.num_epoch=epochs
        self.workers=workers
        self.num_classes=num_classes
        self.alpha = alpha
        self.model = model
        self.model_name = model_name

    def fit(self, dataset, test_dataset):
        curr_date = strftime("%d-%H:%M:%S", gmtime())
        blue = lambda x:'\033[94m' + x + '\033[0m'

        # initialise dataloader as single thread else socket errors
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))


        classifier = InceptionV4(self.num_classes)
        classifier.cuda()

        # possibly load model for fine-tuning
        if self.model is not None:
            classifier.load_state_dict(torch.load(self.model))

        optimizer = optim.Adadelta(classifier.parameters())
        
        num_batch = len(dataset)//self.batchsize
        test_acc = []

        class_sample_count = dataset.T1_class_counts
        print("Class sample count: ", class_sample_count)
        weights = 1 / torch.from_numpy(class_sample_count).type(torch.FloatTensor)

        # loss is combined cross-entropy and MSE loss
        criterion_CE = CrossEntropyLoss(weight=weights.cuda())
        criterion_MSE = MSELoss()

        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor), data[2].type(torch.LongTensor)
                MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()
                
                optimizer.zero_grad()
                classifier = classifier.train()
                pred = classifier(MRF).view(self.batchsize,-1)
                loss = criterion_CE(pred, T1)*self.alpha
                # convert predictions to integer class predictions, add square distance to loss
                loss += criterion_MSE(pred.data.max(1)[1].type(torch.FloatTensor),T1.type(torch.FloatTensor))*(1-self.alpha)
                loss.backward()
                optimizer.step()
                
                if i % (num_batch//40) == 0:
                    print('[%d: %d/%d] train loss: %f' %(epoch, i, num_batch, loss.item()))
    
                if i % (num_batch//5) == 0:
                    j, data = next(enumerate(testdataloader, 0))
                    MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor),  data[2].type(torch.LongTensor)
                    MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                    MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()
                    loss = criterion_CE(pred, T1)*self.alpha
                    loss += criterion_MSE(pred.data.max(1)[1].type(torch.FloatTensor),T1.type(torch.FloatTensor))*(1-self.alpha)

                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(T1.data).cpu().sum()
                    print(pred_choice[0:10])
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(self.batchsize)))

            if self.model_name is None:
                torch.save(classifier.state_dict(),"models/model" + curr_date)
            else:
                torch.save(classifier.state_dict(),"models/" + self.model_name)


class AutoRegMRF(object):
    def __init__(self, batchsize=128, epochs=10, workers=1, model=None, model_name=None):
        self.batchsize=batchsize
        self.num_epoch=epochs
        self.workers=workers
        self.model = model
        self.model_name = model_name

    def fit(self, dataset, test_dataset):
        curr_date = strftime("%d-%H:%M:%S", gmtime())
        print("saving model to: " + "models/model" + curr_date)
        blue = lambda x:'\033[94m' + x + '\033[0m'

        # initialise dataloader as single thread else socket errors
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))

        regressor = InceptionV4(1)
        regressor.cuda()

        # possibly load model for fine-tuning
        if self.model is not None:
            regressor.load_state_dict(torch.load(self.model))



        optimizer = optim.Adagrad(regressor.parameters(), lr=0.001)
        
        num_batch = len(dataset)//self.batchsize
        test_loss = []
        val_loss = []
        criterion = SmoothL1Loss()

        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(torch.FloatTensor)
                MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()
                
                optimizer.zero_grad()
                regressor = regressor.train()
                pred = regressor(MRF).view(self.batchsize,-1)
                loss = criterion(pred, T1)
                
                if i % (num_batch//40) == 0:
                    print('[%d: %d/%d] train loss: %f' %(epoch, i, num_batch, loss.item()))
                loss.backward()
                optimizer.step()
    
                if i % (num_batch//20) == 0:
                    test_loss.append(loss.item())

                    j, data = next(enumerate(testdataloader, 0))
                    MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor),  data[2].type(torch.FloatTensor)
                    MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                    MRF, T1, T2 = MRF.cuda(), T1.cuda(), T2.cuda()

                    regressor = regressor.eval()
                    pred = regressor(MRF).view(self.batchsize,-1)

                    # print(pred[0:10])
                    loss = criterion(pred, T1)
                    val_loss.append(loss.item())

                    print('[%d: %d/%d] %s loss: %f' %(epoch, i, num_batch, blue('test'), loss.item()))


            if self.model_name is None:
                torch.save(regressor.state_dict(),"models/model" + curr_date)
            else:
                torch.save(regressor.state_dict(),"models/" + self.model_name)

            np.save("outputs/test_loss" + curr_date, np.array(test_loss))
            np.save("outputs/val_loss" + curr_date, np.array(val_loss))
