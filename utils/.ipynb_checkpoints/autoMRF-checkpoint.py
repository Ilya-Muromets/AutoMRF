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
# from utils.architectures.vde import VDE
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, SmoothL1Loss
from time import gmtime, strftime

class AutoClassMRF(object):
    def __init__(self, batchsize=128, epochs=10, workers=1, num_classes=8, alpha=0.9, model=None, model_name=None, steps_per_batch=1024):
        self.batchsize=batchsize
        self.num_epoch=epochs
        self.workers=workers
        self.num_classes=num_classes
        self.alpha = alpha
        self.model = model
        self.model_name = model_name
        self.steps_per_batch = steps_per_batch

    def fit(self, dataset, test_dataset):
        device = torch.device("cuda:0")
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
        classifier.to(device)

        # possibly load model for fine-tuning
        if self.model is not None:
            classifier.load_state_dict(torch.load(self.model))

        optimizer = optim.Adagrad(classifier.parameters(), lr=1e-3)
        
        test_acc = []

        class_sample_count = dataset.T1_class_counts
        print("Class sample count: ", class_sample_count)
        weights = 1 / torch.from_numpy(class_sample_count).type(torch.FloatTensor)

        # keep track of loss
        train_loss = []
        val_loss = []

        # loss is combined cross-entropy and MSE loss
        criterion_CE = CrossEntropyLoss()#weight=weights.to(device))
        criterion_MSE = MSELoss()
        start = time.time()

        def update_loss():
            if self.alpha >= 1:
                loss = criterion_CE(pred, T1)
            elif self.alpha <= 0:
                loss = criterion_MSE(pred_MSE,T1_MSE)
            else:
                loss = criterion_CE(pred, T1)*self.alpha
                loss += criterion_MSE(pred_MSE,T1_MSE)*(1-self.alpha)
            return loss

        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                if i > self.steps_per_batch:
                    break

                MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor), data[2].type(torch.LongTensor)
                MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                MRF, T1, T2 = MRF.to(device), T1.to(device), T2.to(device)
                
                # select nonzero T1 locs
                nonzero_locs = torch.where(torch.norm(MRF, dim=(1,2)) > 0.1,
                                             torch.tensor(1).to(device), 
                                             torch.tensor(0).to(device))
                nonzero_locs = nonzero_locs.type(torch.ByteTensor)
                MRF = MRF[nonzero_locs]
                T1 = T1[nonzero_locs]

                # catch if we threw out all the batch
                if MRF.size()[0] == 0:
                    break

                optimizer.zero_grad()
                classifier = classifier.train()
                pred = classifier(MRF).view(MRF.size()[0],-1)

                # convert class probabilities to choice for MSE
                pred_MSE = Variable(pred.data.max(1)[1].type(torch.FloatTensor), requires_grad=True)
                pred_MSE = pred_MSE.to(device)/self.num_classes # normalize
                T1_MSE = T1.type(torch.FloatTensor).to(device)/self.num_classes

                loss = update_loss()
                # print(loss)
                # print()
                loss.backward()
                optimizer.step()
                
                if i % (self.steps_per_batch//40) == 0:
                    ce_loss = np.float32(criterion_CE(pred, T1).item()*self.alpha)
                    mse_loss =  np.float32(criterion_MSE(pred_MSE,T1_MSE).item()*(1-self.alpha))
                    train_loss.append([epoch, i + epoch*self.steps_per_batch, ce_loss, mse_loss])

                    print('[%d: %d/%d] MSE loss: %f CE loss: %f' %(epoch, i, self.steps_per_batch, mse_loss, ce_loss))
    
                if i % (self.steps_per_batch//5) == 0:

                    j, data = next(enumerate(testdataloader, 0))
                    MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor),  data[2].type(torch.LongTensor)
                    MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                    MRF, T1, T2 = MRF.to(device), T1.to(device), T2.to(device)

                    classifier = classifier.eval()
                    pred = classifier(MRF).view(MRF.size()[0],-1)

                    pred_MSE = Variable(pred.data.max(1)[1].type(torch.FloatTensor), requires_grad=True)
                    pred_MSE = pred_MSE.to(device)/self.num_classes # normalize
                    T1_MSE = T1.type(torch.FloatTensor).to(device)/self.num_classes

                    loss = update_loss()

                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(T1.data).cpu().sum()
                    print(pred_choice[0:10])
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, self.steps_per_batch, blue('test'), loss.item(), correct.item()/float(self.batchsize)))
                    print("Time elapsed: ", (time.time() - start)/60, " minutes")

            if self.model_name is None:
                torch.save(classifier.state_dict(),"models/model" + curr_date)
                np.save("outputs/loss" + curr_date, np.array(train_loss))

            else:
                torch.save(classifier.state_dict(),"models/" + self.model_name)
                np.save("outputs/loss" + self.model_name, np.array(train_loss))


            # np.save("outputs/val_loss" + curr_date, np.array(val_loss))

class AutoRegMRF(object):
    def __init__(self, batchsize=128, epochs=10, workers=1, alpha=0.9, model=None, model_name=None, steps_per_batch=1024):
        self.batchsize=batchsize
        self.num_epoch=epochs
        self.workers=workers
        self.alpha = alpha
        self.model = model
        self.model_name = model_name
        self.steps_per_batch = steps_per_batch


    def fit(self, dataset, test_dataset):
        device = torch.device("cuda:0")
        curr_date = strftime("%d-%H:%M:%S", gmtime())
        blue = lambda x:'\033[94m' + x + '\033[0m'

        # initialise dataloader as single thread else socket errors
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))


        regressor = InceptionV4(1)
        regressor.to(device)

        # possibly load model for fine-tuning
        if self.model is not None:
            regressor.load_state_dict(torch.load(self.model))

        optimizer = optim.Adagrad(regressor.parameters(), lr=1e-3)
        
        test_acc = []

        # keep track of loss
        train_loss = []
        val_loss = []

        criterion = MSELoss()
        start = time.time()

        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                if i > self.steps_per_batch:
                    break

                MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(torch.FloatTensor)
                MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                MRF, T1, T2 = MRF.to(device), T1.to(device), T2.to(device)
                
                # select nonzero T1 locs
                nonzero_locs = torch.where(torch.norm(MRF, dim=(1,2)) > 0.1,
                                             torch.tensor(1).to(device), 
                                             torch.tensor(0).to(device))
                nonzero_locs = nonzero_locs.type(torch.ByteTensor)
                MRF = MRF[nonzero_locs]
                T1 = T1[nonzero_locs]

                # catch if we threw out all the batch
                if MRF.size()[0] == 0:
                    break

                optimizer.zero_grad()
                regressor = regressor.train()
                pred = regressor(MRF).view(MRF.size()[0])
                loss = criterion(pred, T1)
                loss.backward()
                optimizer.step()
                
                if i % (self.steps_per_batch//100) == 0:
                    train_loss.append([epoch, i + epoch*self.steps_per_batch, np.float32(loss.item())/(2**16)])
                    print('[%d: %d/%d] train loss: %f' %(epoch, i, self.steps_per_batch, np.float32(loss.item())/(2**16)))
    
                if i % (self.steps_per_batch//4) == 0:
                    j, data = next(enumerate(testdataloader, 0))
                    MRF, T1, T2 = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor),  data[2].type(torch.FloatTensor)
                    MRF, T1, T2 = Variable(MRF), Variable(T1), Variable(T2)
                    MRF, T1, T2 = MRF.to(device), T1.to(device), T2.to(device)

                    regressor = regressor.eval()
                    pred = regressor(MRF).view(MRF.size()[0])
                    loss = criterion(pred, T1)
                    loss.backward()
                    val_loss.append([epoch, i + epoch*self.steps_per_batch, np.float32(loss.item())/(2**16)])

                    print(pred[0:10])
                    print('[%d: %d/%d] %s loss: %f ' %(epoch, i, self.steps_per_batch, blue('test'), np.float32(loss.item())/(2**16)))
                    print("Time elapsed: ", (time.time() - start)/60, " minutes")

            if self.model_name is None:
                torch.save(regressor.state_dict(),"models/model" + curr_date)
                np.save("outputs/train_loss" + curr_date, np.array(train_loss))
                np.save("outputs/val_loss" + curr_date, np.array(val_loss))    
            else:
                torch.save(regressor.state_dict(),"models/" + self.model_name)
                np.save("outputs/train_loss"  + self.model_name, np.array(train_loss))
                np.save("outputs/val_loss"  + self.model_name, np.array(val_loss))    

# class AutoVDE(object):
#     def __init__(self, numbatches=256, batchsize=512, input_size=500, encoder_size=100, verbose=False, workers=8):
#         self.real_encoder = VDE(input_size=input_size, encoder_size=encoder_size, verbose=verbose, cuda=True)
#         self.imag_encoder = VDE(input_size=input_size, encoder_size=encoder_size, verbose=verbose, cuda=True)
#         self.numbatches = numbatches
#         self.batchsize = batchsize
#         self.workers = workers

#     def fit(self, dataset):
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
#                                                 shuffle=True, num_workers=int(self.workers))

#         for i, data in enumerate(dataloader, 0):
#             if i > self.numbatches:
#                 break
#             if i % (self.numbatches//10) == 0:
#                 print("Batch: ", i)
#             MRF, T1, T2 = data
#             batch_real = MRF[:,0,:].squeeze()
#             batch_imag = MRF[:,1,:].squeeze()
#             print(batch_real.size())
#             print(batch_imag.size())


#             self.real_encoder.fit(batch_real)
#             self.imag_encoder.fit(batch_imag)

            

