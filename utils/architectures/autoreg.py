import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


# class SimpleReg(nn.Module):
#     def __init__(self, input_size=500):
#         super(SimpleReg, self).__init__()
#         self.conv1 = torch.nn.Conv1d(input_size, 256, 1)
#         self.conv2 = torch.nn.Conv1d(256, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 64, 1)
#         self.conv4 = torch.nn.Conv1d(64, 2, 1)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(64)

#     def forward(self, x):
#         batch_size = x.size()[0]
#         input_size = x.size()[1]
#         x = x.view(batch_size, input_size, -1)
#         x = F.relu(self.bn1(self.conv1(x))) #self.conv1(x)
#         x = F.relu(self.bn2(self.conv2(x))) #self.conv2(x)
#         x = F.relu(self.bn3(self.conv3(x))) #self.conv3(x)
#         x = self.conv4(x)
#         x = F.log_softmax(x)
#         return x


class SimpleReg(nn.Module):
    def __init__(self, input_size=500):
        super(SimpleReg, self).__init__()
        self.convT1_1 = torch.nn.Conv1d(1, 1, 32)
        self.convT1_2 = torch.nn.Conv1d(1, 1, 32, 2)
        self.convT1_3 = torch.nn.Conv1d(1, 1, 32, 4)
        self.convT1_4 = torch.nn.Conv1d(1, 1, 32, 8)
        self.convT1_5 = torch.nn.Conv1d(1, 1, 32, 16)
        self.fccT1 = torch.nn.Linear(911,1)

        self.convT2_1 = torch.nn.Conv1d(1, 1, 32)
        self.convT2_2 = torch.nn.Conv1d(1, 1, 32, 2)
        self.convT2_3 = torch.nn.Conv1d(1, 1, 32, 4)
        self.convT2_4 = torch.nn.Conv1d(1, 1, 32, 8)
        self.convT2_5 = torch.nn.Conv1d(1, 1, 32, 16)
        self.fccT2 = torch.nn.Linear(911,1)

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[1]
        x = x.view(batch_size, -1, input_size)
        x_T1 = torch.cat((self.convT1_1(x), self.convT1_2(x), 
                          self.convT1_3(x), self.convT1_4(x), 
                          self.convT1_5(x)), 2)
        x_T1 = torch.sigmoid(x_T1)

        x_T2 = torch.cat((self.convT2_1(x), self.convT2_2(x), 
                    self.convT2_3(x), self.convT2_4(x), 
                    self.convT2_5(x)), 2)
        x_T2 = torch.sigmoid(x_T2)

        x_T1 = self.fccT1(x_T1)
        x_T2 = self.fccT1(x_T2)
        
        return torch.cat((x_T1,x_T2),1)

class SimplestReg(nn.Module):
    def __init__(self, input_size=500):
        super(SimplestReg, self).__init__()
        self.convT1_1 = torch.nn.Conv1d(2, 8, 3)
        self.convT1_2 = torch.nn.Conv1d(8, 16, 3)
        self.convT1_3 = torch.nn.Conv1d(16, 32, 3)
        self.convT1_4 = torch.nn.Conv1d(32, 8, 3)
        self.convT1_5 = torch.nn.Conv1d(8, 1, 3)
        self.fccT1 = torch.nn.Linear(490,1)

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[1]
        x_T1 = torch.sigmoid(self.convT1_1(x))
        x_T1 = torch.sigmoid(self.convT1_2(x_T1))
        x_T1 = torch.sigmoid(self.convT1_3(x_T1))
        x_T1 = torch.sigmoid(self.convT1_4(x_T1))
        x_T1 = torch.sigmoid(self.convT1_5(x_T1))
        x_T1 = self.fccT1(x_T1)
        return x_T1

class ComplexReg(nn.Module):
    def __init__(self, input_size=500):
        super(ComplexReg, self).__init__()
        self.convT1_1 = torch.nn.Conv1d(2, 1, 32)
        self.convT1_2 = torch.nn.Conv1d(2, 1, 32, 2)
        self.convT1_3 = torch.nn.Conv1d(2, 1, 32, 4)
        self.convT1_4 = torch.nn.Conv1d(2, 1, 32, 8)
        self.convT1_5 = torch.nn.Conv1d(2, 1, 32, 16)
        self.fccT1 = torch.nn.Linear(911,1)

        self.convT2_1 = torch.nn.Conv1d(2, 1, 32)
        self.convT2_2 = torch.nn.Conv1d(2, 1, 32, 2)
        self.convT2_3 = torch.nn.Conv1d(2, 1, 32, 4)
        self.convT2_4 = torch.nn.Conv1d(2, 1, 32, 8)
        self.convT2_5 = torch.nn.Conv1d(2, 1, 32, 16)
        self.fccT2 = torch.nn.Linear(911,1)

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[1]
        x_T1 = torch.cat((self.convT1_1(x), self.convT1_2(x), 
                          self.convT1_3(x), self.convT1_4(x), 
                          self.convT1_5(x)), 2)
        x_T1 = torch.sigmoid(x_T1)

        x_T2 = torch.cat((self.convT2_1(x), self.convT2_2(x), 
                    self.convT2_3(x), self.convT2_4(x), 
                    self.convT2_5(x)), 2)
        x_T2 = torch.sigmoid(x_T2)

        x_T1 = self.fccT1(x_T1)
        x_T2 = self.fccT1(x_T2)
        
        return torch.cat((x_T1,x_T2),1)

class ComplexRNN(nn.Module):
    def __init__(self):
        super(ComplexRNN, self).__init__()
        self.rnn_T1 = nn.RNN(
            input_size=500,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        
        self.rnn_T2 = nn.RNN(
            input_size=500,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fcc_T1 = torch.nn.Linear(64,1)
        self.fcc_T2 = torch.nn.Linear(64,1)

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[2]

        x_T1, state = self.rnn_T1(x)
        x_T1 = self.fcc_T1(x_T1.contiguous().view(batch_size,-1))

        x_T2, state = self.rnn_T2(x)
        x_T2 = self.fcc_T2(x_T2.contiguous().view(batch_size,-1))
        
        return torch.cat((x_T1,x_T2),1)

        
