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


class SimpleClass(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleClass, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(8)
        self.conv1 = torch.nn.Conv1d(2, 16, 3)
        self.conv2 = torch.nn.Conv1d(16, 64, 3)
        self.conv3 = torch.nn.Conv1d(64, 512, 3)
        self.conv4 = torch.nn.Conv1d(512, 1024, 3)
        
        self.fc1 = nn.Linear(55296, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[-1]
        x = x.view(batch_size, -1, input_size)
        
        x = self.avgpool(x)
        x_T1 = F.relu(self.conv1(x))
        x_T1 = F.relu(self.bn1(self.conv2(x_T1)))
        x_T1 = F.relu(self.bn2(self.conv3(x_T1)))
        x_T1 = F.relu(self.bn3(self.conv4(x_T1)))

        x_T1 = x_T1.view(batch_size, -1)

        x_T1 = F.relu(self.bn4(self.dropout(self.fc1(x_T1))))
        x_T1 = F.relu(self.bn5(self.dropout(self.fc2(x_T1))))
        x_T1 = self.fc3(x_T1)
        
        return x_T1

class SimpleReg(nn.Module):
    def __init__(self):
        super(SimpleReg, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(8)
        self.conv1 = torch.nn.Conv1d(2, 16, 3)
        self.conv2 = torch.nn.Conv1d(16, 64, 3)
        self.conv3 = torch.nn.Conv1d(64, 512, 3)
        self.conv4 = torch.nn.Conv1d(512, 1024, 3)
        
        self.fc1 = nn.Linear(55296, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        

    def forward(self, x):
        batch_size = x.size()[0]
        input_size = x.size()[1]
        x = x.view(batch_size, -1, input_size)
        
        x = self.avgpool(x)
        x_T1 = F.relu(self.conv1(x))
        x_T1 = F.relu(self.bn1(self.conv2(x_T1)))
        x_T1 = F.relu(self.bn2(self.conv3(x_T1)))
        x_T1 = F.relu(self.bn3(self.conv4(x_T1)))

        x_T1 = x_T1.view(batch_size, -1)

        x_T1 = F.relu(self.bn4(self.dropout(self.fc1(x_T1))))
        x_T1 = F.relu(self.bn5(self.dropout(self.fc2(x_T1))))
        x_T1 = self.fc3(x_T1)
        
        return x_T1