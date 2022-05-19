import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size = (64, 250)):
        super(Net, self).__init__()

        # hyper parameters
        self.C = input_size[0] # Channel number
        self.K = 50 # kernel size  
        self.F = 16 # number of filters

        # activation function and dropout
        self.activate = nn.ELU()
        self.dropout = nn.Dropout(p = 0.5)

        # output feature size 
        self.F0 = input_size[1]
        self.F1 = int((self.F0 - self.K) / 3 + 1) 
        self.F2 = int((self.F1 - self.K) / 3 + 1) 
        self.F3 = int(self.F2 - 3)
        self.F4 = int(self.F * self.F3)

        # hidden layer parameter
        self.F5 = 512
    
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = self.F, kernel_size = self.K, stride = 3),
            nn.BatchNorm1d(self.F),
            self.activate
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = self.F, out_channels = self.F, kernel_size = self.K, stride = 3),
            nn.BatchNorm1d(self.F),
            self.activate
        )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size = 4, stride = 1),
            nn.Flatten()
        )

        # self attentation 
        self.Q = nn.Linear(self.F4, self.C)
        self.K = nn.Linear(self.F4, self.C)
        self.V = nn.Linear(self.F4, self.C)
        self.softmax = nn.Softmax(dim = -1)


        self.tanh = nn.Tanh()
        # hiddden layer of NN
        self.FC1 = nn.Sequential(
            nn.Linear(self.C * self.C, self.F5),
            self.activate
        )
        self.FC2 = nn.Linear(self.F5, 4) 


    def forward(self, x) :
        
        if len(x.shape) == 2 :
            x = x.unsqueeze(0)
        batchsize = x.size()[0]

        # conv 1
        x = x.reshape((batchsize * self.C, 1, self.F0))
        x = self.conv1(x)
        
        # conv 2
        x = self.conv2(x)
        
        # maxpool 1
        x = self.maxpool1(x)
        x = x.reshape((batchsize, self.C, self.F4))
    
        # self attentation     
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        w = self.softmax(torch.bmm(q, k.permute(0, 2, 1)))
        m = torch.bmm(w, v)
        
        x = self.tanh(torch.bmm(m, w.permute(0, 2, 1)))
        x = nn.Flatten()(x)
        
        # hidden FC layers
        x = self.FC1(x)
        x = self.FC2(x)

        return x

