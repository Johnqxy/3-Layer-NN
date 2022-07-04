import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden=nn.Linear(2,3)
        self.relu=nn.ReLU()
        self.out=nn.Linear(3,1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        x=self.hidden(x)
        x=self.relu(x)
        x=self.out(x)
        x=self.sigmoid(x)
        return x