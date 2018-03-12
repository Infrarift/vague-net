import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class VagueNet(nn.Module):

    def __init__(self):
        super(VagueNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
