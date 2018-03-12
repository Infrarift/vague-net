import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad, _Loss


class VagueLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(VagueLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, x, y, w):
        _assert_no_grad(y)
        return self._vague_loss(x, y, w, size_average=self.size_average, reduce=self.reduce)

    def _vague_loss(self, x, y, w, size_average=True, reduce=True):
        return self._pointwise_loss(self.loss_function, x, y, w, size_average, reduce)

    def loss_function(self, x, y, w):
        return ((x - y) ** 2) + (w ** 2)

    @staticmethod
    def _pointwise_loss(action, x, y, w, size_average=True, reduce=True):
        d = action(x, y, w)
        if not reduce:
            return d
        return torch.mean(d) if size_average else torch.sum(d)


class VagueNet(nn.Module):

    def __init__(self):
        super(VagueNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

        self.v1 = nn.Linear(3, 1)
        self.v1.weight.data.fill_(1.0)
        self.v1.bias.data.zero_()

        pass

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        y = self.v1(x)
        x = self.fc3(x)

        return x, y


