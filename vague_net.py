import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad, _Loss


class VagueLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(VagueLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, x, y, m, a):
        _assert_no_grad(y)
        return self._vague_loss(x, y, m, a, size_average=self.size_average, reduce=self.reduce)

    def _vague_loss(self, x, y, m, a, size_average=True, reduce=True):
        return self._pointwise_loss(self.loss_function, x, y, m, size_average, reduce)

    def loss_function(self, x, y, m, a):
        first_loss = self.vanilla_loss(x, y)
        if m is None:
            return first_loss
        else:
            return first_loss * m + first_loss * a

    def vanilla_loss(self, x, y):
        return (x - y) ** 2

    @staticmethod
    def _pointwise_loss(action, x, y, m, a, size_average=True, reduce=True):
        d = action(x, y, m, a)
        if not reduce:
            return d
        return torch.mean(d) if size_average else torch.sum(d)


class VagueNet(nn.Module):

    def __init__(self):
        super(VagueNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

        self.fc4 = nn.Linear(2, 3)
        self.fc5 = nn.Linear(3, 3)
        self.fc6 = nn.Linear(3, 1)
        self.s = nn.Sigmoid()

        self.n1_params = []
        self.add_params(self.n1_params, self.fc1)
        self.add_params(self.n1_params, self.fc2)
        self.add_params(self.n1_params, self.fc3)

        self.n2_params = []
        self.add_params(self.n2_params, self.fc4)
        self.add_params(self.n2_params, self.fc5)
        self.add_params(self.n2_params, self.fc6)

    @staticmethod
    def add_params(param_list, nn_module):
        for p in nn_module.parameters():
            param_list.append(p)

    def forward(self, x):

        p = self.fc1(x)
        p = self.fc2(p)
        p = self.fc3(p)
        p = self.s(p)

        c = self.fc4(x)
        c = self.fc5(c)
        c = self.fc6(c)
        c = self.s(c)

        return p, c


