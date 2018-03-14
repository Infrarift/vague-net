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
            return first_loss * m

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

        self.networks = []
        self.params = []

        # First network
        self.create_network(2, (4, 4, 4, 2))

        # Second network
        self.create_network(2, (4, 4, 4, 2))

        # self.s = nn.Sigmoid()

    def create_network(self, start_size=2, layer_list=None):

        network = []
        params = []
        previous_size = start_size

        for i in range(len(layer_list)):
            layer = nn.Linear(previous_size, layer_list[i])
            previous_size = layer_list[i]
            network.append(layer)
            self.add_params(params, layer)

        self.networks.append(network)
        self.params.append(params)
        return network, params

    @staticmethod
    def add_params(param_list, nn_module):
        for p in nn_module.parameters():
            param_list.append(p)

    def forward(self, x):
        p = self.net_forward(x, self.networks[0])
        c = self.net_forward(x, self.networks[1])
        return p, c

    def net_forward(self, x, layers):
        for layer in layers:
            x = layer(x)
            if layer == layers[-2]:
                x = F.tanh(x)
        return x
