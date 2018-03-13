import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad, _Loss


class GanNet(nn.Module):
    def __init__(self):
        super(GanNet, self).__init__()

