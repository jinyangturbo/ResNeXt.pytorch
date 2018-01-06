import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

from torch.autograd.function import InplaceFunction

class InplaceMul(InplaceFunction):
    @staticmethod
    def forward(cls, ctx, input, multiplier):
        ctx.mark_dirty(input)
        ctx.multiplier = multiplier
        output = input
        view_size = [1, input.size(1)] + [1] * (len(input.size()) - 2)
        output.mul_(ctx.multiplier.view(view_size).expand_as(output))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        view_size = [1, grad_output.size(1)] + [1] * (len(grad_output.size()) - 2)
        return grad_output.mul_(ctx.multiplier.view(view_size).expand_as(grad_output))
        
'''
class InMul2d(nn.Module):

    def __init__(self, multiplier):
        super(InMul2d, self).__init__()

    def forward(self, input, multiplier):
        return InplaceMul(input, multiplier)

'''