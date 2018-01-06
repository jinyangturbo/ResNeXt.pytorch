# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

#import InplaceMul
'''
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

class DropCombine(nn.Module):
    def __init__(self, channels, res_drop = 0., p = 0.):
        super(DropCombine, self).__init__()
        self.p = p
        self.res_drop = res_drop
        self.channels = channels
        self.fix_prob = torch.FloatTensor(1, self.channels).fill_(1-self.res_drop).cuda()
        self.fix_mask = torch.bernoulli(self.fix_prob).cuda()
        self.one_mask = torch.FloatTensor(1, self.channels).fill_(1).cuda()
        self.x_prob = torch.FloatTensor(1, self.channels).fill_(1-self.p).cuda()
        self.x_mask = torch.FloatTensor(1, self.channels).fill_(0).cuda()
        # print(self.p)

    def forward(self, res, x):
        view_size = [1, self.channels] + [1] * (len(x.size()) - 2)
        """
        if self.training==True:
            # compute the residual of dropout
            if self.p > 0.:
                self.x_mask = torch.bernoulli(self.x_prob) / (1. - self.p) - self.one_mask
                self.x_op = Variable(self.x_mask.view(view_size).expand_as(x)).cuda()
                
                res = res + x * self.x_op
        if self.res_drop > 0.:
            self.fix_op = Variable(self.fix_mask.view(view_size).expand_as(x)).cuda()
            res = res * self.fix_op 
        """
        self.fix_op = Variable(self.fix_mask.view(view_size), requires_grad=False).cuda()
        #if self.training==False: return res * self.fix_op + x
        self.x_mask = (torch.bernoulli(self.x_prob) / (1. - self.p) - self.one_mask) * self.fix_mask + self.one_mask
        if self.training==False: self.x_mask = self.one_mask
        self.x_op = Variable(self.x_mask.view(view_size), requires_grad=False).cuda()
        return res * self.fix_op + x * self.x_op
        '''
        self.x_mask = (torch.bernoulli(self.x_prob) / (1. - self.p) - self.one_mask) * self.fix_mask + self.one_mask
        #return res.data.mul_(self.fix_mask.view(view_size)) + x.data.mul_(self.x_mask.view(view_size))).cuda()
        return InplaceMul.apply(res,self.fix_mask) + InplaceMul.apply(x,self.x_mask)
        '''

class SFDropoutLayer(nn.Module):
    def __init__(self, in_planes, p):
        super(SFDropoutLayer, self).__init__()
        assert p < 1.
        self.p = p
        self.in_planes = in_planes
        self.prob_tensor = torch.FloatTensor(1).fill_(1-self.p).expand((self.in_planes)).cuda()
        # print(self.p)

    def forward(self, x):
        if self.training==False: return x
        # batch shared dropout mask
        self.mask = torch.bernoulli(self.prob_tensor)
        view_size = [1, self.in_planes] + [1] * (len(x.size()) - 2)
        self.input_mask = Variable((self.mask / (1. - self.p)).view(view_size).expand_as(x)).cuda()
        return x*self.input_mask


class GroupAttDrop(nn.Module):
    def __init__(self, in_planes, cardinality, group_width, is_drop = True):
        super(GroupAttDrop, self).__init__()
        # Select layers
        D = cardinality * group_width
        self.is_drop = is_drop
        self.cardinality = cardinality
        self.group_width = group_width
        self.fc1 = nn.Conv2d(in_planes, D//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(D//16, cardinality, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expand = ExpandConv(cardinality, group_width)
        #self.mask_tensor = torch.FloatTensor(1).fill_(1-self.p).expand((self.in_planes))
        
    def forward(self, x):
        self.w1 = self.avg_pool(x)
        self.w2 = F.relu(self.fc1(self.w1))
        self.w3 = F.sigmoid(self.fc2(self.w2))
        #if self.is_drop == True:
        #    w += Variable(torch.bernoulli(w.data) - w.data)
        #print(w.size())
        #wait = input("PRESS ENTER TO CONTINUE.")
        self.wid = self.expand(self.w3)
        return self.wid
        #if self.training==False: 

class GroupRandDrop(nn.Module):
    def __init__(self, cardinality, group_width, p = 0.5, val = 0.3):
        super(GroupRandDrop, self).__init__()
        assert p < 1.
        self.p = p
        self.val = val
        self.cardinality = cardinality
        self.group_width = group_width
        self.expand = ExpandConv(cardinality, group_width)
        self.prob_tensor = torch.FloatTensor(1,cardinality).fill_(1.-self.p).cuda()
        self.one_tensor = torch.FloatTensor(1,cardinality).fill_(1.).cuda()
        # print(self.p)

    def forward(self,x):
        if self.training==False: return x
        # batch shared dropout mask
        self.mask = torch.bernoulli(self.prob_tensor)
        self.mask = self.val * self.mask / (1.- self.p) + (1.-self.val) * self.one_tensor
        self.wid_mask = self.expand(Variable(self.mask, requires_grad=False))
        return x * self.wid_mask 
        
class ExpandConv(nn.Module):
    def __init__(self, cardinality, group_width):
        super(ExpandConv, self).__init__()
        self.D = cardinality * group_width
        self.cardinality = cardinality
        self.group_width = group_width
        self.one_tensor = Variable(torch.ones(1,1,self.group_width), requires_grad=False).cuda()
        #self.one_tensor = torch.FloatTensor(1,1,self.group_width).fill_(1.).cuda()
        
    def forward(self, x):
        self.wid = torch.matmul(x.view(-1, self.cardinality,1),self.one_tensor.expand(x.size(0),-1,-1))
        self.wid = self.wid.view(-1, self.D, 1, 1)
        return self.wid
    


class DropCombineBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, res_drop = 0., p = 0., preact = False):
        super(DropCombineBottleneck, self).__init__()
        self.layer = ResNeXtBottleneck(in_channels, out_channels, stride, cardinality, base_width, widen_factor, res_drop = 0.05, p = 0.2, preact = preact)

    def forward(self, x):
        return self.layer(x)   
    
    
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, res_drop = 0., p = 0., preact = False):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        #width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * base_width
        self.preact = preact
        self.pre_bn = nn.BatchNorm2d(in_channels)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))
        self.combine = DropCombine(out_channels, res_drop, p)
        #self.sfdrop = SFDropoutLayer(out_channels, p)
        #self.channeldrop = nn.Dropout2d(p)

    def forward(self, x):
        '''
        #for pre-activation residual
        bottleneck = F.relu(self.pre_bn(x))
        bottleneck = self.conv_reduce(bottleneck)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        #bottleneck = self.bn_expand(bottleneck)
        #return self.combine(self.shortcut(x), bottleneck)
        #return self.shortcut(x) + self.sfdrop(bottleneck)
        #return self.shortcut(x) + self.channeldrop(bottleneck)
        return self.shortcut(x) + bottleneck
        '''
        if self.preact == True: bottleneck = F.relu(self.pre_bn(x))
        else: bottleneck = x
        bottleneck = self.conv_reduce(bottleneck)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        if self.preact == False: bottleneck = self.bn_expand(bottleneck)     
        #out = self.shortcut(x) + bottleneck
        out = self.combine(self.shortcut(x), bottleneck)
        if self.preact == False: out = F.relu(out, inplace=True)
        return out

    
class DropNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(DropNeXtBottleneck, self).__init__()
        #width_ratio = out_channels / (widen_factor * 64.)
        self.group_width = base_width
        D = cardinality * self.group_width 
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.groupdrop = GroupRandDrop(cardinality,self.group_width)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.groupdrop(bottleneck)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)

class SENeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(SENeXtBottleneck, self).__init__()
        #width_ratio = out_channels / (widen_factor * 64.)
        self.cardinality = cardinality
        self.group_width = base_width
        D = self.cardinality * self.group_width
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        # Select layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(D, D//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(D//16, cardinality, kernel_size=1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)       
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        # Squeeze
        w = self.avg_pool(bottleneck)
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Expand
        wid = GroupAttAvg(w,self.cardinality,self.group_width)
        #print(wid.size())
        #print(bottleneck.size())
        #wait = input("PRESS ENTER TO CONTINUE.")
        bottleneck = bottleneck * wid
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)  
    
class SelNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(SelNeXtBottleneck, self).__init__()
        #width_ratio = out_channels / (widen_factor * 64.)
        self.cardinality = cardinality
        self.group_width = base_width
        D = self.cardinality * self.group_width
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.select = GroupAttDrop(self.cardinality * self.group_width, self.cardinality, self.group_width)
        # Select layers
        #self.fc1 = nn.Conv2d(D, D//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        #self.fc2 = nn.Conv2d(D//16, cardinality, kernel_size=1)
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))
  
    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        # Select groups
        #w = self.avg_pool(bottleneck)
        #w = F.relu(self.fc1(w))
        #w = F.sigmoid(self.fc2(w))
        #mask = GroupAttDrop(w,self.cardinality,self.group_width)
        mask = self.select(bottleneck)
        # compute groups
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        # drop groups
        bottleneck = bottleneck * mask
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True) 
    
class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, model, cardinality, depth, nlabels, base_width, widen_factor=4, band_width = 64, preact = False):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.preact = preact
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = band_width
        self.stages = [64, band_width * self.widen_factor, 2* band_width * self.widen_factor, 4 * band_width * self.widen_factor]
        model_map  = {'ResNext': ResNeXtBottleneck,
              'DropNext': DropNeXtBottleneck,
              'SENext': SENeXtBottleneck,
              'DropCombine' : DropCombineBottleneck,
              'SelNext': SelNeXtBottleneck}
        self.Bottleneck = model_map[model]
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1, 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2, 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 4, 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, width_ratio, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, self.Bottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width * width_ratio, self.widen_factor,preact = self.preact))
            else:
                block.add_module(name_,
                                 self.Bottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width * width_ratio,
                                                    self.widen_factor,preact = self.preact))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)
