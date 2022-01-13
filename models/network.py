# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement

": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.6

Authors: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''
import torch.fft
import torch
import torch.nn as nn
from models import basicblock as B
from models import ns_model
from models import utv_model
from models import nli_model
from models import lc_model


class UTVNet(nn.Module):
    def __init__(self):
        super(UTVNet, self).__init__()
        self.a = utv_model.ADMM(1, 8, 1)
        self.noiselevel = nli_model.IRCNN(3, 24, 32)
        self.denoise = ns_model.UNet()
        self.LIGHT = lc_model.LIRCNN(3, 3, 48)
        self.device = torch.device('cuda')
        self.hyp = B.HyPaNet()

    def forward(self, x):
        level01, level02, level03, level = self.noiselevel(x)
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        level1 = level01
        level2 = level02
        level3 = level03
        levelsam = level
        smoothr = self.a(r, level1.squeeze(0)).unsqueeze(0).unsqueeze(0)
        smoothg = self.a(g, level2.squeeze(0)).unsqueeze(0).unsqueeze(0)
        smoothb = self.a(b, level3.squeeze(0)).unsqueeze(0).unsqueeze(0)
        smooth1 = torch.cat((smoothr, smoothg, smoothb), 1)
        denoise = self.denoise(x - smooth1, levelsam)
        smooth = self.LIGHT(smooth1)
        out = denoise + smooth
        return out


def make_model(args, parent=False):
    return UTVNet()
