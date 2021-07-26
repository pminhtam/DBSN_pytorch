from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_net import DBSN_branch
from utils.utils import init_weights, weights_init_kaiming


class DBSN_Model(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch,
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num,
                br2_blindspot_conv_ks, br2_block_num,
                activate_fun):
        super(DBSN_Model,self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        # Head of DBSN
        lyr = []
        lyr.append(nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=blindspot_conv_bias))
        lyr.append(self.relu())
        self.dbsn_head = nn.Sequential(*lyr)
        init_weights(self.dbsn_head)

        self.br1 = DBSN_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
        self.br2 = DBSN_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)

        # Concat two branches
        self.concat = nn.Conv2d(mid_ch*2,mid_ch,kernel_size=1,bias=blindspot_conv_bias)
        self.concat.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=blindspot_conv_bias))
        self.dbsn_tail=nn.Sequential(*lyr)
        init_weights(self.dbsn_tail)

    def forward(self, x):
        x = self.dbsn_head(x)
        x1 = self.br1(x)
        x2 = self.br2(x)
        x_concat = torch.cat((x1,x2), dim=1)
        x = self.concat(x_concat)
        return self.dbsn_tail(x), x



