"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.softsplat import FunctionSoftsplat


def warp_pwc(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    
    """
    B, C, H, W = x.size()
    
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).to(device)
    vgrid = Variable(grid).to(device) + flo
    
    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    
    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)        
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask


def forwardwarp(
    tenInput,   # (B, C, H, W)  # C == 3
    tenFlow,    # (B, 2, H, W)
    d=None      # None
):
    tenFlow = tenFlow[:, : 2].contiguous()  # (B, 2, H, W)
    if d == None:
        tenSoftmax = FunctionSoftsplat(tenInput=tenInput, tenFlow=tenFlow, tenMetric=d, strType='average')
    else:
        tenSoftmax = FunctionSoftsplat(tenInput=tenInput, tenFlow=tenFlow, tenMetric=d, strType='linear')
    return tenSoftmax   # (B, C, H, W)


def warp(tenInput, tenFlow):
    backwarp_tenGrid = {}
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(
            -1.0, 1.0, tenFlow.shape[3]
        ).view(
            1, 1, 1, tenFlow.shape[3]
        ).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1
        )
        tenVertical = torch.linspace(
            -1.0, 1.0, tenFlow.shape[2]
        ).view(
            1, 1, tenFlow.shape[2], 1
        ).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3]
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)
    tenFlow = torch.cat(
        [
            tenFlow[:, 0: 1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1: 2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
        ], 1
    )
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True
    )
