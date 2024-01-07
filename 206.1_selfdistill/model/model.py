import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from model.arch import *
from loss.loss import *
import random


device = torch.device("cuda")


class Model:
    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0, load_path=None, training=True):
        self.dmvfn = DMVFN()
        self.optimG = AdamW(self.dmvfn.parameters(), lr=1e-6, weight_decay=1e-3)
        self.lap = LapLoss()
        self.vggloss = VGGPerceptualLoss()
        self.device()
        if training:
            if local_rank != -1:
                self.dmvfn = DDP(self.dmvfn, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            if resume_path is not None:
                assert resume_epoch>=1
                print(local_rank,": loading...... ", '{}/dmvfn_{}.pkl'.format(resume_path, str(resume_epoch-1)))
                self.dmvfn.load_state_dict(torch.load('{}/dmvfn_{}.pkl'.format(resume_path, str(resume_epoch-1))), strict=True)
            else:
                if load_path is not None:
                    self.dmvfn.load_state_dict(torch.load(load_path), strict=True)
        else:
            state_dict = torch.load(load_path)
            model_state_dict = self.dmvfn.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict['module.'+k]
            self.dmvfn.load_state_dict(model_state_dict)

    def train(self, imgs, learning_rate=0):
        self.dmvfn.train()
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        b, num_total, c, h, w = imgs.shape  # 8, [9 for kitti, 7 for vimeo], 3, 256, 256
        num_vis = num_total - 2
        loss_avg = 0
        for i in range(2, num_total - 2):
            imgn2, imgn1 = imgs[:, i - 2], imgs[:, i - 1]
            img0, img1, img2 = imgs[:, i], imgs[:, i + 1], imgs[:, i + 2]
            t_pred = random.randint(2, min(num_vis, num_total - i) - 1)
            merged_ls, distill_flow_ls = self.dmvfn(
                torch.cat((imgn2, imgn1, img0, img1, img2), 1), t_pred,
                scale=[4, 4, 4, 2, 2, 2, 1, 1, 1],
            )   # (9)(B, C, H, W)
            loss_G = 0.0
            loss_l1, loss_vgg = 0, 0
            loss_distill = 0
            gt = imgs[:, i + t_pred]
            for j in range(9):
                loss_l1 += (self.lap(merged_ls[j], gt)).mean() * (0.8 ** (8 - j))
                loss_distill += F.l1_loss(distill_flow_ls[i][0], distill_flow_ls[i][1])
            loss_vgg += (self.vggloss(merged_ls[-1], gt)).mean()
            self.optimG.zero_grad()
            loss_G += loss_l1 + loss_vgg * 0.5 + loss_distill * 0.001
            loss_avg += loss_G
            loss_G.backward()
            self.optimG.step()
        return loss_avg / (num_total - 2 - 2)

    def eval(self, imgs, name, scale_list=[4, 4, 4, 2, 2, 2, 1, 1, 1]):
        self.dmvfn.eval()
        b, num_total, c, h, w = imgs.shape 
        preds = []
        if name == 'KittiValDataset':
            assert num_total == 7
        elif name == 'VimeoValDataset':
            assert num_total == 7
        elif name == 'VimeoValDataset_wo01':
            assert num_total == 5
        num_vis = num_total - 2
        imgn2, imgn1 = imgs[:, 0], imgs[:, 1]
        img0, img1, img2 = imgs[:, 2], imgs[:, 3], imgs[:, 4]
        for t_pred in range(2, num_vis):
            merged_ls = self.dmvfn(
                torch.cat((imgn2, imgn1, img0, img1, img2), 1), t_pred,
                scale=[4, 4, 4, 2, 2, 2, 1, 1, 1],
                training=False
            )   # (9)(B, C, H, W)
            preds.append(merged_ls[-1])
        return torch.stack(preds, 1)

    def device(self):
        self.dmvfn.to(device)
        self.lap.to(device)
        self.vggloss.to(device)

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.dmvfn.state_dict(),'{}/dmvfn_{}.pkl'.format(path, str(epoch)))
