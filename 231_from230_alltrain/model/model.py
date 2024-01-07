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


def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Model:
    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0, load_path=None, training=True):
        self.dmvfn = DMVFN()
        self.optimG = AdamW(self.dmvfn.parameters(), lr=1e-6, weight_decay=1e-3)
        self.lap = LapLoss()
        self.vggloss = VGGPerceptualLoss()
        self.encoder_target = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 8, 4, 2, 1)
        )
        self.device()
        if training:
            if local_rank != -1:
                self.dmvfn = DDP(self.dmvfn, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
                self.encoder_target = DDP(self.encoder_target, device_ids=[local_rank], output_device=local_rank)
            hard_update(self.encoder_target, self.dmvfn.module.encoder)
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
        b, num_total, c, h, w = imgs.shape  # 8, [9 for kitti, 7 for vimeo, 20 for MovingMNIST], 3, 256, 256
        num_vis = num_total - 2
        loss_avg = 0
        cnt = 0
        for i in range(2, num_vis):
            imgn2, imgn1 = imgs[:, i - 2], imgs[:, i - 1]
            img0, img1, img2 = imgs[:, i], imgs[:, i + 1], imgs[:, i + 2]
            for t_pred in range(2, min(num_vis, num_total - i)):
                # t_pred = random.randint(2, min(num_vis, num_total - i) - 1)
                merged_ls = self.dmvfn(
                    torch.cat((imgn2, imgn1, img0, img1, img2), 1), t_pred,
                    scale=[4, 4, 4, 2, 2, 2, 1, 1, 1],
                )   # (9)(B, C, H, W)
                loss_G = 0.0
                loss_l1, loss_vgg = 0, 0
                gt = imgs[:, i + t_pred]
                k = 2
                for j in range(9):
                    if j < k:
                        loss_l1 += (self.lap(merged_ls[j], img2)).mean() * (0.8 ** (8 - j))
                    else:
                        loss_l1 += (self.lap(merged_ls[j], gt)).mean() * (0.8 ** (8 - j))
                loss_vgg += (self.vggloss(merged_ls[-1], gt)).mean()
                loss_vgg += (self.encoder_target(merged_ls[-1]) - self.encoder_target(gt)).abs().mean()
                self.optimG.zero_grad()
                loss_G += loss_l1 + loss_vgg * 0.5
                loss_avg += loss_G
                loss_G.backward()
                self.optimG.step()
                hard_update(self.encoder_target, self.dmvfn.module.encoder)
                cnt += 1
        return loss_avg / cnt

    def eval(self, imgs, name, scale_list=[4, 4, 4, 2, 2, 2, 1, 1, 1]):
        self.dmvfn.eval()
        b, num_total, c, h, w = imgs.shape 
        preds = []
        if name == 'KittiValDataset':
            assert num_total == 7
            start_ind = 2
        elif name == 'VimeoValDataset':
            assert num_total == 7
            start_ind = 2
        elif name == 'MovingMNISTValDataset':
            assert num_total == 20
            start_ind = 8
        num_vis = num_total - start_ind
        imgn2, imgn1 = imgs[:, start_ind - 2], imgs[:, start_ind - 1]
        img0, img1, img2 = imgs[:, start_ind], imgs[:, start_ind + 1], imgs[:, start_ind + 2]
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
        self.encoder_target.to(device)

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.dmvfn.state_dict(),'{}/dmvfn_{}.pkl'.format(path, str(epoch)))
